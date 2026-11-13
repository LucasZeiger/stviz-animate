use anyhow::{anyhow, Context};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    mem::{align_of, size_of},
    path::Path,
    sync::Arc,
};

const STVIZ_MAGIC: [u8; 8] = *b"STVIZ\0\0\0";
const STVIZ_VERSION: u32 = 1;
const HEADER_BYTES_WITH_MAGIC: usize = 8 + 4 + 8;
const LEGACY_HEADER_BYTES: usize = 8;

/// Saved scenes live in an optional block after the data section:
///
/// ```text
/// [header][metadata JSON][pad][data ...][scenes JSON + space padding][footer]
/// footer = block_len: u64 LE | scenes version: u32 LE | b"STVZSCN\0"
/// ```
///
/// Readers that don't know about scenes stop at the end of the data section,
/// so files with scenes still open in older versions of the app. Adding or
/// removing a scene rewrites only this block, never the arrays.
const SCENES_MAGIC: [u8; 8] = *b"STVZSCN\0";
const SCENES_VERSION: u32 = 1;
const SCENES_FOOTER_BYTES: usize = 8 + 4 + 8;
/// Refuse absurd scene blocks rather than allocating for them.
const MAX_SCENES_BYTES: u64 = 64 << 20;

#[derive(Debug, Deserialize, Clone)]
pub struct SpaceMeta {
    pub name: String,
    pub dims: u32,
    pub offset: u64,
    pub len_bytes: u64,
    /// [min_x, min_y, max_x, max_y] for dims==2
    pub bbox: [f32; 4],
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "kind")]
pub enum ObsMeta {
    #[serde(rename = "categorical")]
    Categorical {
        name: String,
        offset: u64,
        len_bytes: u64,
        categories: Vec<String>,
        /// Optional per-category palette (u32 packed rgba8)
        palette_rgba8: Option<Vec<u32>>,
    },
    #[serde(rename = "continuous")]
    Continuous {
        name: String,
        offset: u64,
        len_bytes: u64,
    },
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct ExprMeta {
    pub kind: String, // "csc"
    pub n_genes: u32,
    pub nnz: u64,
    pub var_names: Vec<String>,
    pub indptr_offset: u64,
    pub indptr_len_bytes: u64,
    pub indices_offset: u64,
    pub indices_len_bytes: u64,
    pub data_offset: u64,
    pub data_len_bytes: u64,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct StvizMeta {
    pub version: u32,
    pub n_points: u32,
    /// The file this was converted from, when the converter recorded it.
    #[serde(default)]
    pub source: Option<String>,
    pub spaces: Vec<SpaceMeta>,
    pub obs: Vec<ObsMeta>,
    pub expr: Option<ExprMeta>,
    /// Cell names (the .h5ad obs index), when the converter recorded them.
    #[serde(default)]
    pub obs_names: Option<Vec<String>>,
}

/// The scenes saved in a .stviz file. Each scene is kept as raw JSON so that
/// scenes this version can't read (e.g. saved by a newer version) survive
/// when other scenes are added or deleted.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SavedScenes {
    pub scenes: Vec<serde_json::Value>,
    /// Why existing scene data could not be read, if it couldn't.
    pub warning: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct ScenesBlock {
    version: u32,
    #[serde(default)]
    scenes: Vec<serde_json::Value>,
}

/// Location of an existing scenes block, from the file's footer.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScenesFooter {
    block_start: u64,
    block_len: u64,
    version: u32,
}

pub struct Dataset {
    mmap: Mmap,
    data_start: usize,
    /// End of the last data block (absolute file offset); scenes start at or after it.
    data_end: u64,
    pub meta: StvizMeta,
    gene_index: Option<HashMap<String, u32>>,
    /// Scenes stored in the file when it was opened.
    pub saved_scenes: SavedScenes,
}

impl Dataset {
    pub fn load(path: &Path) -> anyhow::Result<Arc<Self>> {
        let file = File::open(path).with_context(|| format!("open: {}", path.display()))?;
        if file.metadata().map(|m| m.len() == 0).unwrap_or(false) {
            return Err(anyhow!("the file is empty: {}", path.display()));
        }
        // Safety: the mapping is read-only. The converter never modifies a
        // .stviz in place (it writes a new file and renames it over the old
        // one). Saving scenes (`write_scenes`) only writes the scenes block
        // after the data section, which is read once here and never through
        // the mapping afterwards, so the bytes we hand out do not change.
        let mmap = unsafe { Mmap::map(&file).context("mmap")? };

        let (json_start, json_len, header_version) = Self::parse_header(&mmap)?;
        let json_end = Self::checked_add_usize(json_start, json_len, "json end")?;
        if json_end > mmap.len() {
            return Err(anyhow!("invalid json length in header"));
        }

        let meta: StvizMeta = serde_json::from_slice(&mmap[json_start..json_end]).context(
            "the file's metadata could not be read (it may be damaged or from an \
             incompatible converter; try converting the .h5ad again)",
        )?;
        if meta.version != STVIZ_VERSION {
            return Err(anyhow!(
                "unsupported .stviz metadata version {} (expected {})",
                meta.version,
                STVIZ_VERSION
            ));
        }
        if let Some(v) = header_version {
            if v != meta.version {
                return Err(anyhow!(
                    "header version ({v}) does not match metadata version ({})",
                    meta.version
                ));
            }
        }

        let pad = (16 - (json_end % 16)) % 16;
        let data_start = Self::checked_add_usize(json_end, pad, "data start")?;
        if data_start > mmap.len() {
            return Err(anyhow!("invalid padding/data_start"));
        }

        let mut gene_index = None;
        if let Some(expr) = meta.expr.as_ref() {
            if expr.var_names.len() > u32::MAX as usize {
                return Err(anyhow!("too many genes for u32 indexing"));
            }
            let mut m = HashMap::with_capacity(expr.var_names.len());
            for (i, g) in expr.var_names.iter().enumerate() {
                m.insert(g.clone(), i as u32);
            }
            gene_index = Some(m);
        }

        let mut ds = Self {
            mmap,
            data_start,
            data_end: 0,
            meta,
            gene_index,
            saved_scenes: SavedScenes::default(),
        };
        ds.data_end = ds.validate_layout()?;
        ds.saved_scenes = read_scenes_block(&ds.mmap, ds.data_end);

        Ok(Arc::new(ds))
    }

    /// Read only the `source` recorded in a .stviz file's metadata.
    pub fn peek_source(path: &Path) -> Option<String> {
        #[derive(Deserialize)]
        struct SourceOnly {
            #[serde(default)]
            source: Option<String>,
        }
        const MAX_META_BYTES: usize = 512 << 20;
        let mut file = File::open(path).ok()?;
        let mut header = [0u8; HEADER_BYTES_WITH_MAGIC];
        file.read_exact(&mut header).ok()?;
        let (_start, json_len, _version) = Self::parse_header(&header).ok()?;
        if !header.starts_with(&STVIZ_MAGIC) || json_len > MAX_META_BYTES {
            return None;
        }
        let mut json = vec![0u8; json_len];
        file.read_exact(&mut json).ok()?;
        serde_json::from_slice::<SourceOnly>(&json).ok()?.source
    }

    fn parse_header(mmap: &[u8]) -> anyhow::Result<(usize, usize, Option<u32>)> {
        if mmap.starts_with(&STVIZ_MAGIC) {
            if mmap.len() < HEADER_BYTES_WITH_MAGIC {
                return Err(anyhow!("file too small for .stviz header"));
            }
            let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
            if version != STVIZ_VERSION {
                return Err(anyhow!(
                    "unsupported .stviz header version {} (expected {})",
                    version,
                    STVIZ_VERSION
                ));
            }
            let json_len_u64 = u64::from_le_bytes(mmap[12..20].try_into().unwrap());
            let json_len = usize::try_from(json_len_u64)
                .map_err(|_| anyhow!("json length does not fit platform usize"))?;
            Ok((HEADER_BYTES_WITH_MAGIC, json_len, Some(version)))
        } else {
            if mmap.len() < LEGACY_HEADER_BYTES {
                return Err(anyhow!("file too small"));
            }
            let json_len_u64 = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
            let json_len = usize::try_from(json_len_u64)
                .map_err(|_| anyhow!("json length does not fit platform usize"))?;
            Ok((LEGACY_HEADER_BYTES, json_len, None))
        }
    }

    fn checked_add_usize(a: usize, b: usize, ctx: &str) -> anyhow::Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| anyhow!("overflow while computing {ctx}"))
    }

    fn checked_mul_usize(a: usize, b: usize, ctx: &str) -> anyhow::Result<usize> {
        a.checked_mul(b)
            .ok_or_else(|| anyhow!("overflow while computing {ctx}"))
    }

    fn u64_to_usize(v: u64, ctx: &str) -> anyhow::Result<usize> {
        usize::try_from(v).map_err(|_| anyhow!("{ctx} does not fit platform usize"))
    }

    /// Check a data block lies inside the file; returns its end offset.
    fn validate_range(&self, offset: u64, len: u64, ctx: &str) -> anyhow::Result<u64> {
        let off = Self::u64_to_usize(offset, &format!("{ctx} offset"))?;
        let len = Self::u64_to_usize(len, &format!("{ctx} length"))?;
        let start = Self::checked_add_usize(self.data_start, off, &format!("{ctx} start"))?;
        let end = Self::checked_add_usize(start, len, &format!("{ctx} end"))?;
        if end > self.mmap.len() {
            return Err(anyhow!("{ctx} block out of file bounds"));
        }
        Ok(end as u64)
    }

    /// Validate every block and return where the data section ends.
    fn validate_layout(&self) -> anyhow::Result<u64> {
        let mut data_end = self.data_start as u64;
        let n_points = self.meta.n_points as usize;
        if self.meta.spaces.is_empty() {
            return Err(anyhow!("dataset has no spaces"));
        }

        let expected_vec2_bytes = Self::checked_mul_usize(
            Self::checked_mul_usize(n_points, 2, "n_points * 2")?,
            size_of::<f32>(),
            "space length bytes",
        )? as u64;
        let expected_scalar_bytes =
            Self::checked_mul_usize(n_points, size_of::<u32>(), "obs length bytes")? as u64;

        for (i, s) in self.meta.spaces.iter().enumerate() {
            if s.dims != 2 {
                return Err(anyhow!("space[{i}] dims={} (expected 2)", s.dims));
            }
            if s.len_bytes != expected_vec2_bytes {
                return Err(anyhow!(
                    "space[{i}] has invalid len_bytes={} (expected {})",
                    s.len_bytes,
                    expected_vec2_bytes
                ));
            }
            if s.len_bytes % (size_of::<f32>() as u64) != 0 {
                return Err(anyhow!("space[{i}] len_bytes is not f32-aligned"));
            }
            data_end =
                data_end.max(self.validate_range(s.offset, s.len_bytes, &format!("space[{i}]"))?);
        }

        for (i, o) in self.meta.obs.iter().enumerate() {
            match o {
                ObsMeta::Categorical {
                    len_bytes,
                    offset,
                    categories,
                    palette_rgba8,
                    ..
                } => {
                    if *len_bytes != expected_scalar_bytes {
                        return Err(anyhow!(
                            "obs[{i}] categorical len_bytes={} (expected {})",
                            len_bytes,
                            expected_scalar_bytes
                        ));
                    }
                    if len_bytes % (size_of::<u32>() as u64) != 0 {
                        return Err(anyhow!("obs[{i}] categorical len_bytes is not u32-aligned"));
                    }
                    if n_points > 0 && categories.is_empty() {
                        return Err(anyhow!("obs[{i}] categorical has no categories"));
                    }
                    if let Some(p) = palette_rgba8 {
                        if p.len() != categories.len() {
                            return Err(anyhow!(
                                "obs[{i}] categorical palette length {} does not match category count {}",
                                p.len(),
                                categories.len()
                            ));
                        }
                    }
                    data_end = data_end.max(self.validate_range(
                        *offset,
                        *len_bytes,
                        &format!("obs[{i}] categorical"),
                    )?);
                }
                ObsMeta::Continuous {
                    len_bytes, offset, ..
                } => {
                    if *len_bytes != expected_scalar_bytes {
                        return Err(anyhow!(
                            "obs[{i}] continuous len_bytes={} (expected {})",
                            len_bytes,
                            expected_scalar_bytes
                        ));
                    }
                    if len_bytes % (size_of::<f32>() as u64) != 0 {
                        return Err(anyhow!("obs[{i}] continuous len_bytes is not f32-aligned"));
                    }
                    data_end = data_end.max(self.validate_range(
                        *offset,
                        *len_bytes,
                        &format!("obs[{i}] continuous"),
                    )?);
                }
            }
        }

        if let Some(expr) = self.meta.expr.as_ref() {
            data_end = data_end.max(self.validate_expr(expr, n_points)?);
        }

        if let Some(names) = self.meta.obs_names.as_ref() {
            if names.len() != n_points {
                return Err(anyhow!(
                    "obs_names has {} entries (expected {n_points})",
                    names.len()
                ));
            }
        }

        Ok(data_end)
    }

    /// Validate the expression matrix; returns the end of its last block.
    fn validate_expr(&self, e: &ExprMeta, n_points: usize) -> anyhow::Result<u64> {
        if e.kind != "csc" {
            return Err(anyhow!("unsupported expr kind '{}'", e.kind));
        }

        let n_genes = e.n_genes as usize;
        if e.var_names.len() != n_genes {
            return Err(anyhow!(
                "expr var_names length {} does not match n_genes {}",
                e.var_names.len(),
                n_genes
            ));
        }

        let nnz = Self::u64_to_usize(e.nnz, "expr nnz")?;
        let expect_indptr =
            Self::checked_mul_usize(n_genes + 1, size_of::<u32>(), "expr indptr bytes")? as u64;
        let expect_indices =
            Self::checked_mul_usize(nnz, size_of::<u32>(), "expr indices bytes")? as u64;
        let expect_data = Self::checked_mul_usize(nnz, size_of::<f32>(), "expr data bytes")? as u64;

        if e.indptr_len_bytes != expect_indptr {
            return Err(anyhow!(
                "expr indptr_len_bytes={} (expected {})",
                e.indptr_len_bytes,
                expect_indptr
            ));
        }
        if e.indices_len_bytes != expect_indices {
            return Err(anyhow!(
                "expr indices_len_bytes={} (expected {})",
                e.indices_len_bytes,
                expect_indices
            ));
        }
        if e.data_len_bytes != expect_data {
            return Err(anyhow!(
                "expr data_len_bytes={} (expected {})",
                e.data_len_bytes,
                expect_data
            ));
        }

        let end = self
            .validate_range(e.indptr_offset, e.indptr_len_bytes, "expr.indptr")?
            .max(self.validate_range(e.indices_offset, e.indices_len_bytes, "expr.indices")?)
            .max(self.validate_range(e.data_offset, e.data_len_bytes, "expr.data")?);

        let indptr_b = self.slice_bytes(e.indptr_offset, e.indptr_len_bytes)?;
        let mut prev = 0usize;
        for (i, chunk) in indptr_b.chunks_exact(4).enumerate() {
            let v = u32::from_le_bytes(chunk.try_into().unwrap()) as usize;
            if i > 0 && v < prev {
                return Err(anyhow!("expr indptr is not monotonic at position {}", i));
            }
            if v > nnz {
                return Err(anyhow!("expr indptr points past nnz at position {}", i));
            }
            prev = v;
        }
        if prev != nnz {
            return Err(anyhow!(
                "expr indptr last value {} does not equal nnz {}",
                prev,
                nnz
            ));
        }

        let indices_b = self.slice_bytes(e.indices_offset, e.indices_len_bytes)?;
        for (i, chunk) in indices_b.chunks_exact(4).enumerate() {
            let cell = u32::from_le_bytes(chunk.try_into().unwrap()) as usize;
            if cell >= n_points {
                return Err(anyhow!(
                    "expr indices[{}] out of range ({} >= n_points {})",
                    i,
                    cell,
                    n_points
                ));
            }
        }

        Ok(end)
    }

    fn slice_bytes(&self, offset: u64, len: u64) -> anyhow::Result<&[u8]> {
        let off = Self::u64_to_usize(offset, "offset")?;
        let len = Self::u64_to_usize(len, "length")?;
        let start = Self::checked_add_usize(self.data_start, off, "slice start")?;
        let end = Self::checked_add_usize(start, len, "slice end")?;
        if end > self.mmap.len() {
            return Err(anyhow!("out-of-bounds slice"));
        }
        Ok(&self.mmap[start..end])
    }

    fn slice_typed<T>(&self, offset: u64, len_bytes: u64, kind: &str) -> anyhow::Result<&[T]> {
        let elem = size_of::<T>();
        let len = Self::u64_to_usize(len_bytes, kind)?;
        if len % elem != 0 {
            return Err(anyhow!("{} length is not aligned to element size", kind));
        }
        let b = self.slice_bytes(offset, len_bytes)?;
        if (b.as_ptr() as usize) % align_of::<T>() != 0 {
            return Err(anyhow!("unaligned {} data", kind));
        }
        let n = len / elem;
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, n) })
    }

    pub fn space_f32_2d(&self, space_idx: usize) -> anyhow::Result<&[f32]> {
        let s = self
            .meta
            .spaces
            .get(space_idx)
            .ok_or_else(|| anyhow!("invalid space index"))?;
        if s.dims != 2 {
            return Err(anyhow!("space dims != 2 (got {})", s.dims));
        }
        self.slice_typed::<f32>(s.offset, s.len_bytes, "space")
    }

    pub fn obs_categorical(
        &self,
        obs_idx: usize,
    ) -> anyhow::Result<(&str, &[u32], &[String], Option<&[u32]>)> {
        let o = self
            .meta
            .obs
            .get(obs_idx)
            .ok_or_else(|| anyhow!("invalid obs index"))?;
        match o {
            ObsMeta::Categorical {
                name,
                offset,
                len_bytes,
                categories,
                palette_rgba8,
            } => {
                let labels = self.slice_typed::<u32>(*offset, *len_bytes, "categorical obs")?;
                let pal = palette_rgba8.as_ref().map(|v| v.as_slice());
                Ok((name.as_str(), labels, categories.as_slice(), pal))
            }
            _ => Err(anyhow!("obs is not categorical")),
        }
    }

    pub fn obs_continuous(&self, obs_idx: usize) -> anyhow::Result<(&str, &[f32])> {
        let o = self
            .meta
            .obs
            .get(obs_idx)
            .ok_or_else(|| anyhow!("invalid obs index"))?;
        match o {
            ObsMeta::Continuous {
                name,
                offset,
                len_bytes,
            } => {
                let vals = self.slice_typed::<f32>(*offset, *len_bytes, "continuous obs")?;
                Ok((name.as_str(), vals))
            }
            _ => Err(anyhow!("obs is not continuous")),
        }
    }

    /// The cell's name from the .h5ad, or its row number when the file has
    /// no names (converted before names were recorded).
    pub fn cell_name(&self, cell: u32) -> String {
        self.meta
            .obs_names
            .as_ref()
            .and_then(|names| names.get(cell as usize))
            .cloned()
            .unwrap_or_else(|| cell.to_string())
    }

    /// Expression of one gene in one cell (0.0 where the cell has no entry).
    pub fn gene_value(&self, gene_id: u32, cell: u32) -> anyhow::Result<f32> {
        let e = self
            .meta
            .expr
            .as_ref()
            .ok_or_else(|| anyhow!("no expr in file"))?;
        let indptr = self.slice_typed::<u32>(e.indptr_offset, e.indptr_len_bytes, "expr indptr")?;
        let indices =
            self.slice_typed::<u32>(e.indices_offset, e.indices_len_bytes, "expr indices")?;
        let data = self.slice_typed::<f32>(e.data_offset, e.data_len_bytes, "expr data")?;
        let g = gene_id as usize;
        if g + 1 >= indptr.len() {
            return Err(anyhow!("gene_id out of range"));
        }
        let (start, end) = (indptr[g] as usize, indptr[g + 1] as usize);
        if start > end || end > indices.len() || end > data.len() {
            return Err(anyhow!("invalid csc pointers"));
        }
        // Row indices are usually sorted within a column, but nothing in the
        // format promises it, so scan.
        Ok(indices[start..end]
            .iter()
            .position(|&c| c == cell)
            .map(|k| data[start + k])
            .unwrap_or(0.0))
    }

    /// Replace the scenes stored in this dataset's file. See `write_scenes`.
    pub fn write_scenes(&self, path: &Path, scenes: &[serde_json::Value]) -> anyhow::Result<()> {
        write_scenes(path, self.data_end, scenes)
    }

    pub fn find_gene(&self, gene: &str) -> Option<u32> {
        self.gene_index.as_ref()?.get(gene).copied()
    }

    /// If expression is present (CSC), fill a per-cell vector for gene_id.
    /// Output length = n_points, missing cells are 0.0.
    pub fn gene_vector_csc(&self, gene_id: u32) -> anyhow::Result<Vec<f32>> {
        let e = self
            .meta
            .expr
            .as_ref()
            .ok_or_else(|| anyhow!("no expr in file"))?;
        if e.kind != "csc" {
            return Err(anyhow!("expr kind is not csc"));
        }
        if gene_id >= e.n_genes {
            return Err(anyhow!("gene_id out of range"));
        }

        let indptr = self.slice_typed::<u32>(e.indptr_offset, e.indptr_len_bytes, "expr indptr")?;
        let indices =
            self.slice_typed::<u32>(e.indices_offset, e.indices_len_bytes, "expr indices")?;
        let data = self.slice_typed::<f32>(e.data_offset, e.data_len_bytes, "expr data")?;

        let g = gene_id as usize;
        if g + 1 >= indptr.len() {
            return Err(anyhow!("indptr too short"));
        }
        let start = indptr[g] as usize;
        let end = indptr[g + 1] as usize;
        if end > indices.len() || end > data.len() || start > end {
            return Err(anyhow!("invalid csc pointers"));
        }

        let mut out = Vec::new();
        out.try_reserve_exact(self.meta.n_points as usize)
            .map_err(|_| anyhow!("gene vector allocation failed"))?;
        out.resize(self.meta.n_points as usize, 0.0f32);

        for k in start..end {
            let cell = indices[k] as usize;
            if cell < out.len() {
                out[cell] = data[k];
            }
        }
        Ok(out)
    }
}

fn parse_scenes_footer(tail: &[u8], file_len: u64) -> Option<ScenesFooter> {
    if tail.len() != SCENES_FOOTER_BYTES || !tail.ends_with(&SCENES_MAGIC) {
        return None;
    }
    let block_len = u64::from_le_bytes(tail[0..8].try_into().unwrap());
    let version = u32::from_le_bytes(tail[8..12].try_into().unwrap());
    let block_start = file_len
        .checked_sub(SCENES_FOOTER_BYTES as u64)?
        .checked_sub(block_len)?;
    Some(ScenesFooter {
        block_start,
        block_len,
        version,
    })
}

/// Find the scenes footer at the end of `bytes` (a whole file), if any. A
/// footer whose block would overlap the data section is not ours and is ignored.
fn find_scenes_footer(bytes: &[u8], data_end: u64) -> Option<ScenesFooter> {
    let tail = bytes.get(bytes.len().checked_sub(SCENES_FOOTER_BYTES)?..)?;
    parse_scenes_footer(tail, bytes.len() as u64).filter(|f| f.block_start >= data_end)
}

fn read_scenes_block(bytes: &[u8], data_end: u64) -> SavedScenes {
    let Some(footer) = find_scenes_footer(bytes, data_end) else {
        return SavedScenes::default();
    };
    let warn = |msg: &str| SavedScenes {
        scenes: Vec::new(),
        warning: Some(msg.to_string()),
    };
    if footer.version != SCENES_VERSION {
        return warn(
            "This file's saved scenes were made by a newer version of stviz-animate. \
             Update the app to see them.",
        );
    }
    if footer.block_len > MAX_SCENES_BYTES {
        return warn("This file's saved scenes are too large to read.");
    }
    let start = footer.block_start as usize;
    let end = start + footer.block_len as usize;
    match serde_json::from_slice::<ScenesBlock>(&bytes[start..end]) {
        Ok(block) => SavedScenes {
            scenes: block.scenes,
            warning: None,
        },
        Err(_) => warn(
            "This file's saved scenes are damaged and could not be read. \
             Saving a scene will replace them.",
        ),
    }
}

/// Read the scenes from a .stviz file without opening its data, e.g. to carry
/// them over when the file is converted again. `None` if there are none.
pub fn read_scenes_from_path(path: &Path) -> Option<Vec<serde_json::Value>> {
    let mut file = File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    let mut tail = [0u8; SCENES_FOOTER_BYTES];
    file.seek(SeekFrom::Start(len.checked_sub(tail.len() as u64)?))
        .ok()?;
    file.read_exact(&mut tail).ok()?;
    let footer = parse_scenes_footer(&tail, len)?;
    if footer.version != SCENES_VERSION || footer.block_len > MAX_SCENES_BYTES {
        return None;
    }
    let mut block = vec![0u8; footer.block_len as usize];
    file.seek(SeekFrom::Start(footer.block_start)).ok()?;
    file.read_exact(&mut block).ok()?;
    let block: ScenesBlock = serde_json::from_slice(&block).ok()?;
    (!block.scenes.is_empty()).then_some(block.scenes)
}

/// Replace the scenes block of the .stviz at `path`. `data_end` is where the
/// file's data section ends; nothing before it is touched.
///
/// An existing block is overwritten in place; otherwise the block is appended.
/// The file never shrinks: a shorter block is padded with spaces (valid JSON
/// whitespace), because Windows refuses to truncate a file while it is
/// memory-mapped, and the app keeps the open dataset mapped.
pub fn write_scenes(
    path: &Path,
    data_end: u64,
    scenes: &[serde_json::Value],
) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open for writing: {}", path.display()))?;
    let len = file.metadata()?.len();
    if len < data_end {
        return Err(anyhow!(
            "the file changed on disk since it was opened; open it again"
        ));
    }
    let mut footer = None;
    if len >= data_end + SCENES_FOOTER_BYTES as u64 {
        let mut tail = [0u8; SCENES_FOOTER_BYTES];
        file.seek(SeekFrom::Start(len - tail.len() as u64))?;
        file.read_exact(&mut tail)?;
        footer = parse_scenes_footer(&tail, len).filter(|f| f.block_start >= data_end);
    }
    let (start, min_len) = match footer {
        Some(f) if f.version != SCENES_VERSION => {
            return Err(anyhow!(
                "this file's scenes were saved by a newer version of stviz-animate; \
                 update the app to change them"
            ))
        }
        Some(f) => (f.block_start, f.block_len),
        None => (len, 0),
    };

    let mut block = serde_json::to_vec(&ScenesBlock {
        version: SCENES_VERSION,
        scenes: scenes.to_vec(),
    })?;
    if block.len() as u64 > MAX_SCENES_BYTES {
        return Err(anyhow!("too much scene data to save"));
    }
    if (block.len() as u64) < min_len {
        block.resize(min_len as usize, b' ');
    }
    let mut out = block;
    let block_len = out.len() as u64;
    out.extend_from_slice(&block_len.to_le_bytes());
    out.extend_from_slice(&SCENES_VERSION.to_le_bytes());
    out.extend_from_slice(&SCENES_MAGIC);

    file.seek(SeekFrom::Start(start))?;
    file.write_all(&out)
        .with_context(|| format!("write scenes: {}", path.display()))?;
    file.sync_all()?;
    Ok(())
}

/// Builds small .stviz files for tests.
#[cfg(test)]
pub mod test_support {
    use super::{HEADER_BYTES_WITH_MAGIC, STVIZ_MAGIC, STVIZ_VERSION};
    use std::path::Path;

    /// Metadata and data section for `n` points: spaces "spatial" and "X_umap",
    /// a categorical "leiden" (3 categories), a continuous "score", and one
    /// gene "G0" expressed in every cell.
    pub fn stviz_parts(n: u32, source: Option<&str>) -> (serde_json::Value, Vec<u8>) {
        let n = n as usize;
        let mut data = Vec::new();
        let mut push = |bytes: Vec<u8>| {
            let off = data.len();
            data.extend_from_slice(&bytes);
            (off, bytes.len())
        };
        let f32s = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let u32s = |v: Vec<u32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();

        let spatial = push(f32s((0..n * 2).map(|i| i as f32).collect()));
        let umap = push(f32s((0..n * 2).map(|i| -(i as f32)).collect()));
        let leiden = push(u32s((0..n).map(|i| (i % 3) as u32).collect()));
        let score = push(f32s((0..n).map(|i| i as f32 * 0.5).collect()));
        let indptr = push(u32s(vec![0, n as u32]));
        let indices = push(u32s((0..n as u32).collect()));
        let expr = push(f32s((0..n).map(|i| i as f32 + 1.0).collect()));
        let hi = (n * 2).saturating_sub(1) as f32;
        let meta = serde_json::json!({
            "version": STVIZ_VERSION,
            "n_points": n,
            "source": source,
            "spaces": [
                {"name": "spatial", "dims": 2, "offset": spatial.0, "len_bytes": spatial.1,
                 "bbox": [0.0, 1.0, hi - 1.0, hi]},
                {"name": "X_umap", "dims": 2, "offset": umap.0, "len_bytes": umap.1,
                 "bbox": [1.0 - hi, -hi, 0.0, -1.0]},
            ],
            "obs": [
                {"kind": "categorical", "name": "leiden", "offset": leiden.0,
                 "len_bytes": leiden.1, "categories": ["0", "1", "2"], "palette_rgba8": null},
                {"kind": "continuous", "name": "score", "offset": score.0, "len_bytes": score.1},
            ],
            "expr": {
                "kind": "csc", "n_genes": 1, "nnz": n, "var_names": ["G0"],
                "indptr_offset": indptr.0, "indptr_len_bytes": indptr.1,
                "indices_offset": indices.0, "indices_len_bytes": indices.1,
                "data_offset": expr.0, "data_len_bytes": expr.1,
            },
        });
        (meta, data)
    }

    pub fn encode_stviz(meta: &serde_json::Value, data: &[u8]) -> Vec<u8> {
        let json = serde_json::to_vec(meta).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&STVIZ_MAGIC);
        out.extend_from_slice(&STVIZ_VERSION.to_le_bytes());
        out.extend_from_slice(&(json.len() as u64).to_le_bytes());
        out.extend_from_slice(&json);
        debug_assert_eq!(out.len(), HEADER_BYTES_WITH_MAGIC + json.len());
        while out.len() % 16 != 0 {
            out.push(0);
        }
        out.extend_from_slice(data);
        out
    }

    pub fn write_stviz(path: &Path, n: u32, source: Option<&str>) {
        let (meta, data) = stviz_parts(n, source);
        std::fs::write(path, encode_stviz(&meta, &data)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        read_scenes_from_path, test_support::*, Dataset, SavedScenes, SCENES_MAGIC, SCENES_VERSION,
    };
    use std::path::PathBuf;

    fn temp_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stviz_data_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn load_err(name: &str, bytes: &[u8]) -> String {
        let path = temp_file(name);
        std::fs::write(&path, bytes).unwrap();
        match Dataset::load(&path) {
            Ok(_) => panic!("{name} should not load"),
            Err(e) => format!("{e:#}"),
        }
    }

    #[test]
    fn loads_a_valid_file() {
        let path = temp_file("valid.stviz");
        write_stviz(&path, 6, Some("/data/a.h5ad"));
        let ds = Dataset::load(&path).unwrap();
        assert_eq!(ds.meta.n_points, 6);
        assert_eq!(ds.meta.source.as_deref(), Some("/data/a.h5ad"));
        assert_eq!(Dataset::peek_source(&path).as_deref(), Some("/data/a.h5ad"));
        assert_eq!(ds.space_f32_2d(0).unwrap().len(), 12);
        let (name, labels, categories, palette) = ds.obs_categorical(0).unwrap();
        assert_eq!((name, labels.len(), categories.len()), ("leiden", 6, 3));
        assert!(palette.is_none());
        assert_eq!(ds.obs_continuous(1).unwrap().1[2], 1.0);
        assert!(ds.obs_continuous(0).is_err());
        let gene = ds.find_gene("G0").unwrap();
        assert_eq!(
            ds.gene_vector_csc(gene).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert!(ds.find_gene("missing").is_none());
        assert_eq!(ds.gene_value(gene, 4).unwrap(), 5.0);
        // No names recorded: the row number stands in.
        assert_eq!(ds.cell_name(3), "3");
    }

    #[test]
    fn reads_cell_names_and_checks_their_count() {
        let (mut meta, data) = stviz_parts(3, None);
        meta["obs_names"] = serde_json::json!(["AAAC-1", "AAAG-1", "AACT-1"]);
        let path = temp_file("names.stviz");
        std::fs::write(&path, encode_stviz(&meta, &data)).unwrap();
        let ds = Dataset::load(&path).unwrap();
        assert_eq!(ds.cell_name(1), "AAAG-1");

        meta["obs_names"] = serde_json::json!(["only-one"]);
        assert!(load_err("names_short.stviz", &encode_stviz(&meta, &data)).contains("obs_names"));
    }

    #[test]
    fn peek_source_is_none_without_a_source() {
        let path = temp_file("nosource.stviz");
        write_stviz(&path, 3, None);
        assert_eq!(Dataset::peek_source(&path), None);
        assert_eq!(
            Dataset::peek_source(&temp_file("does-not-exist.stviz")),
            None
        );
    }

    #[test]
    fn rejects_damaged_files() {
        let (meta, data) = stviz_parts(6, None);
        let good = encode_stviz(&meta, &data);

        assert!(load_err("empty.stviz", b"").contains("empty"));
        assert!(load_err("tiny.stviz", b"abc").contains("too small"));
        assert!(!load_err("garbage.stviz", &[0xFFu8; 64]).is_empty());
        // Truncated data section.
        assert!(load_err("short.stviz", &good[..good.len() - 8]).contains("out of file bounds"));

        let mut wrong_version = good.clone();
        wrong_version[8] = 9;
        assert!(load_err("version.stviz", &wrong_version).contains("unsupported"));

        let mut bad = meta.clone();
        bad["n_points"] = 7.into();
        assert!(load_err("points.stviz", &encode_stviz(&bad, &data)).contains("len_bytes"));

        let mut bad = meta.clone();
        bad["spaces"] = serde_json::json!([]);
        assert!(load_err("nospaces.stviz", &encode_stviz(&bad, &data)).contains("no spaces"));

        let mut bad = meta.clone();
        bad["expr"]["nnz"] = 5.into();
        assert!(load_err("nnz.stviz", &encode_stviz(&bad, &data)).contains("expr"));

        // A gene's cell index pointing past the last cell.
        let off = meta["expr"]["indices_offset"].as_u64().unwrap() as usize;
        let mut bad_data = data.clone();
        bad_data[off..off + 4].copy_from_slice(&99u32.to_le_bytes());
        assert!(load_err("indices.stviz", &encode_stviz(&meta, &bad_data)).contains("out of range"));
    }

    fn scene(name: &str, payload: usize) -> serde_json::Value {
        serde_json::json!({"name": name, "view": {"blob": "x".repeat(payload)}})
    }

    fn names(ds: &Dataset) -> Vec<String> {
        ds.saved_scenes
            .scenes
            .iter()
            .map(|s| s["name"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn scenes_round_trip_without_touching_the_data() {
        let path = temp_file("scenes.stviz");
        write_stviz(&path, 6, None);
        let original = std::fs::read(&path).unwrap();
        let ds = Dataset::load(&path).unwrap();
        assert!(ds.saved_scenes.scenes.is_empty() && ds.saved_scenes.warning.is_none());
        assert_eq!(ds.data_end, original.len() as u64);

        // Written while the dataset is open (and mapped), as the app does.
        ds.write_scenes(&path, &[scene("a", 10), scene("b", 500)])
            .unwrap();
        let grown = std::fs::read(&path).unwrap();
        assert_eq!(
            &grown[..original.len()],
            &original[..],
            "data must not change"
        );
        assert_eq!(ds.space_f32_2d(0).unwrap()[3], 3.0);

        let ds2 = Dataset::load(&path).unwrap();
        assert_eq!(names(&ds2), ["a", "b"]);
        assert_eq!(ds2.saved_scenes.scenes[1], scene("b", 500));
        assert_eq!(ds2.space_f32_2d(1).unwrap(), ds.space_f32_2d(1).unwrap());
        assert_eq!(read_scenes_from_path(&path).unwrap().len(), 2);

        // Fewer scenes: rewritten in place, padded, never shrinking the file.
        ds2.write_scenes(&path, &[scene("c", 1)]).unwrap();
        let after = std::fs::read(&path).unwrap();
        assert_eq!(after.len(), grown.len());
        assert_eq!(&after[..original.len()], &original[..]);
        let ds3 = Dataset::load(&path).unwrap();
        assert_eq!(names(&ds3), ["c"]);

        // More scenes than fit: the block grows in place.
        ds3.write_scenes(&path, &[scene("c", 1), scene("d", 4000)])
            .unwrap();
        let ds4 = Dataset::load(&path).unwrap();
        assert_eq!(names(&ds4), ["c", "d"]);
        assert!(std::fs::metadata(&path).unwrap().len() > grown.len() as u64);

        // Deleting every scene leaves a valid, empty block.
        ds4.write_scenes(&path, &[]).unwrap();
        let ds5 = Dataset::load(&path).unwrap();
        assert!(ds5.saved_scenes.scenes.is_empty() && ds5.saved_scenes.warning.is_none());
        assert_eq!(read_scenes_from_path(&path), None);
    }

    #[test]
    fn unreadable_scene_blocks_never_stop_the_file_from_opening() {
        let (meta, data) = stviz_parts(6, None);
        let good = encode_stviz(&meta, &data);
        let footer = |len: u64, version: u32| {
            let mut f = len.to_le_bytes().to_vec();
            f.extend_from_slice(&version.to_le_bytes());
            f.extend_from_slice(&SCENES_MAGIC);
            f
        };

        // Damaged JSON: opens with a warning, and saving replaces the block.
        let mut bytes = good.clone();
        bytes.extend_from_slice(b"{not json");
        bytes.extend(footer(9, SCENES_VERSION));
        let path = temp_file("damaged-scenes.stviz");
        std::fs::write(&path, &bytes).unwrap();
        let ds = Dataset::load(&path).unwrap();
        assert!(ds
            .saved_scenes
            .warning
            .as_deref()
            .unwrap()
            .contains("damaged"));
        ds.write_scenes(&path, &[scene("ok", 1)]).unwrap();
        assert_eq!(names(&Dataset::load(&path).unwrap()), ["ok"]);

        // A newer scenes version: opens with a warning and is not overwritten.
        let mut bytes = good.clone();
        bytes.extend_from_slice(b"{}");
        bytes.extend(footer(2, SCENES_VERSION + 1));
        let path = temp_file("newer-scenes.stviz");
        std::fs::write(&path, &bytes).unwrap();
        let ds = Dataset::load(&path).unwrap();
        assert!(ds
            .saved_scenes
            .warning
            .as_deref()
            .unwrap()
            .contains("newer"));
        assert!(ds.write_scenes(&path, &[scene("x", 1)]).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);

        // A footer whose block would reach into the data section is ignored,
        // and saving appends a fresh block after it.
        let mut bytes = good.clone();
        bytes.extend(footer(64, SCENES_VERSION));
        let path = temp_file("overlap-scenes.stviz");
        std::fs::write(&path, &bytes).unwrap();
        let ds = Dataset::load(&path).unwrap();
        assert_eq!(ds.saved_scenes, SavedScenes::default());
        ds.write_scenes(&path, &[scene("y", 1)]).unwrap();
        let written = std::fs::read(&path).unwrap();
        assert_eq!(&written[..bytes.len()], &bytes[..]);
        assert_eq!(names(&Dataset::load(&path).unwrap()), ["y"]);
    }
}
