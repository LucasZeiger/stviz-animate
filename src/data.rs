use anyhow::{anyhow, Context};
use memmap2::Mmap;
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    mem::{align_of, size_of},
    path::Path,
    sync::Arc,
};

const STVIZ_MAGIC: [u8; 8] = *b"STVIZ\0\0\0";
const STVIZ_VERSION: u32 = 1;
const HEADER_BYTES_WITH_MAGIC: usize = 8 + 4 + 8;
const LEGACY_HEADER_BYTES: usize = 8;

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
    pub spaces: Vec<SpaceMeta>,
    pub obs: Vec<ObsMeta>,
    pub expr: Option<ExprMeta>,
}

pub struct Dataset {
    mmap: Mmap,
    data_start: usize,
    pub meta: StvizMeta,
    gene_index: Option<HashMap<String, u32>>,
}

impl Dataset {
    pub fn load(path: &Path) -> anyhow::Result<Arc<Self>> {
        let file = File::open(path).with_context(|| format!("open: {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file).context("mmap")? };

        let (json_start, json_len, header_version) = Self::parse_header(&mmap)?;
        let json_end = Self::checked_add_usize(json_start, json_len, "json end")?;
        if json_end > mmap.len() {
            return Err(anyhow!("invalid json length in header"));
        }

        let meta: StvizMeta =
            serde_json::from_slice(&mmap[json_start..json_end]).context("parse meta json")?;
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

        let ds = Self {
            mmap,
            data_start,
            meta,
            gene_index,
        };
        ds.validate_layout()?;

        Ok(Arc::new(ds))
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

    fn validate_range(&self, offset: u64, len: u64, ctx: &str) -> anyhow::Result<()> {
        let off = Self::u64_to_usize(offset, &format!("{ctx} offset"))?;
        let len = Self::u64_to_usize(len, &format!("{ctx} length"))?;
        let start = Self::checked_add_usize(self.data_start, off, &format!("{ctx} start"))?;
        let end = Self::checked_add_usize(start, len, &format!("{ctx} end"))?;
        if end > self.mmap.len() {
            return Err(anyhow!("{ctx} block out of file bounds"));
        }
        Ok(())
    }

    fn validate_layout(&self) -> anyhow::Result<()> {
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
            self.validate_range(s.offset, s.len_bytes, &format!("space[{i}]"))?;
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
                    self.validate_range(*offset, *len_bytes, &format!("obs[{i}] categorical"))?;
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
                    self.validate_range(*offset, *len_bytes, &format!("obs[{i}] continuous"))?;
                }
            }
        }

        if let Some(expr) = self.meta.expr.as_ref() {
            self.validate_expr(expr, n_points)?;
        }

        Ok(())
    }

    fn validate_expr(&self, e: &ExprMeta, n_points: usize) -> anyhow::Result<()> {
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

        self.validate_range(e.indptr_offset, e.indptr_len_bytes, "expr.indptr")?;
        self.validate_range(e.indices_offset, e.indices_len_bytes, "expr.indices")?;
        self.validate_range(e.data_offset, e.data_len_bytes, "expr.data")?;

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

        Ok(())
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
