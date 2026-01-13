use anyhow::{anyhow, Context};
use memmap2::Mmap;
use serde::Deserialize;
use std::{collections::HashMap, fs::File, path::Path, sync::Arc};

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
    Continuous { name: String, offset: u64, len_bytes: u64 },
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

        if mmap.len() < 8 {
            return Err(anyhow!("file too small"));
        }

        let json_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let json_start = 8usize;
        let json_end = json_start + json_len;
        if mmap.len() < json_end {
            return Err(anyhow!("invalid json_len"));
        }

        let meta: StvizMeta = serde_json::from_slice(&mmap[json_start..json_end]).context("parse meta json")?;

        // padding to 16-byte boundary
        let pad = (16 - (json_end % 16)) % 16;
        let data_start = json_end + pad;

        if mmap.len() < data_start {
            return Err(anyhow!("invalid padding/data_start"));
        }

        let gene_index = meta.expr.as_ref().map(|e| {
            let mut m = HashMap::with_capacity(e.var_names.len());
            for (i, g) in e.var_names.iter().enumerate() {
                m.insert(g.clone(), i as u32);
            }
            m
        });

        Ok(Arc::new(Self {
            mmap,
            data_start,
            meta,
            gene_index,
        }))
    }

    fn slice_bytes(&self, offset: u64, len: u64) -> anyhow::Result<&[u8]> {
        let start = self.data_start + offset as usize;
        let end = start + len as usize;
        if end > self.mmap.len() {
            return Err(anyhow!("out of bounds slice"));
        }
        Ok(&self.mmap[start..end])
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
        let b = self.slice_bytes(s.offset, s.len_bytes)?;
        if (b.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
            return Err(anyhow!("unaligned f32 data"));
        }
        let n = (s.len_bytes as usize) / std::mem::size_of::<f32>();
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n) })
    }

    pub fn obs_categorical(&self, obs_idx: usize) -> anyhow::Result<(&str, &[u32], &[String], Option<&[u32]>)> {
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
                let b = self.slice_bytes(*offset, *len_bytes)?;
                if (b.as_ptr() as usize) % std::mem::align_of::<u32>() != 0 {
                    return Err(anyhow!("unaligned u32 data"));
                }
                let n = (*len_bytes as usize) / std::mem::size_of::<u32>();
                let labels = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u32, n) };
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
            ObsMeta::Continuous { name, offset, len_bytes } => {
                let b = self.slice_bytes(*offset, *len_bytes)?;
                if (b.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
                    return Err(anyhow!("unaligned f32 data"));
                }
                let n = (*len_bytes as usize) / std::mem::size_of::<f32>();
                let vals = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n) };
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
        let e = self.meta.expr.as_ref().ok_or_else(|| anyhow!("no expr in file"))?;
        if e.kind != "csc" {
            return Err(anyhow!("expr kind is not csc"));
        }
        if gene_id >= e.n_genes {
            return Err(anyhow!("gene_id out of range"));
        }

        let indptr_b = self.slice_bytes(e.indptr_offset, e.indptr_len_bytes)?;
        let indices_b = self.slice_bytes(e.indices_offset, e.indices_len_bytes)?;
        let data_b = self.slice_bytes(e.data_offset, e.data_len_bytes)?;

        if (indptr_b.as_ptr() as usize) % std::mem::align_of::<u32>() != 0 {
            return Err(anyhow!("unaligned indptr"));
        }
        if (indices_b.as_ptr() as usize) % std::mem::align_of::<u32>() != 0 {
            return Err(anyhow!("unaligned indices"));
        }
        if (data_b.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
            return Err(anyhow!("unaligned data"));
        }

        let indptr_n = (e.indptr_len_bytes as usize) / 4;
        let indices_n = (e.indices_len_bytes as usize) / 4;
        let data_n = (e.data_len_bytes as usize) / 4;

        let indptr = unsafe { std::slice::from_raw_parts(indptr_b.as_ptr() as *const u32, indptr_n) };
        let indices = unsafe { std::slice::from_raw_parts(indices_b.as_ptr() as *const u32, indices_n) };
        let data = unsafe { std::slice::from_raw_parts(data_b.as_ptr() as *const f32, data_n) };

        let g = gene_id as usize;
        if g + 1 >= indptr.len() {
            return Err(anyhow!("indptr too short"));
        }
        let start = indptr[g] as usize;
        let end = indptr[g + 1] as usize;
        if end > indices.len() || end > data.len() || start > end {
            return Err(anyhow!("invalid csc pointers"));
        }

        let mut out = vec![0.0f32; self.meta.n_points as usize];
        for k in start..end {
            let cell = indices[k] as usize;
            if cell < out.len() {
                out[cell] = data[k];
            }
        }
        Ok(out)
    }
}
