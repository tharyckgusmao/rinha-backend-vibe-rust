//! IVF index — offline build + instant load via mmap.
//!
//! Files produced by build:
//!   ivf-centroids.bin  — [NLIST * DIMS_PADDED] i16, flat
//!   ivf-vectors.bin    — [N * DIMS_PADDED] i16, grouped by cell
//!   ivf-labels.bin     — [N] u8, same order as vectors
//!   ivf-offsets.bin    — [NLIST + 1] u32, prefix-sum cell boundaries

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::dataset::{LoadedDataset, ReferenceLabel, VECTOR_DIMENSIONS};

pub const NLIST: usize = 8192;
pub const NPROBE: usize = 3;
const KMEANS_ITERS: usize = 5;
pub const DIMS_PADDED: usize = 16;

/// Number of coarse clusters (level 1)
const COARSE_CLUSTERS: usize = 64;
/// How many coarse clusters to probe at level 1
const COARSE_PROBE: usize = 4;
/// Centroids per coarse cluster (NLIST / COARSE_CLUSTERS)
const FINE_PER_COARSE: usize = NLIST / COARSE_CLUSTERS; // 128

pub const CENTROIDS_FILE: &str = "ivf-centroids.bin";
pub const VECTORS_FILE: &str = "ivf-vectors.bin";
pub const LABELS_FILE: &str = "ivf-labels.bin";
pub const OFFSETS_FILE: &str = "ivf-offsets.bin";

/// Runtime index — loaded from pre-built files, instant startup.
pub struct IvfIndex {
    centroids: Mmap,   // [NLIST * DIMS_PADDED] i16
    vectors: Mmap,     // [N * DIMS_PADDED] i16
    labels: Mmap,      // [N] u8
    offsets: Vec<u32>, // [NLIST + 1] cell boundaries
    /// Coarse centroids: mean of each group of FINE_PER_COARSE centroids
    coarse: Vec<[i32; DIMS_PADDED]>,
    total: usize,
}

impl IvfIndex {
    /// Load pre-built IVF index from directory. Instant (mmap).
    pub fn load(dir: &Path) -> io::Result<Self> {
        let centroids_file = File::open(dir.join(CENTROIDS_FILE))?;
        let vectors_file = File::open(dir.join(VECTORS_FILE))?;
        let labels_file = File::open(dir.join(LABELS_FILE))?;

        let centroids = unsafe { Mmap::map(&centroids_file)? };
        let vectors = unsafe { Mmap::map(&vectors_file)? };
        let labels = unsafe { Mmap::map(&labels_file)? };

        let offsets_raw = fs::read(dir.join(OFFSETS_FILE))?;
        let offsets: Vec<u32> = offsets_raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Build coarse centroids at load time (mean of each group of 128 fine centroids)
        let cent_slice: &[i16] = unsafe {
            std::slice::from_raw_parts(
                centroids.as_ptr() as *const i16,
                centroids.len() / 2,
            )
        };
        let mut coarse = vec![[0i32; DIMS_PADDED]; COARSE_CLUSTERS];
        for g in 0..COARSE_CLUSTERS {
            for f in 0..FINE_PER_COARSE {
                let idx = g * FINE_PER_COARSE + f;
                let off = idx * DIMS_PADDED;
                for d in 0..DIMS_PADDED {
                    coarse[g][d] += cent_slice[off + d] as i32;
                }
            }
            for d in 0..DIMS_PADDED {
                coarse[g][d] /= FINE_PER_COARSE as i32;
            }
        }

        let total = labels.len();
        eprintln!("[ivf] loaded: {} vectors, {} cells, {} coarse clusters", total, NLIST, COARSE_CLUSTERS);

        Ok(Self { centroids, vectors, labels, offsets, coarse, total })
    }

    pub fn is_ready(&self) -> bool {
        self.total >= 5
    }

    pub fn fraud_votes(&self, query: &[i16; VECTOR_DIMENSIONS]) -> usize {
        let mut q = [0i16; DIMS_PADDED];
        q[..VECTOR_DIMENSIONS].copy_from_slice(query);

        let probes = self.closest_cells(&q);

        let mut top_dists = [i64::MAX; 5];
        let mut top_idx = [0u32; 5];
        let mut top_len = 0usize;
        let vectors = self.vectors_slice();

        for cell_idx in probes {
            let start = self.offsets[cell_idx as usize] as usize;
            let end = self.offsets[cell_idx as usize + 1] as usize;
            for i in start..end {
                let offset = i * DIMS_PADDED;
                let v = unsafe { vectors.get_unchecked(offset..offset + DIMS_PADDED) };
                let dist = l2_16dims(&q, v);

                if top_len < 5 {
                    top_dists[top_len] = dist;
                    top_idx[top_len] = i as u32;
                    top_len += 1;
                    if top_len == 5 {
                        sort5(&mut top_dists, &mut top_idx);
                    }
                } else if dist < top_dists[4] {
                    top_dists[4] = dist;
                    top_idx[4] = i as u32;
                    sort5(&mut top_dists, &mut top_idx);
                }
            }
        }

        // Count fraud labels — direct access to mmap'd labels
        let labels = self.labels_slice();
        let mut fraud = 0u32;
        for i in 0..top_len {
            fraud += unsafe { *labels.get_unchecked(top_idx[i] as usize) } as u32;
        }
        fraud as usize
    }

    fn closest_cells(&self, query: &[i16; DIMS_PADDED]) -> [u16; NPROBE] {
        // Level 1: find top COARSE_PROBE coarse clusters (64 comparisons)
        let q32: [i32; DIMS_PADDED] = {
            let mut out = [0i32; DIMS_PADDED];
            for i in 0..DIMS_PADDED { out[i] = query[i] as i32; }
            out
        };

        let mut coarse_best = [(i64::MAX, 0u16); COARSE_PROBE];
        for g in 0..COARSE_CLUSTERS {
            let c = &self.coarse[g];
            let mut dist = 0i64;
            for d in 0..DIMS_PADDED {
                let diff = (q32[d] - c[d]) as i64;
                dist += diff * diff;
            }
            if dist < coarse_best[COARSE_PROBE - 1].0 {
                coarse_best[COARSE_PROBE - 1] = (dist, g as u16);
                let mut j = COARSE_PROBE - 1;
                while j > 0 && coarse_best[j].0 < coarse_best[j - 1].0 {
                    coarse_best.swap(j, j - 1);
                    j -= 1;
                }
            }
        }

        // Level 2: search fine centroids only within top coarse clusters
        // COARSE_PROBE * FINE_PER_COARSE = 4 * 128 = 512 comparisons (vs 8192 before)
        let centroids = self.centroids_slice();
        let mut best = [(i64::MAX, 0u16); NPROBE];

        for ci in 0..COARSE_PROBE {
            let group = coarse_best[ci].1 as usize;
            let start = group * FINE_PER_COARSE;
            let end = start + FINE_PER_COARSE;
            for idx in start..end {
                let offset = idx * DIMS_PADDED;
                let c = unsafe { centroids.get_unchecked(offset..offset + DIMS_PADDED) };
                let dist = l2_16dims(query, c);

                if dist < best[NPROBE - 1].0 {
                    best[NPROBE - 1] = (dist, idx as u16);
                    let mut j = NPROBE - 1;
                    while j > 0 && best[j].0 < best[j - 1].0 {
                        best.swap(j, j - 1);
                        j -= 1;
                    }
                }
            }
        }

        [best[0].1, best[1].1, best[2].1]
    }

    fn centroids_slice(&self) -> &[i16] {
        unsafe {
            std::slice::from_raw_parts(
                self.centroids.as_ptr() as *const i16,
                self.centroids.len() / 2,
            )
        }
    }

    fn vectors_slice(&self) -> &[i16] {
        unsafe {
            std::slice::from_raw_parts(
                self.vectors.as_ptr() as *const i16,
                self.vectors.len() / 2,
            )
        }
    }

    fn labels_slice(&self) -> &[u8] {
        &self.labels
    }
}

/// Sort 5 elements (insertion sort, optimal for tiny arrays)
#[inline(always)]
fn sort5(dists: &mut [i64; 5], indices: &mut [u32; 5]) {
    for i in 1..5 {
        let mut j = i;
        while j > 0 && dists[j] < dists[j - 1] {
            dists.swap(j, j - 1);
            indices.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// Offline IVF build — called by build-dataset binary.
pub fn build_ivf(dataset: &LoadedDataset, output_dir: &Path) -> io::Result<()> {
    let n = dataset.total_vectors();
    eprintln!("[ivf-build] {} vectors, {} cells, {} iters", n, NLIST, KMEANS_ITERS);

    // Init centroids
    let mut centroids = vec![[0i32; VECTOR_DIMENSIONS]; NLIST];
    let step = n / NLIST;
    for i in 0..NLIST {
        let v = dataset.vector_at(i * step);
        for d in 0..VECTOR_DIMENSIONS {
            centroids[i][d] = v[d] as i32;
        }
    }

    // K-means
    let mut assignments = vec![0u16; n];
    for iter in 0..KMEANS_ITERS {
        // Parallel assignment using chunks
        let chunk_size = n / 8;
        std::thread::scope(|s| {
            for (chunk_idx, chunk) in assignments.chunks_mut(chunk_size).enumerate() {
                let centroids = &centroids;
                let dataset = &dataset;
                let base = chunk_idx * chunk_size;
                s.spawn(move || {
                    for (local_i, assignment) in chunk.iter_mut().enumerate() {
                        let i = base + local_i;
                        let v = dataset.vector_at(i);
                        let mut best_dist = i64::MAX;
                        let mut best = 0u16;
                        for (ci, c) in centroids.iter().enumerate() {
                            let mut d = 0i64;
                            for dim in 0..VECTOR_DIMENSIONS {
                                let diff = v[dim] as i64 - c[dim] as i64;
                                d += diff * diff;
                            }
                            if d < best_dist {
                                best_dist = d;
                                best = ci as u16;
                            }
                        }
                        *assignment = best;
                    }
                });
            }
        });

        let mut sums = vec![[0i64; VECTOR_DIMENSIONS]; NLIST];
        let mut counts = vec![0u32; NLIST];
        for i in 0..n {
            let c = assignments[i] as usize;
            let v = dataset.vector_at(i);
            counts[c] += 1;
            for d in 0..VECTOR_DIMENSIONS {
                sums[c][d] += v[d] as i64;
            }
        }
        for c in 0..NLIST {
            if counts[c] > 0 {
                for d in 0..VECTOR_DIMENSIONS {
                    centroids[c][d] = (sums[c][d] / counts[c] as i64) as i32;
                }
            }
        }

        let mx = counts.iter().max().unwrap_or(&0);
        let mn = counts.iter().filter(|c| **c > 0).min().unwrap_or(&0);
        eprintln!("[ivf-build] iter {}: min={} max={} avg={}", iter, mn, mx, n / NLIST);
    }

    // Count per cell
    let mut counts = vec![0u32; NLIST];
    for &a in &assignments {
        counts[a as usize] += 1;
    }

    // Offsets (prefix sum)
    let mut offsets = vec![0u32; NLIST + 1];
    for i in 0..NLIST {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    // Write vectors and labels in cell order, padded to 16 dims
    let mut vectors_out = vec![0i16; n * DIMS_PADDED];
    let mut labels_out = vec![0u8; n];
    let mut pos = vec![0u32; NLIST];

    for i in 0..n {
        let cell = assignments[i] as usize;
        let dest = (offsets[cell] + pos[cell]) as usize;
        pos[cell] += 1;

        let src = dataset.vector_at(i);
        let dst_off = dest * DIMS_PADDED;
        for d in 0..VECTOR_DIMENSIONS {
            vectors_out[dst_off + d] = src[d];
        }
        labels_out[dest] = if dataset.label_at(i) == ReferenceLabel::Fraud { 1 } else { 0 };
    }

    // Write centroids padded
    let mut centroids_out = vec![0i16; NLIST * DIMS_PADDED];
    for c in 0..NLIST {
        for d in 0..VECTOR_DIMENSIONS {
            centroids_out[c * DIMS_PADDED + d] = centroids[c][d] as i16;
        }
    }

    // Write files
    fs::create_dir_all(output_dir)?;

    let mut f = File::create(output_dir.join(CENTROIDS_FILE))?;
    for v in &centroids_out { f.write_all(&v.to_le_bytes())?; }

    let mut f = File::create(output_dir.join(VECTORS_FILE))?;
    for v in &vectors_out { f.write_all(&v.to_le_bytes())?; }

    let mut f = File::create(output_dir.join(LABELS_FILE))?;
    f.write_all(&labels_out)?;

    let mut f = File::create(output_dir.join(OFFSETS_FILE))?;
    for o in &offsets { f.write_all(&o.to_le_bytes())?; }

    eprintln!("[ivf-build] done, wrote to {}", output_dir.display());
    Ok(())
}

#[inline(always)]
fn l2_16dims(a: &[i16], b: &[i16]) -> i64 {
    let d0 = a[0] as i32 - b[0] as i32;
    let d1 = a[1] as i32 - b[1] as i32;
    let d2 = a[2] as i32 - b[2] as i32;
    let d3 = a[3] as i32 - b[3] as i32;
    let d4 = a[4] as i32 - b[4] as i32;
    let d5 = a[5] as i32 - b[5] as i32;
    let d6 = a[6] as i32 - b[6] as i32;
    let d7 = a[7] as i32 - b[7] as i32;
    let acc0 = (d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7) as i64;

    let d8 = a[8] as i32 - b[8] as i32;
    let d9 = a[9] as i32 - b[9] as i32;
    let d10 = a[10] as i32 - b[10] as i32;
    let d11 = a[11] as i32 - b[11] as i32;
    let d12 = a[12] as i32 - b[12] as i32;
    let d13 = a[13] as i32 - b[13] as i32;
    let d14 = a[14] as i32 - b[14] as i32;
    let d15 = a[15] as i32 - b[15] as i32;
    let acc1 = (d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13 + d14*d14 + d15*d15) as i64;

    acc0 + acc1
}

struct _Unused; // keep file valid
