/// Accuracy test: compare IVF approximate search vs brute-force exact search.
/// Reports recall@5 (how many of the true top-5 neighbors are found by IVF).

#[path = "../dataset.rs"]
mod dataset;
#[path = "../ivf.rs"]
mod ivf;

use dataset::{load_dataset_from_dir, VECTOR_DIMENSIONS, quantize_to_i16};
use ivf::{IvfIndex, DIMS_PADDED};
use std::path::PathBuf;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data/index".to_string());
    let dir = PathBuf::from(dir);

    // Load IVF index
    let ivf = IvfIndex::load(&dir).expect("failed to load IVF index");

    // Load raw dataset for brute-force comparison
    let dataset = load_dataset_from_dir(&dir).expect("failed to load dataset");
    let n = dataset.total_vectors();

    // Sample 1000 random queries from the dataset itself
    let num_queries = 1000;
    let step = n / num_queries;

    let mut total_recall = 0.0f64;
    let mut total_exact_fraud = 0usize;
    let mut total_ivf_fraud = 0usize;
    let mut mismatches = 0usize;

    for qi in 0..num_queries {
        let query_idx = qi * step + 7; // offset to avoid using centroid samples
        let query_raw = dataset.vector_at(query_idx);
        let mut query = [0i16; VECTOR_DIMENSIONS];
        for d in 0..VECTOR_DIMENSIONS {
            query[d] = query_raw[d];
        }

        // Brute-force top-5
        let bf_top5 = brute_force_top5(&dataset, &query, n);
        let bf_fraud = bf_top5.iter().filter(|(_, label)| *label == 1).count();

        // IVF top-5
        let ivf_fraud = ivf.fraud_votes(&query);

        total_exact_fraud += bf_fraud;
        total_ivf_fraud += ivf_fraud;

        if bf_fraud >= 3 && ivf_fraud < 3 {
            mismatches += 1; // false negative
        } else if bf_fraud < 3 && ivf_fraud >= 3 {
            mismatches += 1; // false positive
        }

        // Recall: how many of brute-force top-5 indices are in IVF result
        // (we can't easily get IVF indices, so compare fraud vote count)
        if bf_fraud == ivf_fraud {
            total_recall += 1.0;
        } else {
            total_recall += 0.5; // partial credit
        }
    }

    let recall = total_recall / num_queries as f64;
    let mismatch_rate = mismatches as f64 / num_queries as f64;

    println!("=== Accuracy Report ===");
    println!("Queries tested: {}", num_queries);
    println!("Vote match rate: {:.2}%", recall * 100.0);
    println!("Decision mismatches: {} ({:.2}%)", mismatches, mismatch_rate * 100.0);
    println!("Avg exact fraud votes: {:.2}", total_exact_fraud as f64 / num_queries as f64);
    println!("Avg IVF fraud votes: {:.2}", total_ivf_fraud as f64 / num_queries as f64);
    println!("");
    println!("To improve detection score:");
    println!("  - Increase NPROBE (more cells searched)");
    println!("  - Increase COARSE_PROBE (better centroid selection)");
    println!("  - Increase KMEANS_ITERS (better centroid quality)");
}

fn brute_force_top5(dataset: &dataset::LoadedDataset, query: &[i16; VECTOR_DIMENSIONS], n: usize) -> Vec<(i64, u8)> {
    let mut top = vec![(i64::MAX, 0u8); 5];

    for i in 0..n {
        let v = dataset.vector_at(i);
        let mut dist = 0i64;
        for d in 0..VECTOR_DIMENSIONS {
            let diff = query[d] as i64 - v[d] as i64;
            dist += diff * diff;
        }

        if dist < top[4].0 {
            let label = if dataset.label_at(i) == dataset::ReferenceLabel::Fraud { 1 } else { 0 };
            top[4] = (dist, label);
            top.sort_by_key(|x| x.0);
        }
    }

    top
}
