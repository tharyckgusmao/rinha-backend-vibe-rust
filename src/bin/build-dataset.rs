#[path = "../dataset.rs"]
mod dataset;
#[path = "../ivf.rs"]
mod ivf;

use std::path::PathBuf;

use dataset::{META_FILE_NAME, build_dataset_from_gzip_file, load_dataset_from_dir};
use ivf::{CENTROIDS_FILE, build_ivf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: build-dataset <references.json.gz> <output-dir>")?;
    let output = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: build-dataset <references.json.gz> <output-dir>")?;

    // Step 1: Build base dataset (vectors, labels, meta)
    if !output.join(META_FILE_NAME).exists() {
        let meta = build_dataset_from_gzip_file(&input, &output)?;
        eprintln!(
            "dataset: {} vectors (fraud={}, legit={}) -> {}",
            meta.total_vectors, meta.fraud_vectors, meta.legit_vectors, output.display()
        );
    } else {
        eprintln!("dataset already exists, skipping base build");
    }

    // Step 2: Build IVF index
    if !output.join(CENTROIDS_FILE).exists() {
        let dataset = load_dataset_from_dir(&output)?;
        build_ivf(&dataset, &output)?;
    } else {
        eprintln!("IVF index already exists, skipping");
    }

    Ok(())
}
