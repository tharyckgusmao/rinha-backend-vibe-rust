use crate::{
    dataset::{
        LoadedDataset, ReferenceLabel, VECTOR_DIMENSIONS, partition_key_from_i16, quantize_to_i16,
    },
    vector::QueryVector,
};

pub struct SearchIndex {
    dataset: LoadedDataset,
}

impl SearchIndex {
    pub fn new(dataset: LoadedDataset) -> Self {
        Self { dataset }
    }

    pub fn is_ready(&self) -> bool {
        self.dataset.meta.total_vectors >= 5
    }

    pub fn search_top_k(&self, query: &QueryVector, k: usize) -> Vec<SearchHit> {
        let quantized = quantize_query(query);
        baseline_search_top_k(&self.dataset, &quantized, k)
    }

    pub fn fraud_score(&self, query: &QueryVector) -> f64 {
        self.fraud_votes(query) as f64 / 5.0
    }

    pub fn fraud_votes(&self, query: &QueryVector) -> usize {
        let hits = self.search_top_k(query, 5);
        hits.iter()
            .filter(|hit| hit.label == ReferenceLabel::Fraud)
            .count()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchHit {
    pub distance: i64,
    pub label: ReferenceLabel,
}

fn baseline_search_top_k(
    dataset: &LoadedDataset,
    query: &[i16; VECTOR_DIMENSIONS],
    k: usize,
) -> Vec<SearchHit> {
    let mut hits = Vec::with_capacity(k);

    let primary_partition = partition_key_from_i16(query);
    scan_partition(dataset, query, primary_partition, k, &mut hits);

    hits
}

fn quantize_query(query: &QueryVector) -> [i16; VECTOR_DIMENSIONS] {
    let mut output = [0; VECTOR_DIMENSIONS];
    for idx in 0..VECTOR_DIMENSIONS {
        output[idx] = quantize_to_i16(query[idx]);
    }
    output
}

fn squared_l2_i16(query: &[i16; VECTOR_DIMENSIONS], candidate: &[i16]) -> i64 {
    let mut acc = 0_i64;
    for i in 0..VECTOR_DIMENSIONS {
        let diff = query[i] as i64 - candidate[i] as i64;
        acc += diff * diff;
    }
    acc
}

fn scan_partition(
    dataset: &LoadedDataset,
    query: &[i16; VECTOR_DIMENSIONS],
    partition: usize,
    k: usize,
    hits: &mut Vec<SearchHit>,
) {
    for block in dataset.partition_block_range(partition) {
        if hits.len() == k {
            let lower_bound =
                lower_bound_l2_i16(query, dataset.block_min(block), dataset.block_max(block));
            if lower_bound >= hits[k - 1].distance {
                continue;
            }
        }

        for idx in dataset.block_vector_range(partition, block) {
            let distance = squared_l2_i16(query, dataset.vector_at(idx));
            let label = dataset.label_at(idx);

            if hits.len() < k {
                hits.push(SearchHit { distance, label });
                if hits.len() == k {
                    hits.sort_by(|a, b| a.distance.cmp(&b.distance));
                }
                continue;
            }

            if distance < hits[k - 1].distance {
                hits[k - 1] = SearchHit { distance, label };
                hits.sort_by(|a, b| a.distance.cmp(&b.distance));
            }
        }
    }
}

fn lower_bound_l2_i16(query: &[i16; VECTOR_DIMENSIONS], min: &[i16], max: &[i16]) -> i64 {
    let mut acc = 0_i64;
    for dim in 0..VECTOR_DIMENSIONS {
        let q = query[dim] as i64;
        let lo = min[dim] as i64;
        let hi = max[dim] as i64;
        let diff = if q < lo {
            lo - q
        } else if q > hi {
            q - hi
        } else {
            0
        };
        acc += diff * diff;
    }
    acc
}

#[cfg(test)]
mod tests {
    use crate::dataset::{
        BLOCK_SIZE, DatasetMeta, LoadedDataset, PARTITION_COUNT, QUANTIZATION_SCALE, ReferenceLabel,
    };

    use super::SearchIndex;

    #[test]
    fn computes_top_k_and_fraud_score() {
        let dataset = LoadedDataset::from_owned(
            DatasetMeta {
                version: 1,
                dimensions: 14,
                total_vectors: 5,
                fraud_vectors: 2,
                legit_vectors: 3,
                vectors_f32_path: "vectors-f32.bin".into(),
                vectors_fp16_path: "vectors-fp16.bin".into(),
                vectors_i16_path: "vectors-i16.bin".into(),
                block_bounds_i16_path: "block-bounds-i16.bin".into(),
                labels_path: "labels.bin".into(),
                quantization_scale: QUANTIZATION_SCALE,
                partition_offsets: offsets_with_first_bucket(5),
                partition_block_offsets: offsets_with_first_bucket(1),
                block_size: BLOCK_SIZE,
            },
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
            .into_iter()
            .map(|value| (value * QUANTIZATION_SCALE as f32).round() as i16)
            .collect(),
            block_bounds_fixture(),
            vec![
                ReferenceLabel::Legit,
                ReferenceLabel::Fraud,
                ReferenceLabel::Fraud,
                ReferenceLabel::Legit,
                ReferenceLabel::Legit,
            ],
        );
        let index = SearchIndex::new(dataset);
        let query = [0.0; 14];

        let hits = index.search_top_k(&query, 5);

        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].label, ReferenceLabel::Legit);
        assert_eq!(index.fraud_score(&query), 0.4);
    }

    fn block_bounds_fixture() -> Vec<i16> {
        let min = [0; 14];
        let mut max = [0; 14];
        max[0] = (0.4 * QUANTIZATION_SCALE as f32).round() as i16;
        min.into_iter().chain(max).collect()
    }

    fn offsets_with_first_bucket(value: u64) -> Vec<u64> {
        let mut offsets = vec![0; PARTITION_COUNT + 1];
        for offset in offsets.iter_mut().skip(1) {
            *offset = value;
        }
        offsets
    }
}
