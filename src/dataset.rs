use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use half::f16;
use memmap2::Mmap;
use serde::{Deserialize, Deserializer, Serialize, de::Visitor};

pub const VECTOR_DIMENSIONS: usize = 14;
pub const FORMAT_VERSION: u32 = 1;
pub const QUANTIZATION_SCALE: i16 = 10_000;
pub const PARTITION_BITS: usize = 8;
pub const PARTITION_COUNT: usize = 1 << PARTITION_BITS;
pub const BLOCK_SIZE: usize = 64;
const PARTITION_THRESHOLD: i16 = QUANTIZATION_SCALE / 2;
const PARTITION_DIMS: [usize; PARTITION_BITS] = [0, 2, 7, 12, 8, 1, 3, 4];

pub const F32_FILE_NAME: &str = "vectors-f32.bin";
pub const FP16_FILE_NAME: &str = "vectors-fp16.bin";
pub const I16_FILE_NAME: &str = "vectors-i16.bin";
pub const BLOCK_BOUNDS_FILE_NAME: &str = "block-bounds-i16.bin";
pub const LABELS_FILE_NAME: &str = "labels.bin";
pub const META_FILE_NAME: &str = "meta.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceLabel {
    Legit,
    Fraud,
}

impl ReferenceLabel {
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Legit => 0,
            Self::Fraud => 1,
        }
    }
}

impl<'de> Deserialize<'de> for ReferenceLabel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "legit" => Ok(Self::Legit),
            "fraud" => Ok(Self::Fraud),
            _ => Err(serde::de::Error::custom("label must be 'legit' or 'fraud'")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceRecord {
    pub vector: [f32; VECTOR_DIMENSIONS],
    pub label: ReferenceLabel,
}

impl<'de> Deserialize<'de> for ReferenceRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawReferenceRecord {
            vector: Vec<f32>,
            label: ReferenceLabel,
        }

        let raw = RawReferenceRecord::deserialize(deserializer)?;
        let vector: [f32; VECTOR_DIMENSIONS] =
            raw.vector.try_into().map_err(|vector: Vec<f32>| {
                serde::de::Error::custom(format!(
                    "vector must have {} dimensions, got {}",
                    VECTOR_DIMENSIONS,
                    vector.len()
                ))
            })?;

        Ok(Self {
            vector,
            label: raw.label,
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DatasetMeta {
    pub version: u32,
    pub dimensions: usize,
    pub total_vectors: u64,
    pub fraud_vectors: u64,
    pub legit_vectors: u64,
    pub vectors_f32_path: String,
    pub vectors_fp16_path: String,
    pub vectors_i16_path: String,
    pub block_bounds_i16_path: String,
    pub labels_path: String,
    pub quantization_scale: i16,
    pub partition_offsets: Vec<u64>,
    pub partition_block_offsets: Vec<u64>,
    pub block_size: usize,
}

impl<'de> Deserialize<'de> for DatasetMeta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawMeta {
            version: u32,
            dimensions: usize,
            total_vectors: u64,
            fraud_vectors: u64,
            legit_vectors: u64,
            vectors_f32_path: String,
            vectors_fp16_path: String,
            vectors_i16_path: String,
            block_bounds_i16_path: String,
            labels_path: String,
            quantization_scale: i16,
            partition_offsets: Vec<u64>,
            partition_block_offsets: Vec<u64>,
            block_size: usize,
        }

        let raw = RawMeta::deserialize(deserializer)?;
        Ok(Self {
            version: raw.version,
            dimensions: raw.dimensions,
            total_vectors: raw.total_vectors,
            fraud_vectors: raw.fraud_vectors,
            legit_vectors: raw.legit_vectors,
            vectors_f32_path: raw.vectors_f32_path,
            vectors_fp16_path: raw.vectors_fp16_path,
            vectors_i16_path: raw.vectors_i16_path,
            block_bounds_i16_path: raw.block_bounds_i16_path,
            labels_path: raw.labels_path,
            quantization_scale: raw.quantization_scale,
            partition_offsets: raw.partition_offsets,
            partition_block_offsets: raw.partition_block_offsets,
            block_size: raw.block_size,
        })
    }
}

pub struct LoadedDataset {
    pub meta: DatasetMeta,
    vectors_i16: VectorStorage,
    #[allow(dead_code)]
    block_bounds_i16: BoundsStorage,
    labels: LabelStorage,
}

enum VectorStorage {
    Mmap(Mmap),
    #[cfg(test)]
    #[allow(dead_code)]
    Owned(Vec<i16>),
}

enum LabelStorage {
    Mmap(Mmap),
    #[cfg(test)]
    #[allow(dead_code)]
    Owned(Vec<u8>),
}

#[allow(dead_code)]
enum BoundsStorage {
    Mmap(Mmap),
    #[cfg(test)]
    #[allow(dead_code)]
    Owned(Vec<i16>),
}

impl LoadedDataset {
    pub fn total_vectors(&self) -> usize {
        self.meta.total_vectors as usize
    }

    pub fn vector_at(&self, index: usize) -> &[i16] {
        let start = index * VECTOR_DIMENSIONS;
        let end = start + VECTOR_DIMENSIONS;
        &self.vectors_i16_slice()[start..end]
    }

    pub fn label_at(&self, index: usize) -> ReferenceLabel {
        match self.labels_slice()[index] {
            0 => ReferenceLabel::Legit,
            1 => ReferenceLabel::Fraud,
            other => panic!("invalid label value in dataset: {}", other),
        }
    }

    pub fn partition_range(&self, partition: usize) -> std::ops::Range<usize> {
        let start = self.meta.partition_offsets[partition] as usize;
        let end = self.meta.partition_offsets[partition + 1] as usize;
        start..end
    }

    #[allow(dead_code)]
    pub fn partition_block_range(&self, partition: usize) -> std::ops::Range<usize> {
        let start = self.meta.partition_block_offsets[partition] as usize;
        let end = self.meta.partition_block_offsets[partition + 1] as usize;
        start..end
    }

    #[allow(dead_code)]
    pub fn block_vector_range(&self, partition: usize, block: usize) -> std::ops::Range<usize> {
        let partition_range = self.partition_range(partition);
        let local_block = block - self.meta.partition_block_offsets[partition] as usize;
        let start = partition_range.start + local_block * self.meta.block_size;
        let end = (start + self.meta.block_size).min(partition_range.end);
        start..end
    }

    #[allow(dead_code)]
    pub fn block_min(&self, block: usize) -> &[i16] {
        let start = block * VECTOR_DIMENSIONS * 2;
        let end = start + VECTOR_DIMENSIONS;
        &self.block_bounds_i16_slice()[start..end]
    }

    #[allow(dead_code)]
    pub fn block_max(&self, block: usize) -> &[i16] {
        let start = block * VECTOR_DIMENSIONS * 2 + VECTOR_DIMENSIONS;
        let end = start + VECTOR_DIMENSIONS;
        &self.block_bounds_i16_slice()[start..end]
    }

    fn vectors_i16_slice(&self) -> &[i16] {
        match &self.vectors_i16 {
            VectorStorage::Mmap(map) => unsafe {
                std::slice::from_raw_parts(
                    map.as_ptr() as *const i16,
                    map.len() / std::mem::size_of::<i16>(),
                )
            },
            #[cfg(test)]
            VectorStorage::Owned(values) => values,
        }
    }

    fn labels_slice(&self) -> &[u8] {
        match &self.labels {
            LabelStorage::Mmap(map) => unsafe {
                std::slice::from_raw_parts(map.as_ptr(), map.len())
            },
            #[cfg(test)]
            LabelStorage::Owned(values) => values,
        }
    }

    #[allow(dead_code)]
    fn block_bounds_i16_slice(&self) -> &[i16] {
        match &self.block_bounds_i16 {
            BoundsStorage::Mmap(map) => unsafe {
                std::slice::from_raw_parts(
                    map.as_ptr() as *const i16,
                    map.len() / std::mem::size_of::<i16>(),
                )
            },
            #[cfg(test)]
            BoundsStorage::Owned(values) => values,
        }
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub fn from_owned(
        meta: DatasetMeta,
        vectors_i16: Vec<i16>,
        block_bounds_i16: Vec<i16>,
        labels: Vec<ReferenceLabel>,
    ) -> Self {
        Self {
            meta,
            vectors_i16: VectorStorage::Owned(vectors_i16),
            block_bounds_i16: BoundsStorage::Owned(block_bounds_i16),
            labels: LabelStorage::Owned(labels.into_iter().map(|label| label.to_byte()).collect()),
        }
    }
}

pub fn build_dataset_from_gzip_file(
    input_path: &Path,
    output_dir: &Path,
) -> Result<DatasetMeta, DatasetBuildError> {
    fs::create_dir_all(output_dir).map_err(|source| DatasetBuildError::CreateDir {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let input = File::open(input_path).map_err(|source| DatasetBuildError::OpenInput {
        path: input_path.to_path_buf(),
        source,
    })?;
    let decoder = GzDecoder::new(BufReader::new(input));

    let f32_path = output_dir.join(F32_FILE_NAME);
    let fp16_path = output_dir.join(FP16_FILE_NAME);
    let i16_path = output_dir.join(I16_FILE_NAME);
    let block_bounds_path = output_dir.join(BLOCK_BOUNDS_FILE_NAME);
    let labels_path = output_dir.join(LABELS_FILE_NAME);
    let meta_path = output_dir.join(META_FILE_NAME);

    let f32_writer = File::create(&f32_path).map_err(|source| DatasetBuildError::CreateOutput {
        path: f32_path.clone(),
        source,
    })?;
    let fp16_writer =
        File::create(&fp16_path).map_err(|source| DatasetBuildError::CreateOutput {
            path: fp16_path.clone(),
            source,
        })?;
    let i16_writer = File::create(&i16_path).map_err(|source| DatasetBuildError::CreateOutput {
        path: i16_path.clone(),
        source,
    })?;
    let block_bounds_writer =
        File::create(&block_bounds_path).map_err(|source| DatasetBuildError::CreateOutput {
            path: block_bounds_path.clone(),
            source,
        })?;
    let labels_writer =
        File::create(&labels_path).map_err(|source| DatasetBuildError::CreateOutput {
            path: labels_path.clone(),
            source,
        })?;

    let mut outputs = BinaryDatasetWriter::new(
        f32_writer,
        fp16_writer,
        i16_writer,
        block_bounds_writer,
        labels_writer,
    );
    let meta = convert_reference_reader(decoder, &mut outputs)?;

    let meta_json = serde_json::to_vec_pretty(&meta)
        .map_err(|source| DatasetBuildError::SerializeMeta { source })?;
    fs::write(&meta_path, meta_json).map_err(|source| DatasetBuildError::WriteMeta {
        path: meta_path,
        source,
    })?;

    Ok(meta)
}

#[allow(dead_code)]
pub fn load_dataset_from_dir(dir: &Path) -> Result<LoadedDataset, DatasetLoadError> {
    let meta_path = dir.join(META_FILE_NAME);
    let meta_raw = fs::read_to_string(&meta_path).map_err(|source| DatasetLoadError::Read {
        path: meta_path.clone(),
        source,
    })?;
    let meta: DatasetMeta =
        serde_json::from_str(&meta_raw).map_err(|source| DatasetLoadError::ParseMeta {
            path: meta_path.clone(),
            source,
        })?;

    validate_meta(&meta).map_err(DatasetLoadError::InvalidMeta)?;

    let vectors_path = dir.join(&meta.vectors_i16_path);
    let block_bounds_path = dir.join(&meta.block_bounds_i16_path);
    let labels_path = dir.join(&meta.labels_path);

    let expected_vector_bytes = meta.total_vectors as usize * VECTOR_DIMENSIONS * 2;
    let vector_file = File::open(&vectors_path).map_err(|source| DatasetLoadError::Read {
        path: vectors_path.clone(),
        source,
    })?;
    let block_bounds_file =
        File::open(&block_bounds_path).map_err(|source| DatasetLoadError::Read {
            path: block_bounds_path.clone(),
            source,
        })?;
    let label_file = File::open(&labels_path).map_err(|source| DatasetLoadError::Read {
        path: labels_path.clone(),
        source,
    })?;

    let vector_len = vector_file
        .metadata()
        .map_err(|source| DatasetLoadError::Read {
            path: vectors_path.clone(),
            source,
        })?
        .len() as usize;
    let label_len = label_file
        .metadata()
        .map_err(|source| DatasetLoadError::Read {
            path: labels_path.clone(),
            source,
        })?
        .len() as usize;
    let block_bounds_len = block_bounds_file
        .metadata()
        .map_err(|source| DatasetLoadError::Read {
            path: block_bounds_path.clone(),
            source,
        })?
        .len() as usize;
    let expected_block_bounds_bytes = meta
        .partition_block_offsets
        .last()
        .copied()
        .unwrap_or_default() as usize
        * VECTOR_DIMENSIONS
        * 2
        * 2;

    if vector_len != expected_vector_bytes {
        return Err(DatasetLoadError::InvalidVectorBytes {
            expected: expected_vector_bytes,
            actual: vector_len,
        });
    }

    if label_len != meta.total_vectors as usize {
        return Err(DatasetLoadError::InvalidLabelBytes {
            expected: meta.total_vectors as usize,
            actual: label_len,
        });
    }
    if block_bounds_len != expected_block_bounds_bytes {
        return Err(DatasetLoadError::InvalidBlockBoundsBytes {
            expected: expected_block_bounds_bytes,
            actual: block_bounds_len,
        });
    }

    let vectors_i16 = unsafe { Mmap::map(&vector_file) }.map_err(DatasetLoadError::Io)?;
    let block_bounds_i16 =
        unsafe { Mmap::map(&block_bounds_file) }.map_err(DatasetLoadError::Io)?;
    let labels = unsafe { Mmap::map(&label_file) }.map_err(DatasetLoadError::Io)?;

    for byte in labels.iter() {
        if *byte != 0 && *byte != 1 {
            return Err(DatasetLoadError::InvalidLabelValue(*byte));
        }
    }

    Ok(LoadedDataset {
        meta,
        vectors_i16: VectorStorage::Mmap(vectors_i16),
        block_bounds_i16: BoundsStorage::Mmap(block_bounds_i16),
        labels: LabelStorage::Mmap(labels),
    })
}

pub fn convert_reference_reader<R, W>(
    reader: R,
    output: &mut BinaryDatasetWriter<W>,
) -> Result<DatasetMeta, DatasetBuildError>
where
    R: Read,
    W: Write,
{
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    deserializer.deserialize_seq(ReferenceSequenceVisitor { output })?;

    output.flush()?;

    Ok(DatasetMeta {
        version: FORMAT_VERSION,
        dimensions: VECTOR_DIMENSIONS,
        total_vectors: output.total_vectors,
        fraud_vectors: output.fraud_vectors,
        legit_vectors: output.total_vectors - output.fraud_vectors,
        vectors_f32_path: F32_FILE_NAME.into(),
        vectors_fp16_path: FP16_FILE_NAME.into(),
        vectors_i16_path: I16_FILE_NAME.into(),
        block_bounds_i16_path: BLOCK_BOUNDS_FILE_NAME.into(),
        labels_path: LABELS_FILE_NAME.into(),
        quantization_scale: QUANTIZATION_SCALE,
        partition_offsets: output.partition_offsets(),
        partition_block_offsets: output.partition_block_offsets(),
        block_size: BLOCK_SIZE,
    })
}

pub struct BinaryDatasetWriter<W>
where
    W: Write,
{
    f32_writer: BufWriter<W>,
    fp16_writer: BufWriter<W>,
    i16_writer: BufWriter<W>,
    block_bounds_writer: BufWriter<W>,
    labels_writer: BufWriter<W>,
    total_vectors: u64,
    fraud_vectors: u64,
    partition_vectors: [Vec<i16>; PARTITION_COUNT],
    partition_labels: [Vec<u8>; PARTITION_COUNT],
    partition_counts: [u64; PARTITION_COUNT],
}

impl<W> BinaryDatasetWriter<W>
where
    W: Write,
{
    pub fn new(
        f32_writer: W,
        fp16_writer: W,
        i16_writer: W,
        block_bounds_writer: W,
        labels_writer: W,
    ) -> Self {
        Self {
            f32_writer: BufWriter::new(f32_writer),
            fp16_writer: BufWriter::new(fp16_writer),
            i16_writer: BufWriter::new(i16_writer),
            block_bounds_writer: BufWriter::new(block_bounds_writer),
            labels_writer: BufWriter::new(labels_writer),
            total_vectors: 0,
            fraud_vectors: 0,
            partition_vectors: std::array::from_fn(|_| Vec::new()),
            partition_labels: std::array::from_fn(|_| Vec::new()),
            partition_counts: [0; PARTITION_COUNT],
        }
    }

    pub fn write_record(&mut self, record: &ReferenceRecord) -> Result<(), DatasetBuildError> {
        let partition = partition_key_from_f32(&record.vector);
        for value in record.vector {
            self.f32_writer
                .write_all(&value.to_le_bytes())
                .map_err(DatasetBuildError::Io)?;

            let quantized = f16::from_f32(value);
            self.fp16_writer
                .write_all(&quantized.to_bits().to_le_bytes())
                .map_err(DatasetBuildError::Io)?;

            let quantized_i16 = quantize_to_i16(value);
            self.partition_vectors[partition].push(quantized_i16);
        }

        self.partition_labels[partition].push(record.label.to_byte());
        self.partition_counts[partition] += 1;

        self.total_vectors += 1;
        if record.label == ReferenceLabel::Fraud {
            self.fraud_vectors += 1;
        }

        Ok(())
    }
    pub fn flush(&mut self) -> Result<(), DatasetBuildError> {
        self.f32_writer.flush().map_err(DatasetBuildError::Io)?;
        self.fp16_writer.flush().map_err(DatasetBuildError::Io)?;

        for partition in 0..PARTITION_COUNT {
            self.flush_partition(partition)?;
        }

        self.i16_writer.flush().map_err(DatasetBuildError::Io)?;
        self.block_bounds_writer
            .flush()
            .map_err(DatasetBuildError::Io)?;
        self.labels_writer.flush().map_err(DatasetBuildError::Io)?;
        Ok(())
    }

    fn flush_partition(&mut self, partition: usize) -> Result<(), DatasetBuildError> {
        let vectors = &self.partition_vectors[partition];
        let labels = &self.partition_labels[partition];
        let vector_count = labels.len();
        let mut sorted_indices: Vec<usize> = (0..vector_count).collect();

        // Keep each block spatially coherent so min/max bounds can prune real work.
        sorted_indices
            .sort_unstable_by_key(|idx| sort_key_for_vector(vector_at_flat(vectors, *idx)));

        for block_indices in sorted_indices.chunks(BLOCK_SIZE) {
            let bounds = block_bounds_for_indices(vectors, block_indices);
            for value in bounds.min {
                self.block_bounds_writer
                    .write_all(&value.to_le_bytes())
                    .map_err(DatasetBuildError::Io)?;
            }
            for value in bounds.max {
                self.block_bounds_writer
                    .write_all(&value.to_le_bytes())
                    .map_err(DatasetBuildError::Io)?;
            }
        }

        for idx in sorted_indices {
            for value in vector_at_flat(vectors, idx) {
                self.i16_writer
                    .write_all(&value.to_le_bytes())
                    .map_err(DatasetBuildError::Io)?;
            }
            self.labels_writer
                .write_all(&[labels[idx]])
                .map_err(DatasetBuildError::Io)?;
        }

        Ok(())
    }

    pub fn partition_offsets(&self) -> Vec<u64> {
        let mut offsets = Vec::with_capacity(PARTITION_COUNT + 1);
        let mut acc = 0_u64;
        offsets.push(acc);
        for count in self.partition_counts {
            acc += count;
            offsets.push(acc);
        }
        offsets
    }

    pub fn partition_block_offsets(&self) -> Vec<u64> {
        let mut offsets = Vec::with_capacity(PARTITION_COUNT + 1);
        let mut acc = 0_u64;
        offsets.push(acc);
        for count in self.partition_counts {
            acc += count.div_ceil(BLOCK_SIZE as u64);
            offsets.push(acc);
        }
        offsets
    }
}

struct BlockBounds {
    min: [i16; VECTOR_DIMENSIONS],
    max: [i16; VECTOR_DIMENSIONS],
}

fn block_bounds_for_indices(vectors: &[i16], indices: &[usize]) -> BlockBounds {
    let mut min = [i16::MAX; VECTOR_DIMENSIONS];
    let mut max = [i16::MIN; VECTOR_DIMENSIONS];

    for idx in indices {
        let vector = vector_at_flat(vectors, *idx);
        for dim in 0..VECTOR_DIMENSIONS {
            min[dim] = min[dim].min(vector[dim]);
            max[dim] = max[dim].max(vector[dim]);
        }
    }

    BlockBounds { min, max }
}

fn vector_at_flat(vectors: &[i16], idx: usize) -> &[i16] {
    let start = idx * VECTOR_DIMENSIONS;
    let end = start + VECTOR_DIMENSIONS;
    &vectors[start..end]
}

pub(crate) fn sort_key_for_vector(vector: &[i16]) -> u64 {
    let mut key = 0_u64;
    for dim in PARTITION_DIMS {
        let shifted = (vector[dim] as i32 - i16::MIN as i32) as u16;
        key = (key << 16) | shifted as u64;
    }
    key
}

pub fn quantize_to_i16(value: f32) -> i16 {
    let scaled = (value * QUANTIZATION_SCALE as f32).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

pub fn quantize_query(query: &[f32; VECTOR_DIMENSIONS]) -> [i16; VECTOR_DIMENSIONS] {
    let mut output = [0; VECTOR_DIMENSIONS];
    for i in 0..VECTOR_DIMENSIONS {
        output[i] = quantize_to_i16(query[i]);
    }
    output
}

pub fn partition_key_from_f32(vector: &[f32; VECTOR_DIMENSIONS]) -> usize {
    let mut key = 0_usize;
    for (bit, dim) in PARTITION_DIMS.iter().enumerate() {
        if quantize_to_i16(vector[*dim]) > PARTITION_THRESHOLD {
            key |= 1 << bit;
        }
    }
    key
}

#[allow(dead_code)]
pub fn partition_key_from_i16(vector: &[i16; VECTOR_DIMENSIONS]) -> usize {
    let mut key = 0_usize;
    for (bit, dim) in PARTITION_DIMS.iter().enumerate() {
        if vector[*dim] > PARTITION_THRESHOLD {
            key |= 1 << bit;
        }
    }
    key
}

struct ReferenceSequenceVisitor<'a, W>
where
    W: Write,
{
    output: &'a mut BinaryDatasetWriter<W>,
}

impl<'de, W> Visitor<'de> for ReferenceSequenceVisitor<'_, W>
where
    W: Write,
{
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON array of reference vectors")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        while let Some(record) = seq.next_element::<ReferenceRecord>()? {
            self.output
                .write_record(&record)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum DatasetBuildError {
    OpenInput { path: PathBuf, source: io::Error },
    CreateDir { path: PathBuf, source: io::Error },
    CreateOutput { path: PathBuf, source: io::Error },
    WriteMeta { path: PathBuf, source: io::Error },
    SerializeMeta { source: serde_json::Error },
    Json(serde_json::Error),
    Io(io::Error),
}

impl std::fmt::Display for DatasetBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenInput { path, source } => {
                write!(f, "failed to open input {}: {}", path.display(), source)
            }
            Self::CreateDir { path, source } => {
                write!(
                    f,
                    "failed to create output directory {}: {}",
                    path.display(),
                    source
                )
            }
            Self::CreateOutput { path, source } => {
                write!(f, "failed to create output {}: {}", path.display(), source)
            }
            Self::WriteMeta { path, source } => {
                write!(f, "failed to write meta {}: {}", path.display(), source)
            }
            Self::SerializeMeta { source } => write!(f, "failed to serialize meta: {}", source),
            Self::Json(source) => write!(f, "failed to parse dataset json: {}", source),
            Self::Io(source) => write!(f, "dataset io error: {}", source),
        }
    }
}

impl std::error::Error for DatasetBuildError {}

impl From<serde_json::Error> for DatasetBuildError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum DatasetLoadError {
    Read {
        path: PathBuf,
        source: io::Error,
    },
    ParseMeta {
        path: PathBuf,
        source: serde_json::Error,
    },
    InvalidMeta(&'static str),
    InvalidVectorBytes {
        expected: usize,
        actual: usize,
    },
    InvalidBlockBoundsBytes {
        expected: usize,
        actual: usize,
    },
    InvalidLabelBytes {
        expected: usize,
        actual: usize,
    },
    InvalidLabelValue(u8),
    Io(io::Error),
}

impl std::fmt::Display for DatasetLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(f, "failed to read {}: {}", path.display(), source)
            }
            Self::ParseMeta { path, source } => {
                write!(f, "failed to parse {}: {}", path.display(), source)
            }
            Self::InvalidMeta(reason) => write!(f, "invalid dataset meta: {}", reason),
            Self::InvalidVectorBytes { expected, actual } => write!(
                f,
                "invalid vector bytes length: expected {}, got {}",
                expected, actual
            ),
            Self::InvalidBlockBoundsBytes { expected, actual } => write!(
                f,
                "invalid block bounds bytes length: expected {}, got {}",
                expected, actual
            ),
            Self::InvalidLabelBytes { expected, actual } => write!(
                f,
                "invalid label bytes length: expected {}, got {}",
                expected, actual
            ),
            Self::InvalidLabelValue(value) => write!(f, "invalid label value: {}", value),
            Self::Io(source) => write!(f, "dataset mmap io error: {}", source),
        }
    }
}

impl std::error::Error for DatasetLoadError {}

#[allow(dead_code)]
fn validate_meta(meta: &DatasetMeta) -> Result<(), &'static str> {
    if meta.version != FORMAT_VERSION {
        return Err("unsupported dataset format version");
    }
    if meta.dimensions != VECTOR_DIMENSIONS {
        return Err("unsupported vector dimensions");
    }
    if meta.quantization_scale != QUANTIZATION_SCALE {
        return Err("unsupported quantization scale");
    }
    if meta.total_vectors != meta.fraud_vectors + meta.legit_vectors {
        return Err("fraud + legit counts do not match total");
    }
    if meta.partition_offsets.len() != PARTITION_COUNT + 1 {
        return Err("invalid partition offset table length");
    }
    if meta.partition_offsets.first().copied() != Some(0) {
        return Err("partition offsets must start at zero");
    }
    if meta.partition_offsets.last().copied() != Some(meta.total_vectors) {
        return Err("partition offsets must end at total_vectors");
    }
    if meta.partition_block_offsets.len() != PARTITION_COUNT + 1 {
        return Err("invalid partition block offset table length");
    }
    if meta.partition_block_offsets.first().copied() != Some(0) {
        return Err("partition block offsets must start at zero");
    }
    if meta.block_size != BLOCK_SIZE {
        return Err("unsupported block size");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read};

    use flate2::{Compression, write::GzEncoder};

    use super::{
        BinaryDatasetWriter, FORMAT_VERSION, PARTITION_COUNT, VECTOR_DIMENSIONS,
        build_dataset_from_gzip_file, convert_reference_reader, load_dataset_from_dir,
    };

    #[test]
    fn converts_json_array_to_binary_outputs() {
        let json = r#"
        [
          { "vector": [0.01, 0.0833, 0.05, 0.8261, 0.1667, -1, -1, 0.0432, 0.25, 0, 1, 0, 0.2, 0.0416], "label": "legit" },
          { "vector": [0.5796, 0.9167, 1.0, 0.0435, 0, 0.0056, 0.4394, 0.4598, 0.4, 1, 0, 1, 0.85, 0.0032], "label": "fraud" }
        ]
        "#;

        let mut f32_bytes = Vec::new();
        let mut fp16_bytes = Vec::new();
        let mut i16_bytes = Vec::new();
        let mut block_bounds_bytes = Vec::new();
        let mut label_bytes = Vec::new();
        let meta = {
            let mut writer = BinaryDatasetWriter::new(
                &mut f32_bytes,
                &mut fp16_bytes,
                &mut i16_bytes,
                &mut block_bounds_bytes,
                &mut label_bytes,
            );
            convert_reference_reader(Cursor::new(json.as_bytes()), &mut writer).unwrap()
        };

        assert_eq!(meta.version, FORMAT_VERSION);
        assert_eq!(meta.dimensions, VECTOR_DIMENSIONS);
        assert_eq!(meta.total_vectors, 2);
        assert_eq!(meta.fraud_vectors, 1);
        assert_eq!(meta.legit_vectors, 1);
        assert_eq!(meta.partition_offsets.len(), PARTITION_COUNT + 1);
        assert_eq!(meta.partition_block_offsets.len(), PARTITION_COUNT + 1);
        assert_eq!(f32_bytes.len(), 2 * VECTOR_DIMENSIONS * 4);
        assert_eq!(fp16_bytes.len(), 2 * VECTOR_DIMENSIONS * 2);
        assert_eq!(i16_bytes.len(), 2 * VECTOR_DIMENSIONS * 2);
        assert_eq!(block_bounds_bytes.len(), 2 * VECTOR_DIMENSIONS * 2 * 2);
        assert_eq!(label_bytes.len(), 2);
        assert_eq!(label_bytes.iter().filter(|label| **label == 0).count(), 1);
        assert_eq!(label_bytes.iter().filter(|label| **label == 1).count(), 1);
    }

    #[test]
    fn builds_dataset_artifacts_from_gzip_file() {
        let json = r#"[{ "vector": [0,0,0,0,0,-1,-1,0,0,0,1,0,0.15,0], "label": "legit" }]"#;
        let gz_bytes = gzip(json.as_bytes());
        let base_dir = std::env::temp_dir().join(format!(
            "rinha-dataset-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base_dir).unwrap();

        let input_path = base_dir.join("references.json.gz");
        let output_dir = base_dir.join("out");
        std::fs::write(&input_path, gz_bytes).unwrap();

        let meta = build_dataset_from_gzip_file(&input_path, &output_dir).unwrap();
        let loaded = load_dataset_from_dir(&output_dir).unwrap();

        assert_eq!(meta.total_vectors, 1);
        assert_eq!(loaded.meta.total_vectors, 1);
        assert_eq!(loaded.total_vectors(), 1);
        assert_eq!(loaded.vector_at(0).len(), VECTOR_DIMENSIONS);
        assert_eq!(loaded.label_at(0), super::ReferenceLabel::Legit);
        assert_eq!(loaded.partition_range(0).start, 0);
        assert!(output_dir.join("vectors-f32.bin").exists());
        assert!(output_dir.join("vectors-fp16.bin").exists());
        assert!(output_dir.join("vectors-i16.bin").exists());
        assert!(output_dir.join("block-bounds-i16.bin").exists());
        assert!(output_dir.join("labels.bin").exists());
        assert!(output_dir.join("meta.json").exists());

        let mut labels = Vec::new();
        std::fs::File::open(output_dir.join("labels.bin"))
            .unwrap()
            .read_to_end(&mut labels)
            .unwrap();
        assert_eq!(labels, vec![0]);

        std::fs::remove_dir_all(base_dir).unwrap();
    }

    fn gzip(input: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, input).unwrap();
        encoder.finish().unwrap()
    }
}
