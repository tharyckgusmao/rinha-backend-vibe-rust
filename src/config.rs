use std::{collections::HashMap, fs, path::PathBuf};

use serde::Deserialize;

pub struct Config {
    pub port: u16,
    pub normalization_path: PathBuf,
    pub mcc_risk_path: PathBuf,
    pub dataset_dir: PathBuf,
}

impl Config {
    pub fn from_env() -> Self {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9999);

        let normalization_path = std::env::var("NORMALIZATION_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("resources/normalization.json"));
        let mcc_risk_path = std::env::var("MCC_RISK_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("resources/mcc_risk.json"));
        let dataset_dir = std::env::var("DATASET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/index"));

        Self {
            port,
            normalization_path,
            mcc_risk_path,
            dataset_dir,
        }
    }

    pub fn load_vectorizer_config(&self) -> Result<VectorizerConfig, ConfigError> {
        let normalization = read_json_file(&self.normalization_path)?;
        let mcc_risk = read_json_file(&self.mcc_risk_path)?;

        Ok(VectorizerConfig {
            normalization,
            mcc_risk,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct NormalizationConfig {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

#[derive(Debug, Clone)]
pub struct VectorizerConfig {
    pub normalization: NormalizationConfig,
    pub mcc_risk: HashMap<String, f32>,
}

#[derive(Debug)]
pub enum ConfigError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(f, "failed to read {}: {}", path.display(), source)
            }
            Self::Parse { path, source } => {
                write!(f, "failed to parse {}: {}", path.display(), source)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

fn read_json_file<T>(path: &PathBuf) -> Result<T, ConfigError>
where
    T: for<'de> Deserialize<'de>,
{
    let raw = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.clone(),
        source,
    })?;

    serde_json::from_str(&raw).map_err(|source| ConfigError::Parse {
        path: path.clone(),
        source,
    })
}
