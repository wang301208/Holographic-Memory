use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HolographicConfig {
    pub encoding: EncodingConfig,
    pub storage: StorageConfig,
    pub retrieval: RetrievalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    pub fft_window_size: usize,
    pub overlap_ratio: f64,
    pub redundancy_level: u8,
    pub phase_modulation: bool,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: PathBuf,
    pub max_segment_size: usize,
    pub auto_compact: bool,
    pub sync_on_write: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub top_k: usize,
    pub similarity_threshold: f64,
    pub max_association_hops: u32,
    pub enable_partial_recovery: bool,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        EncodingConfig {
            fft_window_size: 1024,
            overlap_ratio: 0.5,
            redundancy_level: 3,
            phase_modulation: true,
            normalize: true,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig {
            data_dir: PathBuf::from("./holographic_data"),
            max_segment_size: 64 * 1024 * 1024,
            auto_compact: true,
            sync_on_write: false,
        }
    }
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        RetrievalConfig {
            top_k: 10,
            similarity_threshold: 0.3,
            max_association_hops: 3,
            enable_partial_recovery: true,
        }
    }
}

impl HolographicConfig {
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: HolographicConfig = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn load_from_str(s: &str) -> Result<Self, ConfigError> {
        let config: HolographicConfig = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.encoding.fft_window_size == 0 || (self.encoding.fft_window_size & (self.encoding.fft_window_size - 1)) != 0 {
            return Err(ConfigError::InvalidValue(
                "fft_window_size 必须是 2 的幂".to_string(),
            ));
        }
        if self.encoding.overlap_ratio < 0.0 || self.encoding.overlap_ratio >= 1.0 {
            return Err(ConfigError::InvalidValue(
                "overlap_ratio 必须在 [0.0, 1.0) 范围内".to_string(),
            ));
        }
        if self.encoding.redundancy_level == 0 {
            return Err(ConfigError::InvalidValue(
                "redundancy_level 必须大于 0".to_string(),
            ));
        }
        if self.retrieval.similarity_threshold < 0.0 || self.retrieval.similarity_threshold > 1.0 {
            return Err(ConfigError::InvalidValue(
                "similarity_threshold 必须在 [0.0, 1.0] 范围内".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("解析错误: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("无效配置值: {0}")]
    InvalidValue(String),
}
