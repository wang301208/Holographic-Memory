use std::path::Path;

use crate::codec::fourier_encoder::FourierEncoder;
use crate::codec::hologram_fragmenter::HologramFragmenter;
use crate::codec::redundancy_weaver::RedundancyWeaver;
use crate::foundation::config::HolographicConfig;
use crate::retrieval::partial_recovery::PartialRecoveryEngine;
use crate::retrieval::similarity_matcher::SimilarityMatcher;
use crate::storage::holographic_index::HolographicIndex;
use crate::storage::persistence::PersistenceEngine;
use crate::types::{AssociatedItem, FragmentId, HologramFragment, IntegrityReport, RetrievalResult};

#[derive(Debug, thiserror::Error)]
pub enum HoloError {
    #[error("编码错误: {0}")]
    Encode(String),
    #[error("解码错误: {0}")]
    Decode(String),
    #[error("存储错误: {0}")]
    Storage(String),
    #[error("检索错误: {0}")]
    Retrieval(String),
    #[error("持久化错误: {0}")]
    Persistence(#[from] crate::storage::persistence::PersistenceError),
}

pub struct HolographicMemory {
    config: HolographicConfig,
    encoder: FourierEncoder,
    fragmenter: HologramFragmenter,
    weaver: RedundancyWeaver,
    index: HolographicIndex,
    matcher: SimilarityMatcher,
    recovery: PartialRecoveryEngine,
    persistence: Option<PersistenceEngine>,
    next_source_id: u64,
}

impl HolographicMemory {
    pub fn new(config: HolographicConfig) -> Self {
        let fragment_size = config.encoding.fft_window_size / 4;
        Self {
            encoder: FourierEncoder::new(config.encoding.clone()),
            fragmenter: HologramFragmenter::new(fragment_size),
            weaver: RedundancyWeaver::new(config.encoding.redundancy_level),
            matcher: SimilarityMatcher::new(config.retrieval.similarity_threshold),
            recovery: PartialRecoveryEngine::new(config.encoding.redundancy_level),
            persistence: None,
            index: HolographicIndex::new(),
            config,
            next_source_id: 1,
        }
    }

    pub fn with_persistence(mut self, data_dir: impl AsRef<Path>) -> Self {
        self.persistence = Some(PersistenceEngine::new(data_dir));
        self
    }

    pub fn store(&mut self, data: &[f64]) -> Result<StoreResult, HoloError> {
        let encode_result = self.encoder.encode(data);
        if encode_result.fragments.is_empty() {
            return Err(HoloError::Encode("编码结果为空".to_string()));
        }

        let source_hash = encode_result.source_hash;
        let fragment_count = encode_result.fragments.len();

        let woven = self.weaver.weave(&encode_result.fragments);
        let total_after_weave = woven.len();

        let mut fragment_ids = Vec::with_capacity(woven.len());
        for fragment in woven {
            fragment_ids.push(fragment.id);
            self.index.insert(fragment);
        }

        if let Some(ref mut persist) = self.persistence {
            persist.save_index(&self.index, "main.idx")?;
        }

        let source_id = self.next_source_id;
        self.next_source_id += 1;

        Ok(StoreResult {
            source_id,
            source_hash,
            fragment_count,
            total_fragments: total_after_weave,
            fragment_ids,
        })
    }

    pub fn retrieve(&mut self, source_hash: u64, expected_len: usize) -> Result<Vec<f64>, HoloError> {
        let fragments: Vec<HologramFragment> = self.index.get_by_source(source_hash)
            .into_iter()
            .cloned()
            .collect();

        if fragments.is_empty() {
            return Err(HoloError::Decode(format!("未找到源 {}", source_hash)));
        }

        let unwoven = self.weaver.unweave(&fragments);
        let decoded = self.encoder.decode(&unwoven, expected_len);
        Ok(decoded)
    }

    pub fn search(&mut self, query: &[f64], top_k: usize) -> Result<Vec<AssociatedItem>, HoloError> {
        let encode_result = self.encoder.encode(query);
        if encode_result.fragments.is_empty() {
            return Err(HoloError::Retrieval("查询编码结果为空".to_string()));
        }

        let candidates: Vec<HologramFragment> = self.index.all_fragments()
            .into_iter()
            .cloned()
            .collect();

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let results = self.matcher.find_similar(&encode_result.fragments[0], &candidates, top_k);
        Ok(results)
    }

    pub fn integrity(&self, source_hash: u64) -> IntegrityReport {
        self.index.integrity_check(source_hash)
    }

    pub fn can_recover(&self, available: usize, total: usize) -> bool {
        self.recovery.can_recover(available, total as u32)
    }

    pub fn recover(&self, available: &[HologramFragment], total: u32) -> RetrievalResult {
        self.recovery.recover(available, total)
    }

    pub fn fragment_count(&self) -> usize {
        self.index.len()
    }

    pub fn source_count(&self) -> usize {
        self.index.all_source_hashes().len()
    }

    pub fn config(&self) -> &HolographicConfig {
        &self.config
    }

    pub fn save(&mut self) -> Result<(), HoloError> {
        if let Some(ref mut persist) = self.persistence {
            persist.save_index(&self.index, "main.idx")?;
        } else {
            return Err(HoloError::Persistence(crate::storage::persistence::PersistenceError::Io(
                std::io::Error::new(std::io::ErrorKind::NotFound, "未配置持久化引擎"),
            )));
        }
        Ok(())
    }

    pub fn load(&mut self) -> Result<(), HoloError> {
        if let Some(ref persist) = self.persistence {
            let loaded = persist.load_index("main.idx")?;
            self.index = loaded;
        } else {
            return Err(HoloError::Persistence(crate::storage::persistence::PersistenceError::Io(
                std::io::Error::new(std::io::ErrorKind::NotFound, "未配置持久化引擎"),
            )));
        }
        Ok(())
    }

    pub fn store_with_fault_tolerance(
        &mut self,
        data: &[f64],
        simulate_damage_pct: f64,
    ) -> Result<FaultToleranceResult, HoloError> {
        let store_result = self.store(data)?;

        let fragments: Vec<HologramFragment> = self.index.get_by_source(store_result.source_hash)
            .into_iter()
            .cloned()
            .collect();
        let total = fragments.len();

        let remove_count = (total as f64 * simulate_damage_pct) as usize;
        let available: Vec<HologramFragment> = fragments.iter()
            .skip(remove_count)
            .cloned()
            .collect();

        let decoded = self.encoder.decode(&available, data.len());

        let inner_start = total.min(64);
        let inner_end = data.len().saturating_sub(64);
        let mse = if inner_end > inner_start {
            data[inner_start..inner_end]
                .iter()
                .zip(decoded[inner_start..inner_end].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / (inner_end - inner_start) as f64
        } else {
            0.0
        };

        let integrity = IntegrityReport::new(total as u32, available.len() as u32);

        Ok(FaultToleranceResult {
            store: store_result,
            total_fragments: total,
            available_fragments: available.len(),
            damage_ratio: simulate_damage_pct,
            mse,
            integrity,
        })
    }
}

pub struct StoreResult {
    pub source_id: u64,
    pub source_hash: u64,
    pub fragment_count: usize,
    pub total_fragments: usize,
    pub fragment_ids: Vec<FragmentId>,
}

pub struct FaultToleranceResult {
    pub store: StoreResult,
    pub total_fragments: usize,
    pub available_fragments: usize,
    pub damage_ratio: f64,
    pub mse: f64,
    pub integrity: IntegrityReport,
}
