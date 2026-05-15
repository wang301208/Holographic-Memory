use std::path::Path;

use crate::codec::fourier_encoder::FourierEncoder;
use crate::codec::hologram_fragmenter::HologramFragmenter;
use crate::codec::redundancy_weaver::RedundancyWeaver;
use crate::codec::reed_solomon::ReedSolomon;
use crate::foundation::config::HolographicConfig;
use crate::retrieval::partial_recovery::PartialRecoveryEngine;
use crate::retrieval::similarity_matcher::SimilarityMatcher;
use crate::storage::holographic_index::HolographicIndex;
use crate::storage::persistence::PersistenceEngine;
use crate::storage::tiered_index::{TieredIndex, TieredConfig};
use crate::storage::mmap_persistence::MmapPersistence;
use crate::types::{AssociatedItem, FragmentId, HologramFragment, IntegrityReport, RetrievalResult};

/// 全息记忆存储的统一错误类型
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
    #[error("RS纠删码错误: {0}")]
    ReedSolomon(String),
    #[error("分层索引错误: {0}")]
    Tiered(#[from] crate::storage::tiered_index::TieredError),
    #[error("Mmap错误: {0}")]
    Mmap(#[from] crate::storage::mmap_persistence::MmapError),
}

enum IndexBackend {
    Simple(HolographicIndex),
    Tiered(Box<TieredIndex>),
}

/// 全息记忆存储引擎 - 统一高级 API
///
/// 将傅里叶编解码、冗余交织、索引、检索、持久化等组件整合为单一接口。
///
/// # 示例
///
/// ```rust
/// use holographic_memory::*;
///
/// let config = HolographicConfig::default();
/// let mut hm = HolographicMemory::new(config);
///
/// let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.02).sin()).collect();
/// let result = hm.store(&data).unwrap();
/// let decoded = hm.retrieve(result.source_hash, data.len()).unwrap();
/// ```
pub struct HolographicMemory {
    config: HolographicConfig,
    encoder: FourierEncoder,
    #[allow(dead_code)]
    fragmenter: HologramFragmenter,
    weaver: RedundancyWeaver,
    index: IndexBackend,
    matcher: SimilarityMatcher,
    recovery: PartialRecoveryEngine,
    persistence: Option<PersistenceEngine>,
    rs_codec: Option<ReedSolomon>,
    mmap: Option<MmapPersistence>,
    next_source_id: u64,
}

impl HolographicMemory {
    /// 创建新的全息记忆存储引擎
    pub fn new(config: HolographicConfig) -> Self {
        let fragment_size = config.encoding.fft_window_size / 4;
        Self {
            encoder: FourierEncoder::new(config.encoding.clone()),
            fragmenter: HologramFragmenter::new(fragment_size),
            weaver: RedundancyWeaver::new(config.encoding.redundancy_level),
            matcher: SimilarityMatcher::new(config.retrieval.similarity_threshold),
            recovery: PartialRecoveryEngine::new(config.encoding.redundancy_level),
            persistence: None,
            index: IndexBackend::Simple(HolographicIndex::new()),
            rs_codec: None,
            mmap: None,
            config,
            next_source_id: 1,
        }
    }

    /// 配置 WAL 持久化引擎
    pub fn with_persistence(mut self, data_dir: impl AsRef<Path>) -> Self {
        self.persistence = Some(PersistenceEngine::new(data_dir));
        self
    }

    /// 配置分层全息索引（L0热内存 + L1 LSM磁盘）
    pub fn with_tiered_index(mut self, tiered_config: TieredConfig) -> Result<Self, HoloError> {
        self.index = IndexBackend::Tiered(Box::new(TieredIndex::new(tiered_config)?));
        Ok(self)
    }

    /// 配置 Reed-Solomon 纠删码（data_shards 数据片 + parity_shards 校验片）
    pub fn with_reed_solomon(mut self, data_shards: usize, parity_shards: usize) -> Result<Self, HoloError> {
        let rs = ReedSolomon::new(data_shards, parity_shards)
            .map_err(|e| HoloError::ReedSolomon(e.to_string()))?;
        self.rs_codec = Some(rs);
        Ok(self)
    }

    /// 配置零拷贝 mmap 持久化
    pub fn with_mmap(mut self, dir: impl AsRef<Path>) -> Self {
        self.mmap = Some(MmapPersistence::new(dir));
        self
    }

    fn insert_fragment(&mut self, fragment: HologramFragment) -> FragmentId {
        match &mut self.index {
            IndexBackend::Simple(idx) => idx.insert(fragment),
            IndexBackend::Tiered(idx) => idx.insert(fragment.clone()).unwrap_or(fragment.id),
        }
    }

    fn get_by_source(&self, source_hash: u64) -> Vec<HologramFragment> {
        match &self.index {
            IndexBackend::Simple(idx) => idx.get_by_source(source_hash).into_iter().cloned().collect(),
            IndexBackend::Tiered(idx) => idx.get_by_source(source_hash).unwrap_or_default(),
        }
    }

    fn all_fragments(&self) -> Vec<HologramFragment> {
        match &self.index {
            IndexBackend::Simple(idx) => idx.all_fragments().into_iter().cloned().collect(),
            IndexBackend::Tiered(_) => Vec::new(),
        }
    }

    fn index_len(&self) -> usize {
        match &self.index {
            IndexBackend::Simple(idx) => idx.len(),
            IndexBackend::Tiered(idx) => idx.len(),
        }
    }

    fn integrity_check(&self, source_hash: u64) -> IntegrityReport {
        match &self.index {
            IndexBackend::Simple(idx) => idx.integrity_check(source_hash),
            IndexBackend::Tiered(idx) => idx.integrity_check(source_hash).unwrap_or_else(|_| IntegrityReport::new(0, 0)),
        }
    }

    fn simple_index(&self) -> Option<&HolographicIndex> {
        match &self.index {
            IndexBackend::Simple(idx) => Some(idx),
            _ => None,
        }
    }

    #[allow(dead_code)]
    fn simple_index_mut(&mut self) -> Option<&mut HolographicIndex> {
        match &mut self.index {
            IndexBackend::Simple(idx) => Some(idx),
            _ => None,
        }
    }

    /// 将数据编码为全息片段并存储到索引
    ///
    /// 返回包含 source_hash 的 StoreResult，可用于后续 retrieve 调用
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
            self.insert_fragment(fragment);
        }

        if let IndexBackend::Simple(ref simple_idx) = self.index {
            if let Some(ref mut persist) = self.persistence {
                let _ = persist.save_index(simple_idx, "main.idx");
            }
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

    /// 使用 RS 纠删码保护存储数据（需先配置 with_reed_solomon）
    pub fn store_with_rs(&mut self, data: &[f64]) -> Result<StoreResult, HoloError> {
        let rs = self.rs_codec.as_ref()
            .ok_or_else(|| HoloError::ReedSolomon("未配置RS纠删码".to_string()))?;

        let encode_result = self.encoder.encode(data);
        if encode_result.fragments.is_empty() {
            return Err(HoloError::Encode("编码结果为空".to_string()));
        }

        let source_hash = encode_result.source_hash;
        let fragment_count = encode_result.fragments.len();

        let woven = self.weaver.weave(&encode_result.fragments);

        let shard_data: Vec<Vec<u8>> = woven.iter()
            .map(|f| bincode::serialize(f).unwrap_or_default())
            .collect();

        let data_shards = rs.data_shards();
        let parity_shards = rs.parity_shards();

        let mut all_shards: Vec<Option<Vec<u8>>> = Vec::with_capacity(data_shards + parity_shards);
        for s in &shard_data {
            all_shards.push(Some(s.clone()));
        }

        if shard_data.len() >= data_shards {
            let data_for_rs: Vec<Vec<u8>> = shard_data[..data_shards].to_vec();
            let block_len = data_for_rs[0].len();
            let padded: Vec<Vec<u8>> = data_for_rs.iter()
                .map(|s| {
                    let mut p = s.clone();
                    while p.len() < block_len { p.push(0); }
                    p
                })
                .collect();
            if let Ok(parity) = rs.encode(&padded) {
                for p in parity {
                    all_shards.push(Some(p));
                }
            }
        }

        let mut fragment_ids = Vec::new();
        for fragment in &woven {
            fragment_ids.push(fragment.id);
            self.insert_fragment(fragment.clone());
        }

        let source_id = self.next_source_id;
        self.next_source_id += 1;

        Ok(StoreResult {
            source_id,
            source_hash,
            fragment_count,
            total_fragments: woven.len(),
            fragment_ids,
        })
    }

    /// 根据 source_hash 检索并解码数据
    pub fn retrieve(&mut self, source_hash: u64, expected_len: usize) -> Result<Vec<f64>, HoloError> {
        let fragments = self.get_by_source(source_hash);

        if fragments.is_empty() {
            return Err(HoloError::Decode(format!("未找到源 {}", source_hash)));
        }

        let unwoven = self.weaver.unweave(&fragments);
        let decoded = self.encoder.decode(&unwoven, expected_len);
        Ok(decoded)
    }

    /// 频域相似度检索，返回 top_k 个最相似的结果
    pub fn search(&mut self, query: &[f64], top_k: usize) -> Result<Vec<AssociatedItem>, HoloError> {
        let encode_result = self.encoder.encode(query);
        if encode_result.fragments.is_empty() {
            return Err(HoloError::Retrieval("查询编码结果为空".to_string()));
        }

        let candidates = self.all_fragments();

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let results = self.matcher.find_similar(&encode_result.fragments[0], &candidates, top_k);
        Ok(results)
    }

    pub fn integrity(&self, source_hash: u64) -> IntegrityReport {
        self.integrity_check(source_hash)
    }

    pub fn can_recover(&self, available: usize, total: usize) -> bool {
        self.recovery.can_recover(available, total as u32)
    }

    pub fn recover(&self, available: &[HologramFragment], total: u32) -> RetrievalResult {
        self.recovery.recover(available, total)
    }

    pub fn fragment_count(&self) -> usize {
        self.index_len()
    }

    pub fn source_count(&self) -> usize {
        match &self.index {
            IndexBackend::Simple(idx) => idx.all_source_hashes().len(),
            IndexBackend::Tiered(_) => 0,
        }
    }

    pub fn config(&self) -> &HolographicConfig {
        &self.config
    }

    pub fn rs_codec(&self) -> Option<&ReedSolomon> {
        self.rs_codec.as_ref()
    }

    pub fn save(&mut self) -> Result<(), HoloError> {
        match &self.index {
            IndexBackend::Simple(idx) => {
                if let Some(ref mut persist) = self.persistence {
                    persist.save_index(idx, "main.idx")?;
                } else {
                    return Err(HoloError::Persistence(crate::storage::persistence::PersistenceError::Io(
                        std::io::Error::new(std::io::ErrorKind::NotFound, "未配置持久化引擎"),
                    )));
                }
                Ok(())
            }
            IndexBackend::Tiered(_) => {
                Err(HoloError::Persistence(crate::storage::persistence::PersistenceError::Io(
                    std::io::Error::new(std::io::ErrorKind::NotFound, "分层索引不支持WAL持久化"),
                )))
            }
        }
    }

    pub fn load(&mut self) -> Result<(), HoloError> {
        if let Some(ref persist) = self.persistence {
            let loaded = persist.load_index("main.idx")?;
            self.index = IndexBackend::Simple(loaded);
        } else {
            return Err(HoloError::Persistence(crate::storage::persistence::PersistenceError::Io(
                std::io::Error::new(std::io::ErrorKind::NotFound, "未配置持久化引擎"),
            )));
        }
        Ok(())
    }

    pub fn save_mmap(&self, filename: &str) -> Result<(), HoloError> {
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| HoloError::Mmap(crate::storage::mmap_persistence::MmapError::InvalidFormat("未配置mmap持久化".to_string())))?;
        if let Some(idx) = self.simple_index() {
            mmap.write_index(idx, filename)?;
            Ok(())
        } else {
            Err(HoloError::Mmap(crate::storage::mmap_persistence::MmapError::InvalidFormat("分层索引不支持mmap写入".to_string())))
        }
    }

    pub fn load_mmap(&mut self, filename: &str) -> Result<(), HoloError> {
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| HoloError::Mmap(crate::storage::mmap_persistence::MmapError::InvalidFormat("未配置mmap持久化".to_string())))?;
        let loaded = mmap.read_index(filename)?;
        self.index = IndexBackend::Simple(loaded);
        Ok(())
    }

    pub fn store_with_fault_tolerance(
        &mut self,
        data: &[f64],
        simulate_damage_pct: f64,
    ) -> Result<FaultToleranceResult, HoloError> {
        let store_result = self.store(data)?;

        let fragments = self.get_by_source(store_result.source_hash);
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

/// 存储操作的结果
pub struct StoreResult {
    pub source_id: u64,
    pub source_hash: u64,
    pub fragment_count: usize,
    pub total_fragments: usize,
    pub fragment_ids: Vec<FragmentId>,
}

/// 容错测试的结果（含损毁模拟和 MSE 评估）
pub struct FaultToleranceResult {
    pub store: StoreResult,
    pub total_fragments: usize,
    pub available_fragments: usize,
    pub damage_ratio: f64,
    pub mse: f64,
    pub integrity: IntegrityReport,
}
