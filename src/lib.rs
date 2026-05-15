//! # 全息记忆存储引擎
//!
//! 基于 FFT 的高容错知识表示与检索系统。将知识通过傅里叶变换映射到频域，
//! 每个存储片段都包含整体信息的缩影，实现超强容错性和联想检索能力。
//!
//! ## 核心特性
//!
//! - 频域全息编码：FFT + Hann 窗，MSE ~1e-31
//! - 50% 数据损毁后仍可恢复
//! - GF(2^8) Reed-Solomon 纠删码
//! - 频域余弦相似度 + 多跳联想检索
//! - 分层索引（L0 热 + L1 LSM 磁盘）
//! - 零拷贝 mmap 持久化
//! - SIMD 加速运算
//!
//! ## 快速开始
//!
//! ```rust
//! use holographic_memory::*;
//!
//! let config = HolographicConfig::default();
//! let mut hm = HolographicMemory::new(config);
//!
//! let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.02).sin()).collect();
//! let result = hm.store(&data).unwrap();
//! let decoded = hm.retrieve(result.source_hash, data.len()).unwrap();
//! ```

#![warn(unsafe_code)]

pub mod foundation;
pub mod codec;
pub mod storage;
pub mod retrieval;
pub mod types;
pub mod api;
pub mod holographic;
pub mod distributed;

pub use types::{
    AssociatedItem, FragmentId, FragmentMeta, HologramFragment, IntegrityReport, PhaseKey,
    RetrievalResult,
};
pub use foundation::config::{HolographicConfig, EncodingConfig, StorageConfig, RetrievalConfig};
pub use foundation::math::FourierTransformer;
pub use foundation::simd_ops::SimdOps;
pub use codec::fourier_encoder::FourierEncoder;
pub use codec::hologram_fragmenter::HologramFragmenter;
pub use codec::redundancy_weaver::RedundancyWeaver;
pub use storage::holographic_index::HolographicIndex;
pub use storage::segment_manager::SegmentManager;
pub use storage::persistence::PersistenceEngine;
pub use retrieval::similarity_matcher::SimilarityMatcher;
pub use retrieval::associative_search::AssociativeSearchEngine;
pub use retrieval::partial_recovery::PartialRecoveryEngine;
pub use codec::parallel_encoder::ParallelEncoder;
pub use codec::sparse_encoder::{SparseEncoder, SparseFragment, EnergyReport};
pub use codec::adaptive_window::{AdaptiveWindowSelector, AdaptiveResult, SignalAnalysis};
pub use codec::reed_solomon::{ReedSolomon, RsError};
pub use codec::quantum_encoder::{QuantumEncoder, SuperpositionState, InterferencePattern, QuantumEncodedData};
pub use storage::lsm_index::{LsmIndex, LsmError, LsmStats, LevelStats};
pub use storage::mmap_persistence::{MmapPersistence, MmapReader, MmapError};
pub use storage::tiered_index::{TieredIndex, TieredConfig, TieredError, TieredStats, Layer};

#[cfg(feature = "http")]
pub use api::http::{AppState, create_router, serve,
    StoreRequest, StoreResponse, RetrieveRequest, RetrieveResponse,
    SearchRequest, SearchResponse, SearchResultItem,
    IntegrityRequest, IntegrityResponse, RecoverRequest, RecoverResponse,
    StatusResponse, ApiError,
};
pub use retrieval::holographic_reasoner::{
    HolographicReasoner, AttentionConfig, InferenceResult, InferenceConclusion,
    InferenceStep, InferenceType,
};
pub use codec::cross_modal::{
    CrossModalReasoner, CrossModalMapping, CrossModalAssociation,
    Modality, ModalityEncoder, TextModalityEncoder, ImageModalityEncoder,
};
pub use storage::adaptive_redundancy::{
    AdaptiveRedundancy, RedundancyStrategy, AdaptiveRedundancyDecision,
    ImportanceScore, ImportanceFactors, ImportanceLevel, SuggestedRsConfig,
};
pub use holographic::{HolographicMemory, HoloError, StoreResult, FaultToleranceResult, AdaptiveStoreResult};
pub use distributed::{
    ConsistentHashRing, VirtualNode, RingError,
    GossipProtocol, GossipMessage, GossipConfig, MembershipState, NodeInfo, NodeStatus,
    AntiEntropyRepair, RepairTask, RepairStatus, MerkleTree, MerkleNode, DiffReport,
    DistributedHolographicNode, DistributedConfig, DistributedError,
    DistributedStoreResult, DistributedRetrieveResult,
};
