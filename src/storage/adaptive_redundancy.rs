use std::collections::HashMap;

use crate::types::FragmentId;

/// 知识重要性等级
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImportanceLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// 知识重要性评分
#[derive(Debug, Clone)]
pub struct ImportanceScore {
    pub level: ImportanceLevel,
    pub score: f64,
    pub factors: ImportanceFactors,
}

/// 重要性评估因子
#[derive(Debug, Clone)]
pub struct ImportanceFactors {
    pub access_frequency: f64,
    pub recency: f64,
    pub connectivity: f64,
    pub reconstruction_value: f64,
}

/// 冗余策略：根据重要性决定冗余度
#[derive(Debug, Clone)]
pub struct RedundancyStrategy {
    pub low_redundancy: u8,
    pub medium_redundancy: u8,
    pub high_redundancy: u8,
    pub critical_redundancy: u8,
    pub rs_data_shards: usize,
    pub rs_parity_low: usize,
    pub rs_parity_medium: usize,
    pub rs_parity_high: usize,
    pub rs_parity_critical: usize,
}

impl Default for RedundancyStrategy {
    fn default() -> Self {
        Self {
            low_redundancy: 1,
            medium_redundancy: 2,
            high_redundancy: 3,
            critical_redundancy: 4,
            rs_data_shards: 4,
            rs_parity_low: 1,
            rs_parity_medium: 2,
            rs_parity_high: 3,
            rs_parity_critical: 4,
        }
    }
}

/// 自适应冗余决策结果
#[derive(Debug, Clone)]
pub struct AdaptiveRedundancyDecision {
    pub importance: ImportanceScore,
    pub redundancy_level: u8,
    pub rs_parity_shards: usize,
    pub estimated_survival_rate: f64,
    pub storage_overhead_ratio: f64,
}

/// 自适应冗余引擎：按知识重要性动态调整 RS 冗余度
///
/// 核心原理：
/// - 访问频率高的知识 → 高冗余（快速恢复需求）
/// - 近期访问的知识 → 中高冗余（热点数据）
/// - 连接度高的知识 → 高冗余（关联枢纽）
/// - 重建价值高的知识 → 关键冗余（不可替代）
pub struct AdaptiveRedundancy {
    strategy: RedundancyStrategy,
    access_counts: HashMap<FragmentId, u64>,
    last_access_time: HashMap<FragmentId, u64>,
    connectivity_scores: HashMap<FragmentId, f64>,
    total_accesses: u64,
    now_secs: u64,
}

impl AdaptiveRedundancy {
    pub fn new(strategy: RedundancyStrategy) -> Self {
        Self {
            strategy,
            access_counts: HashMap::new(),
            last_access_time: HashMap::new(),
            connectivity_scores: HashMap::new(),
            total_accesses: 0,
            now_secs: 0,
        }
    }

    /// 记录片段访问
    pub fn record_access(&mut self, fragment_id: FragmentId) {
        *self.access_counts.entry(fragment_id).or_insert(0) += 1;
        self.total_accesses += 1;
        self.last_access_time.insert(fragment_id, self.now_secs);
    }

    /// 批量记录访问
    pub fn record_accesses(&mut self, fragment_ids: &[FragmentId]) {
        for &id in fragment_ids {
            self.record_access(id);
        }
    }

    /// 记录连接度（从联想图计算）
    pub fn set_connectivity(&mut self, fragment_id: FragmentId, degree: f64) {
        self.connectivity_scores.insert(fragment_id, degree);
    }

    /// 推进时间（模拟时间流逝）
    pub fn advance_time(&mut self, secs: u64) {
        self.now_secs += secs;
    }

    /// 计算知识重要性评分
    pub fn score_importance(&self, fragment_id: FragmentId) -> ImportanceScore {
        let access_count = self.access_counts.get(&fragment_id).copied().unwrap_or(0);
        let last_access = self.last_access_time.get(&fragment_id).copied().unwrap_or(0);
        let connectivity = self.connectivity_scores.get(&fragment_id).copied().unwrap_or(0.0);

        let access_frequency = if self.total_accesses > 0 {
            access_count as f64 / self.total_accesses as f64
        } else {
            0.0
        };

        let elapsed = self.now_secs.saturating_sub(last_access);
        let recency = if elapsed == 0 && last_access > 0 {
            1.0
        } else if elapsed > 0 {
            1.0 / (1.0 + (elapsed as f64).ln_1p())
        } else {
            0.0
        };

        let normalized_connectivity = 1.0 - 1.0 / (1.0 + connectivity);

        let reconstruction_value = 0.3 * access_frequency
            + 0.2 * recency
            + 0.3 * normalized_connectivity
            + 0.2 * access_frequency * normalized_connectivity;

        let factors = ImportanceFactors {
            access_frequency,
            recency,
            connectivity: normalized_connectivity,
            reconstruction_value,
        };

        let score = 0.3 * access_frequency
            + 0.25 * recency
            + 0.25 * normalized_connectivity
            + 0.2 * reconstruction_value;

        let level = if score >= 0.75 {
            ImportanceLevel::Critical
        } else if score >= 0.5 {
            ImportanceLevel::High
        } else if score >= 0.25 {
            ImportanceLevel::Medium
        } else {
            ImportanceLevel::Low
        };

        ImportanceScore { level, score, factors }
    }

    /// 自适应冗余决策：根据重要性选择冗余等级和 RS 校验片数
    pub fn decide(&self, fragment_id: FragmentId) -> AdaptiveRedundancyDecision {
        let importance = self.score_importance(fragment_id);

        let (redundancy_level, rs_parity_shards) = match importance.level {
            ImportanceLevel::Low => (self.strategy.low_redundancy, self.strategy.rs_parity_low),
            ImportanceLevel::Medium => (self.strategy.medium_redundancy, self.strategy.rs_parity_medium),
            ImportanceLevel::High => (self.strategy.high_redundancy, self.strategy.rs_parity_high),
            ImportanceLevel::Critical => (self.strategy.critical_redundancy, self.strategy.rs_parity_critical),
        };

        let total_shards = self.strategy.rs_data_shards + rs_parity_shards;
        let max_tolerable = rs_parity_shards;
        let estimated_survival_rate = if total_shards > 0 {
            1.0 - (max_tolerable as f64 / total_shards as f64).powi(total_shards as i32)
        } else {
            0.0
        };

        let storage_overhead_ratio = if self.strategy.rs_data_shards > 0 {
            (redundancy_level as f64 - 1.0).max(0.0) + rs_parity_shards as f64 / self.strategy.rs_data_shards as f64
        } else {
            0.0
        };

        AdaptiveRedundancyDecision {
            importance,
            redundancy_level,
            rs_parity_shards,
            estimated_survival_rate,
            storage_overhead_ratio,
        }
    }

    /// 批量决策：对多个片段做自适应冗余决策
    pub fn decide_batch(&self, fragment_ids: &[FragmentId]) -> Vec<(FragmentId, AdaptiveRedundancyDecision)> {
        fragment_ids
            .iter()
            .map(|&id| (id, self.decide(id)))
            .collect()
    }

    /// 基于决策结果建议最优 RS 配置
    pub fn suggest_rs_config(&self, fragment_ids: &[FragmentId]) -> SuggestedRsConfig {
        let decisions = self.decide_batch(fragment_ids);

        if decisions.is_empty() {
            return SuggestedRsConfig {
                data_shards: self.strategy.rs_data_shards,
                parity_shards: self.strategy.rs_parity_low,
                avg_importance: 0.0,
                distribution: HashMap::new(),
            };
        }

        let avg_parity: f64 = decisions
            .iter()
            .map(|(_, d)| d.rs_parity_shards as f64)
            .sum::<f64>()
            / decisions.len() as f64;

        let avg_importance: f64 = decisions
            .iter()
            .map(|(_, d)| d.importance.score)
            .sum::<f64>()
            / decisions.len() as f64;

        let mut distribution: HashMap<ImportanceLevel, usize> = HashMap::new();
        for (_, decision) in &decisions {
            *distribution.entry(decision.importance.level.clone()).or_insert(0) += 1;
        }

        let recommended_parity = avg_parity.round() as usize;

        SuggestedRsConfig {
            data_shards: self.strategy.rs_data_shards,
            parity_shards: recommended_parity.max(1),
            avg_importance,
            distribution,
        }
    }

    /// 获取访问统计
    pub fn access_stats(&self) -> (usize, u64) {
        (self.access_counts.len(), self.total_accesses)
    }
}

/// 建议的 RS 配置
#[derive(Debug)]
pub struct SuggestedRsConfig {
    pub data_shards: usize,
    pub parity_shards: usize,
    pub avg_importance: f64,
    pub distribution: HashMap<ImportanceLevel, usize>,
}

impl Default for AdaptiveRedundancy {
    fn default() -> Self {
        Self::new(RedundancyStrategy::default())
    }
}
