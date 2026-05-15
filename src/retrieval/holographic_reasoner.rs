use num_complex::Complex64;
use std::collections::{HashMap, HashSet};

use crate::foundation::math::cosine_similarity;
use crate::retrieval::similarity_matcher::SimilarityMatcher;
use crate::types::{FragmentId, HologramFragment};

/// 推理步骤：描述从查询到结论的一条推理路径
#[derive(Debug, Clone)]
pub struct InferenceStep {
    pub source_id: FragmentId,
    pub target_id: FragmentId,
    pub attention_weight: f64,
    pub reasoning: InferenceType,
}

/// 推理类型
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceType {
    PatternMatch,
    FrequencyPropagation,
    PhaseCoherence,
    CrossModal,
}

/// 推理结果
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub conclusions: Vec<InferenceConclusion>,
    pub reasoning_chain: Vec<InferenceStep>,
    pub coherence_score: f64,
}

/// 推理结论
#[derive(Debug, Clone)]
pub struct InferenceConclusion {
    pub fragment_id: FragmentId,
    pub confidence: f64,
    pub reasoning_type: InferenceType,
    pub evidence_count: usize,
    pub metadata: crate::types::FragmentMeta,
}

/// 频域注意力配置
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub temperature: f64,
    pub top_k_attention: usize,
    pub coherence_threshold: f64,
    pub max_propagation_hops: u32,
    pub decay_factor: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            temperature: 1.0,
            top_k_attention: 10,
            coherence_threshold: 0.3,
            max_propagation_hops: 3,
            decay_factor: 0.5,
        }
    }
}

/// 全息推理引擎：频域注意力 + 多跳频域传播 + 模式匹配推理
///
/// 核心原理：
/// - 注意力权重 ≈ 频域内积（相位匹配）
/// - 多跳联想 ≈ 频域卷积链
/// - 模式匹配 ≈ 频域模板相关性
pub struct HolographicReasoner {
    config: AttentionConfig,
    matcher: SimilarityMatcher,
    pattern_library: HashMap<String, Vec<Complex64>>,
    propagation_graph: HashMap<FragmentId, Vec<PropagationEdge>>,
}

struct PropagationEdge {
    target_id: FragmentId,
    transfer_weight: f64,
    coherence: f64,
}

impl HolographicReasoner {
    pub fn new(config: AttentionConfig) -> Self {
        let matcher = SimilarityMatcher::new(config.coherence_threshold);
        Self {
            config,
            matcher,
            pattern_library: HashMap::new(),
            propagation_graph: HashMap::new(),
        }
    }

    /// 频域注意力推理：查询片段在知识库上的注意力分布 + 推理
    pub fn reason(
        &self,
        query: &HologramFragment,
        knowledge_base: &[HologramFragment],
        top_k: usize,
    ) -> InferenceResult {
        let query_freq: Vec<Complex64> = query.frequency_domain.iter().copied().collect();

        let attention_weights = self.compute_attention(&query_freq, knowledge_base);

        let direct_conclusions: Vec<InferenceConclusion> = attention_weights
            .iter()
            .filter(|(_, weight)| *weight >= self.config.coherence_threshold)
            .map(|&(idx, weight)| InferenceConclusion {
                fragment_id: knowledge_base[idx].id,
                confidence: weight,
                reasoning_type: InferenceType::PatternMatch,
                evidence_count: 1,
                metadata: knowledge_base[idx].metadata.clone(),
            })
            .collect();

        let pattern_conclusions = self.pattern_match_reason(&query_freq, knowledge_base);

        let propagation_conclusions = self.frequency_propagation(
            query, knowledge_base, &direct_conclusions,
        );

        let mut all_conclusions: Vec<InferenceConclusion> = Vec::new();
        all_conclusions.extend(direct_conclusions);
        all_conclusions.extend(pattern_conclusions);
        all_conclusions.extend(propagation_conclusions);

        let mut seen: HashSet<FragmentId> = HashSet::new();
        all_conclusions.retain(|c| seen.insert(c.fragment_id));
        all_conclusions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        all_conclusions.truncate(top_k);

        let reasoning_chain = self.build_reasoning_chain(query, &all_conclusions, knowledge_base);

        let coherence_score = self.compute_overall_coherence(&query_freq, &all_conclusions, knowledge_base);

        InferenceResult {
            conclusions: all_conclusions,
            reasoning_chain,
            coherence_score,
        }
    }

    /// 计算频域注意力权重（多头注意力）
    fn compute_attention(
        &self,
        query_freq: &[Complex64],
        knowledge_base: &[HologramFragment],
    ) -> Vec<(usize, f64)> {
        let n = query_freq.len();
        if n == 0 {
            return Vec::new();
        }

        let head_size = n.div_ceil(self.config.num_heads);

        let mut accumulated: Vec<f64> = vec![0.0; knowledge_base.len()];

        for head_idx in 0..self.config.num_heads {
            let start = head_idx * head_size;
            if start >= n {
                break;
            }
            let end = (start + head_size).min(n);

            let query_head: Vec<Complex64> = query_freq[start..end].to_vec();
            let query_norm: f64 = query_head.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if query_norm < 1e-10 {
                continue;
            }

            for (kb_idx, fragment) in knowledge_base.iter().enumerate() {
                let kb_freq: Vec<Complex64> = fragment.frequency_domain.iter().copied().collect();
                if kb_freq.len() < end {
                    continue;
                }
                let kb_head: Vec<Complex64> = kb_freq[start..end].to_vec();

                let sim = cosine_similarity(&query_head, &kb_head);
                let attention = (sim / self.config.temperature).exp()
                    / (1.0 / self.config.temperature).exp();
                accumulated[kb_idx] += attention;
            }
        }

        let mut scored: Vec<(usize, f64)> = accumulated
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score / self.config.num_heads as f64))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k_attention);
        scored
    }

    /// 模式匹配推理：查询与已注册频域模板的匹配
    fn pattern_match_reason(
        &self,
        query_freq: &[Complex64],
        knowledge_base: &[HologramFragment],
    ) -> Vec<InferenceConclusion> {
        if self.pattern_library.is_empty() {
            return Vec::new();
        }

        let mut conclusions = Vec::new();

        for pattern_freq in self.pattern_library.values() {
            let sim = cosine_similarity(query_freq, pattern_freq);
            if sim < self.config.coherence_threshold {
                continue;
            }

            for fragment in knowledge_base {
                let frag_freq: Vec<Complex64> = fragment.frequency_domain.iter().copied().collect();
                let pattern_sim = cosine_similarity(pattern_freq, &frag_freq);

                if pattern_sim >= self.config.coherence_threshold {
                    let confidence = sim * pattern_sim * 0.8;
                    conclusions.push(InferenceConclusion {
                        fragment_id: fragment.id,
                        confidence,
                        reasoning_type: InferenceType::PhaseCoherence,
                        evidence_count: 2,
                        metadata: fragment.metadata.clone(),
                    });
                }
            }
        }

        conclusions
    }

    /// 频域传播推理：沿传播图多跳推理
    fn frequency_propagation(
        &self,
        query: &HologramFragment,
        _knowledge_base: &[HologramFragment],
        direct_conclusions: &[InferenceConclusion],
    ) -> Vec<InferenceConclusion> {
        if self.propagation_graph.is_empty() {
            return Vec::new();
        }

        let mut conclusions = Vec::new();
        let mut visited: HashSet<FragmentId> = HashSet::new();
        visited.insert(query.id);
        for c in direct_conclusions {
            visited.insert(c.fragment_id);
        }

        for direct in direct_conclusions {
            self.propagate_hops(
                direct.fragment_id,
                direct.confidence,
                0,
                &mut visited,
                &mut conclusions,
            );
        }

        conclusions
    }

    fn propagate_hops(
        &self,
        from_id: FragmentId,
        from_confidence: f64,
        hop: u32,
        visited: &mut HashSet<FragmentId>,
        results: &mut Vec<InferenceConclusion>,
    ) {
        if hop >= self.config.max_propagation_hops {
            return;
        }

        if let Some(edges) = self.propagation_graph.get(&from_id) {
            for edge in edges {
                if visited.contains(&edge.target_id) {
                    continue;
                }
                visited.insert(edge.target_id);

                let decay = self.config.decay_factor.powi(hop as i32 + 1);
                let confidence = from_confidence * edge.transfer_weight * decay * edge.coherence;

                if confidence >= self.config.coherence_threshold * 0.5 {
                    results.push(InferenceConclusion {
                        fragment_id: edge.target_id,
                        confidence,
                        reasoning_type: InferenceType::FrequencyPropagation,
                        evidence_count: (hop + 2) as usize,
                        metadata: crate::types::FragmentMeta::new(0, 0, 0),
                    });

                    self.propagate_hops(edge.target_id, confidence, hop + 1, visited, results);
                }
            }
        }
    }

    /// 构建推理链（可解释性）
    fn build_reasoning_chain(
        &self,
        query: &HologramFragment,
        conclusions: &[InferenceConclusion],
        _knowledge_base: &[HologramFragment],
    ) -> Vec<InferenceStep> {
        conclusions
            .iter()
            .map(|c| InferenceStep {
                source_id: query.id,
                target_id: c.fragment_id,
                attention_weight: c.confidence,
                reasoning: c.reasoning_type.clone(),
            })
            .collect()
    }

    /// 计算整体相干度（推理一致性评估）
    fn compute_overall_coherence(
        &self,
        query_freq: &[Complex64],
        conclusions: &[InferenceConclusion],
        knowledge_base: &[HologramFragment],
    ) -> f64 {
        if conclusions.is_empty() {
            return 0.0;
        }

        let conclusion_ids: HashSet<FragmentId> = conclusions.iter().map(|c| c.fragment_id).collect();

        let conclusion_freqs: Vec<Vec<Complex64>> = knowledge_base
            .iter()
            .filter(|f| conclusion_ids.contains(&f.id))
            .map(|f| f.frequency_domain.iter().copied().collect())
            .collect();

        if conclusion_freqs.len() < 2 {
            return conclusions.iter().map(|c| c.confidence).sum::<f64>() / conclusions.len() as f64;
        }

        let mut pairwise_coherence = 0.0;
        let mut pair_count = 0usize;

        for i in 0..conclusion_freqs.len() {
            for j in (i + 1)..conclusion_freqs.len() {
                let sim = cosine_similarity(&conclusion_freqs[i], &conclusion_freqs[j]);
                pairwise_coherence += sim;
                pair_count += 1;
            }
        }

        let avg_pairwise = if pair_count > 0 { pairwise_coherence / pair_count as f64 } else { 0.0 };
        let query_coherence = conclusion_freqs
            .iter()
            .map(|cf| cosine_similarity(query_freq, cf))
            .sum::<f64>()
            / conclusion_freqs.len() as f64;

        0.5 * avg_pairwise + 0.5 * query_coherence
    }

    /// 注册频域模式模板（用于模式匹配推理）
    pub fn register_pattern(&mut self, name: &str, pattern: &[Complex64]) {
        self.pattern_library.insert(name.to_string(), pattern.to_vec());
    }

    /// 从片段提取并注册频域模式
    pub fn register_pattern_from_fragment(&mut self, name: &str, fragment: &HologramFragment) {
        let freq: Vec<Complex64> = fragment.frequency_domain.iter().copied().collect();
        self.pattern_library.insert(name.to_string(), freq);
    }

    /// 构建频域传播图
    pub fn build_propagation_graph(&mut self, fragments: &[HologramFragment]) {
        self.propagation_graph.clear();

        for fragment in fragments.iter() {
            let similar = self.matcher.find_similar(fragment, fragments, 10);

            let edges: Vec<PropagationEdge> = similar
                .iter()
                .filter(|item| item.fragment_id != fragment.id)
                .map(|item| {
                    let source_freq: Vec<Complex64> = fragment.frequency_domain.iter().copied().collect();
                    let coherence = if !source_freq.is_empty() {
                        let energy: f64 = source_freq.iter().map(|c| c.norm_sqr()).sum();
                        if energy > 0.0 { item.similarity * energy.ln_1p() / (energy + 1.0).ln_1p() } else { 0.0 }
                    } else { 0.0 };

                    PropagationEdge {
                        target_id: item.fragment_id,
                        transfer_weight: item.similarity,
                        coherence: coherence.abs().min(1.0),
                    }
                })
                .collect();

            if !edges.is_empty() {
                self.propagation_graph.insert(fragment.id, edges);
            }
        }
    }

    /// 获取已注册模式数量
    pub fn pattern_count(&self) -> usize {
        self.pattern_library.len()
    }

    /// 获取传播图边数
    pub fn propagation_edge_count(&self) -> usize {
        self.propagation_graph.values().map(|v| v.len()).sum::<usize>() / 2
    }
}

impl Default for HolographicReasoner {
    fn default() -> Self {
        Self::new(AttentionConfig::default())
    }
}
