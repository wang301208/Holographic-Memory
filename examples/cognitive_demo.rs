//! 认知基础设施综合示例：全息推理 + 跨模态联想 + 自适应冗余

use holographic_memory::*;

fn main() {
    println!("=== 全息记忆 · 认知基础设施演示 ===\n");

    let config = HolographicConfig::default();
    let mut hm = HolographicMemory::new(config)
        .with_reasoner(AttentionConfig::default())
        .with_cross_modal()
        .with_adaptive_redundancy(RedundancyStrategy::default());

    let knowledge: Vec<(&str, Vec<f64>)> = vec![
        ("正弦波", (0..256).map(|i| (i as f64 * 0.05).sin()).collect()),
        ("余弦波", (0..256).map(|i| (i as f64 * 0.05).cos()).collect()),
        ("锯齿波", (0..256).map(|i| i as f64 / 256.0).collect()),
        ("方波", (0..256).map(|i| if i < 128 { 1.0 } else { -1.0 }).collect()),
        ("复合波", (0..256).map(|i| {
            (i as f64 * 0.05).sin() + 0.5 * (i as f64 * 0.1).cos()
        }).collect()),
        ("高频波", (0..256).map(|i| (i as f64 * 0.3).sin()).collect()),
    ];

    println!("[1] 自适应存储知识（含重要性评估）");
    let mut store_results = Vec::new();
    for (name, data) in &knowledge {
        let result = hm.adaptive_store(data).unwrap();
        let level = match result.redundancy_decision.importance.level {
            ImportanceLevel::Low => "低",
            ImportanceLevel::Medium => "中",
            ImportanceLevel::High => "高",
            ImportanceLevel::Critical => "关键",
        };
        println!(
            "  存储 '{}' → {} 片段, 重要性={}, 评分={:.3}, RS校验={}片, 冗余度={}, 存活率={:.1}%",
            name,
            result.store.total_fragments,
            level,
            result.redundancy_decision.importance.score,
            result.redundancy_decision.rs_parity_shards,
            result.redundancy_decision.redundancy_level,
            result.redundancy_decision.estimated_survival_rate * 100.0,
        );
        store_results.push(result);
    }

    hm.build_propagation_graph();
    hm.register_reasoning_pattern("sin_pattern", &(0..256).map(|i| (i as f64 * 0.05).sin()).collect::<Vec<_>>());

    println!("\n[2] 全息推理");
    let query: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin() + 0.3 * (i as f64 * 0.1).cos()).collect();
    let inference = hm.reason(&query, 3).unwrap();
    println!("  查询: sin + 0.3*cos 混合波");
    println!("  推理相干度: {:.4}", inference.coherence_score);
    println!("  推理链长度: {}", inference.reasoning_chain.len());
    for (i, conclusion) in inference.conclusions.iter().enumerate() {
        let rtype = match conclusion.reasoning_type {
            InferenceType::PatternMatch => "模式匹配",
            InferenceType::FrequencyPropagation => "频域传播",
            InferenceType::PhaseCoherence => "相位相干",
            InferenceType::CrossModal => "跨模态",
        };
        println!(
            "  结论#{}: 片段={:?}, 置信度={:.4}, 类型={}, 证据={}",
            i + 1, conclusion.fragment_id, conclusion.confidence, rtype, conclusion.evidence_count,
        );
    }

    println!("\n[3] 跨模态联想检索（文本→图像）");
    let text_query: Vec<f64> = (0..256).map(|i| (i as f64 * 0.08).sin()).collect();
    let cross_results = hm.cross_modal_search(&text_query, &Modality::Text, &Modality::Image, 3).unwrap();
    println!("  文本查询 → 图像模态联想:");
    for (i, assoc) in cross_results.iter().enumerate() {
        println!(
            "  关联#{}: 桥接置信度={:.4}, 源={:?} → 目标={:?}",
            i + 1, assoc.bridge_confidence, assoc.source_id, assoc.target_id,
        );
    }

    println!("\n[4] 时间推移后的自适应冗余重评估");
    hm.advance_redundancy_time(3600);
    if let Some(first_result) = store_results.first() {
        if let Some(primary_id) = first_result.store.fragment_ids.first() {
            let ar = hm.adaptive_redundancy();
            if let Some(ar) = ar {
                let new_decision = ar.decide(*primary_id);
                let level = match new_decision.importance.level {
                    ImportanceLevel::Low => "低",
                    ImportanceLevel::Medium => "中",
                    ImportanceLevel::High => "高",
                    ImportanceLevel::Critical => "关键",
                };
                println!(
                    "  1小时后重评估: 重要性={}, 评分={:.3} (recency={:.3})",
                    level, new_decision.importance.score, new_decision.importance.factors.recency,
                );
            }
        }
    }

    println!("\n=== 认知基础设施演示完成 ===");
}
