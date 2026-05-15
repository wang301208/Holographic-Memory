use holographic_memory::*;
use ndarray::Array2;
use num_complex::Complex64;

fn make_frag(id: u64, source: u64, freq_vals: Vec<f64>) -> HologramFragment {
    let freq: Vec<Complex64> = freq_vals.into_iter().map(|v| Complex64::new(v, 0.0)).collect();
    let cols = freq.len().max(1);
    HologramFragment {
        id,
        frequency_domain: Array2::from_shape_vec((1, cols), freq).unwrap(),
        phase_key: PhaseKey::zero(cols),
        redundancy_level: 1,
        metadata: FragmentMeta::new(source, 3, (id % 3) as u32),
    }
}

#[test]
fn test_reasoner_basic_inference() {
    let config = AttentionConfig {
        num_heads: 2,
        temperature: 1.0,
        top_k_attention: 5,
        coherence_threshold: 0.1,
        max_propagation_hops: 2,
        decay_factor: 0.5,
    };
    let reasoner = HolographicReasoner::new(config);

    let query = make_frag(100, 1, vec![1.0, 0.8, 0.6, 0.4]);
    let kb = vec![
        make_frag(1, 1, vec![0.9, 0.7, 0.5, 0.3]),
        make_frag(2, 1, vec![0.1, 0.2, 0.3, 0.4]),
        make_frag(3, 2, vec![0.8, 0.6, 0.4, 0.2]),
    ];

    let result = reasoner.reason(&query, &kb, 5);
    assert!(!result.conclusions.is_empty());
    assert!(result.coherence_score >= 0.0);
    assert!(!result.reasoning_chain.is_empty());
}

#[test]
fn test_reasoner_pattern_match() {
    let config = AttentionConfig {
        coherence_threshold: 0.1,
        ..Default::default()
    };
    let mut reasoner = HolographicReasoner::new(config);

    let pattern: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); 4];
    reasoner.register_pattern("sine", &pattern);
    assert_eq!(reasoner.pattern_count(), 1);

    let query = make_frag(100, 1, vec![1.0, 1.0, 1.0, 1.0]);
    let kb = vec![make_frag(1, 1, vec![0.9, 0.9, 0.9, 0.9])];

    let result = reasoner.reason(&query, &kb, 5);
    assert!(!result.conclusions.is_empty());
}

#[test]
fn test_reasoner_propagation_graph() {
    let mut reasoner = HolographicReasoner::new(AttentionConfig::default());

    let frags = vec![
        make_frag(1, 1, vec![1.0, 0.5, 0.0, 0.0]),
        make_frag(2, 1, vec![0.5, 1.0, 0.5, 0.0]),
        make_frag(3, 1, vec![0.0, 0.5, 1.0, 0.5]),
    ];

    reasoner.build_propagation_graph(&frags);
    assert!(reasoner.propagation_edge_count() > 0);

    let result = reasoner.reason(&frags[0], &frags, 5);
    assert!(!result.conclusions.is_empty());
}

#[test]
fn test_reasoner_multi_head_attention() {
    let config = AttentionConfig {
        num_heads: 4,
        temperature: 0.5,
        top_k_attention: 3,
        ..Default::default()
    };
    let reasoner = HolographicReasoner::new(config);

    let query = make_frag(100, 1, vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02]);
    let kb: Vec<HologramFragment> = (0..10)
        .map(|i| make_frag(i, 1, vec![1.0 - i as f64 * 0.1; 8]))
        .collect();

    let result = reasoner.reason(&query, &kb, 5);
    assert!(!result.conclusions.is_empty());
}

#[test]
fn test_reasoner_coherence_score() {
    let reasoner = HolographicReasoner::new(AttentionConfig::default());

    let query = make_frag(100, 1, vec![1.0, 0.5]);
    let kb = vec![
        make_frag(1, 1, vec![0.9, 0.45]),
        make_frag(2, 1, vec![0.8, 0.4]),
    ];

    let result = reasoner.reason(&query, &kb, 5);
    assert!(result.coherence_score >= 0.0);
    assert!(result.coherence_score <= 1.5);
}

#[test]
fn test_reasoner_empty_kb() {
    let reasoner = HolographicReasoner::new(AttentionConfig::default());
    let query = make_frag(100, 1, vec![1.0, 0.5]);
    let result = reasoner.reason(&query, &[], 5);
    assert!(result.conclusions.is_empty());
    assert_eq!(result.coherence_score, 0.0);
}

#[test]
fn test_cross_modal_text_image_bridge() {
    let mut reasoner = CrossModalReasoner::new();
    reasoner.register_text_image_bridge(4, 4);
    assert_eq!(reasoner.mapping_count(), 2);

    let text_frag = make_frag(1, 100, vec![1.0, 0.8, 0.6, 0.4]);
    let image_frags = vec![
        make_frag(10, 200, vec![0.9, 0.7, 0.5, 0.3]),
        make_frag(11, 200, vec![0.1, 0.2, 0.3, 0.4]),
    ];

    let results = reasoner.cross_modal_search(
        &text_frag, &Modality::Text, &image_frags, &Modality::Image, 5,
    );

    assert!(!results.is_empty());
    assert_eq!(results[0].source_modality, Modality::Text);
    assert_eq!(results[0].target_modality, Modality::Image);
}

#[test]
fn test_cross_modal_associations_compat() {
    let mut reasoner = CrossModalReasoner::new();
    reasoner.register_text_image_bridge(4, 4);

    let text_frag = make_frag(1, 100, vec![1.0, 0.5, 0.3, 0.1]);
    let image_frags = vec![make_frag(10, 200, vec![0.8, 0.4, 0.2, 0.1])];

    let results = reasoner.cross_modal_associations(
        &text_frag, &Modality::Text, &image_frags, &Modality::Image, 5,
    );
    assert!(!results.is_empty());
}

#[test]
fn test_modality_encoder_text() {
    let encoder = TextModalityEncoder::new(64);
    assert_eq!(encoder.embedding_dim(), 64);
    assert_eq!(encoder.modality(), Modality::Text);

    let input: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
    let output = encoder.encode(&input);
    assert_eq!(output.len(), 64);
    assert!(output.iter().all(|&v| v >= 0.0 && v <= 1.0));
}

#[test]
fn test_modality_encoder_image() {
    let encoder = ImageModalityEncoder::new(4);
    assert_eq!(encoder.modality(), Modality::Image);

    let input: Vec<f64> = (0..64).map(|i| (i as f64 * 0.05).cos()).collect();
    let output = encoder.encode(&input);
    assert!(!output.is_empty());
}

#[test]
fn test_adaptive_redundancy_scoring() {
    let mut ar = AdaptiveRedundancy::new(RedundancyStrategy::default());

    ar.record_access(1);
    ar.record_access(1);
    ar.record_access(1);
    ar.record_access(2);
    ar.advance_time(100);

    let score1 = ar.score_importance(1);
    let score2 = ar.score_importance(2);

    assert!(score1.score >= score2.score);
    assert!(score1.factors.access_frequency > score2.factors.access_frequency);
}

#[test]
fn test_adaptive_redundancy_decision() {
    let mut ar = AdaptiveRedundancy::new(RedundancyStrategy::default());

    for _ in 0..100 { ar.record_access(1); }
    ar.set_connectivity(1, 10.0);
    ar.advance_time(1);

    let decision = ar.decide(1);
    assert!(decision.redundancy_level >= 1);
    assert!(decision.estimated_survival_rate >= 0.0);
    assert!(decision.storage_overhead_ratio >= 0.0);
}

#[test]
fn test_adaptive_redundancy_critical_data() {
    let strategy = RedundancyStrategy {
        critical_redundancy: 5,
        rs_parity_critical: 4,
        ..Default::default()
    };
    let mut ar = AdaptiveRedundancy::new(strategy);

    for _ in 0..1000 { ar.record_access(1); }
    ar.set_connectivity(1, 50.0);
    ar.advance_time(1);

    let decision = ar.decide(1);
    assert!(decision.importance.score > 0.0);
}

#[test]
fn test_adaptive_redundancy_batch() {
    let mut ar = AdaptiveRedundancy::new(RedundancyStrategy::default());

    for i in 1..=10u64 {
        for _ in 0..i { ar.record_access(i); }
    }
    ar.advance_time(10);

    let ids: Vec<u64> = (1..=10).collect();
    let decisions = ar.decide_batch(&ids);
    assert_eq!(decisions.len(), 10);
}

#[test]
fn test_adaptive_redundancy_rs_config_suggestion() {
    let mut ar = AdaptiveRedundancy::new(RedundancyStrategy::default());

    for i in 1..=5u64 {
        for _ in 0..(i * 10) { ar.record_access(i); }
    }
    ar.advance_time(10);

    let ids: Vec<u64> = (1..=5).collect();
    let config = ar.suggest_rs_config(&ids);
    assert!(config.parity_shards >= 1);
    assert!(config.avg_importance >= 0.0);
    assert!(!config.distribution.is_empty());
}

#[test]
fn test_importance_level_ordering() {
    assert!(ImportanceLevel::Low < ImportanceLevel::Medium);
    assert!(ImportanceLevel::Medium < ImportanceLevel::High);
    assert!(ImportanceLevel::High < ImportanceLevel::Critical);
}

#[test]
fn test_cross_modal_mapping_apply() {
    let mapping = CrossModalMapping::new(Modality::Text, Modality::Image, 4, 4);
    let input = vec![1.0, 0.5, 0.3, 0.1];
    let output = mapping.apply(&input);
    assert_eq!(output.len(), 4);
}

#[test]
fn test_modality_display() {
    assert_eq!(format!("{}", Modality::Text), "text");
    assert_eq!(format!("{}", Modality::Image), "image");
    assert_eq!(format!("{}", Modality::Custom("lidar".to_string())), "custom:lidar");
}
