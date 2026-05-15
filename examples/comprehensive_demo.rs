use holographic_memory::*;

fn main() {
    println!("=== 全息记忆 v0.3.0 端到端综合示例 ===\n");

    let config = HolographicConfig::default();
    let mut hm = HolographicMemory::new(config);

    println!("[1] 基础存取");
    let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = hm.store(&data).unwrap();
    println!("  存入: {} 片段, source_hash={}", result.total_fragments, result.source_hash);

    let recovered = hm.retrieve(result.source_hash, data.len()).unwrap();
    let mse: f64 = data.iter().zip(recovered.iter())
        .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / data.len() as f64;
    println!("  取出: MSE={:.2e}", mse);

    println!("\n[2] 容错恢复");
    let ft = hm.store_with_fault_tolerance(&data, 0.3).unwrap();
    println!("  30%损毁: 可用{}/{}, MSE={:.2e}, 可恢复={}",
        ft.available_fragments, ft.total_fragments, ft.mse, ft.integrity.recovery_possible);

    println!("\n[3] Reed-Solomon 纠删码");
    let rs = ReedSolomon::new(3, 2).unwrap();
    println!("  {}", rs);
    let shards: Vec<Vec<u8>> = vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]];
    let parity = rs.encode(&shards).unwrap();
    println!("  编码: 3数据+2校验, 验证={}", rs.verify(&shards, &parity));

    println!("\n[4] 量子启发编码");
    let mut qe = QuantumEncoder::new(8);
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let state = qe.encode_superposition(&signal);
    println!("  {}", state);
    println!("  熵={:.4}, 概率分布前4={:.4} {:.4} {:.4} {:.4}",
        state.entropy(),
        state.probability_distribution()[0],
        state.probability_distribution()[1],
        state.probability_distribution()[2],
        state.probability_distribution()[3]);

    let state2 = qe.encode_superposition(&vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let (interfered, pattern) = qe.interfere(&state, &state2);
    println!("  干涉: {}", pattern);
    println!("  干涉态范数={:.4}", interfered.norm);

    println!("\n[5] SIMD 加速运算");
    let a: Vec<f64> = (0..1024).map(|i| (i as f64).sin()).collect();
    let b: Vec<f64> = (0..1024).map(|i| (i as f64).cos()).collect();
    let dot = SimdOps::dot_product(&a, &b);
    let cos_sim = SimdOps::cosine_similarity(&a, &b);
    println!("  内积={:.4}, 余弦相似度={:.6}", dot, cos_sim);

    let hadamard = SimdOps::hadamard_transform(&[1.0, 0.0, 0.0, 1.0]);
    let recovered_h = SimdOps::inverse_hadamard(&hadamard);
    println!("  Hadamard变换往返: [{:.1}, {:.1}, {:.1}, {:.1}]",
        recovered_h[0], recovered_h[1], recovered_h[2], recovered_h[3]);

    println!("\n[6] 分层索引");
    let tiered_dir = std::env::temp_dir().join("holo_tiered_demo_l1");
    let _ = std::fs::remove_dir_all(&tiered_dir);
    let tiered_config = TieredConfig {
        l0_capacity: 50,
        l1_memtable_capacity: 50,
        l1_dir: tiered_dir.clone(),
        promote_threshold: 3,
        demote_after_access: false,
    };
    {
        let mut tiered = TieredIndex::new(tiered_config).unwrap();
        for i in 1..=10u64 {
            let frag = HologramFragment {
                id: i,
                frequency_domain: ndarray::Array2::from_shape_vec((1,1), vec![num_complex::Complex64::new(i as f64, 0.0)]).unwrap(),
                phase_key: PhaseKey::zero(1),
                redundancy_level: 0,
                metadata: FragmentMeta::new(100, 10, (i-1) as u32),
            };
            tiered.insert(frag).unwrap();
        }
        let stats = tiered.stats();
        println!("  {}", stats);
        let _ = std::fs::remove_dir_all(&tiered_dir);
    }

    println!("\n[7] 自适应窗口");
    let selector = AdaptiveWindowSelector::default();
    let signal = vec![1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0, 0.5]; // 周期信号
    let result = selector.select(&signal);
    println!("  窗口={}, 重叠={:.0}%, 原因={}", result.window_size, result.overlap_ratio * 100.0, result.reasoning);

    let analysis = selector.analyze(&signal);
    println!("  谱平坦度={:.4}, ZCR={:.4}, RMS={:.4}", 
        analysis.spectral_flatness, analysis.zero_crossing_rate, analysis.rms);

    println!("\n=== 示例完成 ===");
}
