use holographic_memory::*;

fn main() {
    println!("=== 全息记忆存储 - 完整演示 ===\n");

    let config = HolographicConfig::default();
    println!("配置: FFT窗口={}, 冗余等级={}, 重叠率={}\n",
        config.encoding.fft_window_size,
        config.encoding.redundancy_level,
        config.encoding.overlap_ratio,
    );

    let data: Vec<f64> = (0..512).map(|i| {
        let t = i as f64 * 0.01;
        (t * 3.0).sin() + 0.5 * (t * 7.0).cos()
    }).collect();
    println!("1. 输入数据: {} 个采样点（多频信号）", data.len());

    let mut encoder = FourierEncoder::new(config.encoding.clone());
    let encode_result = encoder.encode(&data);
    println!("   编码完成: {} 个全息片段, source_hash={}", encode_result.fragments.len(), encode_result.source_hash);

    let decoded_full = encoder.decode(&encode_result.fragments, data.len());
    let mse_full: f64 = data[64..448].iter().zip(decoded_full[64..448].iter())
        .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 384.0;
    println!("   完整解码 MSE(内部): {:.2e}\n", mse_full);

    println!("2. 全息分片");
    let fragmenter = HologramFragmenter::new(128);
    let holo_fragments = fragmenter.fragment(&encode_result.fragments);
    println!("   分片后: {} 个全息片段（每片段含整体缩影）\n", holo_fragments.len());

    println!("3. 冗余交织");
    let weaver = RedundancyWeaver::new(config.encoding.redundancy_level);
    let woven = weaver.weave(&encode_result.fragments);
    println!("   原始片段: {}, 交织后: {}", encode_result.fragments.len(), woven.len());
    let unwoven = weaver.unweave(&woven);
    println!("   解织后: {} 个原始片段\n", unwoven.len());

    println!("4. 容错性测试");
    let total = encode_result.fragments.len();
    for &damage_pct in &[0.0, 0.2, 0.3, 0.5] {
        let remove_count = (total as f64 * damage_pct) as usize;
        let available: Vec<HologramFragment> = encode_result.fragments.iter()
            .skip(remove_count).cloned().collect();
        let decoded = encoder.decode(&available, data.len());
        let mse: f64 = data[64..448].iter().zip(decoded[64..448].iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 384.0;
        let integrity = IntegrityReport::new(total as u32, available.len() as u32);
        println!("   损毁 {:.0}%: 可恢复={}, MSE={:.2e}, 损毁率={:.2}",
            damage_pct * 100.0, integrity.recovery_possible, mse, integrity.damage_ratio);
    }

    println!("\n5. 索引与检索");
    let mut index = HolographicIndex::new();
    for fragment in &encode_result.fragments {
        index.insert((*fragment).clone());
    }
    println!("   索引中共 {} 个片段", index.len());
    let integrity = index.integrity_check(encode_result.source_hash);
    println!("   完整性: 总片段={}, 可用={}, 损毁率={:.2}", 
        integrity.fragments_total, integrity.fragments_available, integrity.damage_ratio);

    let matcher = SimilarityMatcher::new(0.0);
    if encode_result.fragments.len() > 1 {
        let sim = matcher.similarity(&encode_result.fragments[0], &encode_result.fragments[1]);
        println!("   片段间相似度: {:.4}", sim);
    }

    println!("\n6. 部分恢复引擎");
    let recovery = PartialRecoveryEngine::new(config.encoding.redundancy_level);
    for &avail in &[10, 8, 5, 3] {
        let can = recovery.can_recover(avail as usize, 10);
        let conf = recovery.estimate_confidence(avail as usize, 10);
        println!("   10片段中可用{}: 可恢复={}, 置信度={:.2}", avail, can, conf);
    }

    println!("\n=== 演示完成 ===");
}
