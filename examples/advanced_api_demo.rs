use holographic_memory::*;

fn main() {
    println!("=== 全息记忆存储 - 高级API演示 ===\n");

    let config = HolographicConfig::default();
    println!("1. 配置初始化");
    println!("   FFT窗口: {}", config.encoding.fft_window_size);
    println!("   冗余等级: {}", config.encoding.redundancy_level);
    println!("   检索Top-K: {}", config.retrieval.top_k);

    println!("\n2. 编码与存储");
    let signals: Vec<(&str, Vec<f64>)> = vec![
        ("低频正弦", (0..256).map(|i| (i as f64 * 0.02).sin()).collect()),
        ("高频余弦", (0..256).map(|i| (i as f64 * 0.15).cos()).collect()),
        ("混合信号", (0..256).map(|i| {
            let t = i as f64 * 0.01;
            (t * 2.0).sin() + 0.5 * (t * 8.0).cos()
        }).collect()),
    ];

    let mut index = HolographicIndex::new();
    let mut encoder = FourierEncoder::new(config.encoding.clone());

    for (name, data) in &signals {
        let result = encoder.encode(data);
        println!("   '{}': {} 片段, source_hash={}", name, result.fragments.len(), result.source_hash);
        for frag in &result.fragments {
            index.insert((*frag).clone());
        }
    }

    println!("\n3. 索引状态");
    println!("   总片段: {}", index.len());
    for &source in &index.all_source_hashes() {
        let integrity = index.integrity_check(source);
        println!("   源 {}: {} 片段, 损毁率={:.2}", source, integrity.fragments_available, integrity.damage_ratio);
    }

    println!("\n4. 相似度检索");
    let query: Vec<f64> = (0..256).map(|i| (i as f64 * 0.02).sin()).collect();
    let query_result = encoder.encode(&query);
    if !query_result.fragments.is_empty() {
        let matcher = SimilarityMatcher::new(0.0);
        let candidates: Vec<HologramFragment> = index.all_fragments().into_iter().cloned().collect();
        let results = matcher.find_similar(&query_result.fragments[0], &candidates, 5);
        println!("   查询='低频正弦', Top-{} 结果:", results.len());
        for (i, r) in results.iter().enumerate() {
            println!("     #{}: 片段id={}, 相似度={:.6}", i+1, r.fragment_id, r.similarity);
        }
    }

    println!("\n5. 联想检索");
    let mut assoc_engine = AssociativeSearchEngine::new(0.1, 2);
    let all_frags: Vec<HologramFragment> = index.all_fragments().into_iter().cloned().collect();
    assoc_engine.build_associations(&all_frags);
    println!("   联想边数: {}", assoc_engine.association_count());

    println!("\n6. 冗余与恢复");
    let weaver = RedundancyWeaver::new(config.encoding.redundancy_level);
    let low_freq_data = &signals[0].1;
    let low_result = encoder.encode(low_freq_data);
    let woven = weaver.weave(&low_result.fragments);
    println!("   原始{}片段 → 交织后{}片段", low_result.fragments.len(), woven.len());

    let total = low_result.fragments.len() as u32;
    for &avail in &[total, total * 3 / 4, total / 2] {
        if avail == 0 { continue; }
        let recovery = PartialRecoveryEngine::new(config.encoding.redundancy_level);
        let can = recovery.can_recover(avail as usize, total);
        let conf = recovery.estimate_confidence(avail as usize, total);
        println!("   可用 {}/{}: 可恢复={}, 置信度={:.2}", avail, total, can, conf);
    }

    println!("\n7. 持久化");
    let dir = std::env::temp_dir().join("holographic_demo");
    let _ = std::fs::remove_dir_all(&dir);
    let persist = PersistenceEngine::new(&dir);
    persist.save_index(&index, "demo.idx").unwrap();
    println!("   保存到: {}", dir.display());
    let loaded = persist.load_index("demo.idx").unwrap();
    println!("   重新加载: {} 片段", loaded.len());

    println!("\n=== 演示完成 ===");
}
