use holographic_memory::*;

fn main() {
    println!("=== 全息记忆存储 - 文本知识库实战 ===\n");

    let config = HolographicConfig {
        encoding: EncodingConfig {
            fft_window_size: 512,
            overlap_ratio: 0.5,
            redundancy_level: 3,
            phase_modulation: true,
            normalize: true,
        },
        retrieval: RetrievalConfig {
            top_k: 5,
            similarity_threshold: 0.0,
            max_association_hops: 2,
            enable_partial_recovery: true,
        },
        ..Default::default()
    };

    let mut hm = HolographicMemory::new(config);

    let knowledge = vec![
        ("全息存储原理", "全息存储基于傅里叶变换将信息映射到频域，每个片段包含整体缩影"),
        ("容错机制", "通过冗余交织编码实现容错，删除50%数据仍可恢复原始信息"),
        ("联想检索", "频域相似度匹配使相似概念自动聚类，支持多跳联想搜索"),
        ("与传统对比", "相比ChromaDB等向量数据库，全息存储在容错性和联想能力上更优"),
        ("应用场景", "适用于知识图谱、语义记忆、容错数据库和联想推理系统"),
    ];

    println!("1. 存储知识库（{} 条）", knowledge.len());
    let mut source_hashes = Vec::new();
    for (title, content) in &knowledge {
        let full_text = format!("{}: {}", title, content);
        let data: Vec<f64> = full_text.as_bytes().iter().map(|&b| b as f64 / 255.0).collect();
        let result = hm.store(&data).unwrap();
        source_hashes.push(result.source_hash);
        println!("   '{}' → {} 片段 (source_hash={})", title, result.fragment_count, result.source_hash);
    }
    println!("   总片段数: {}", hm.fragment_count());

    println!("\n2. 联想检索");
    let queries = vec!["容错", "傅里叶", "知识", "数据库"];
    for query in queries {
        let query_data: Vec<f64> = query.as_bytes().iter().map(|&b| b as f64 / 255.0).collect();
        let results = hm.search(&query_data, 3).unwrap();
        println!("   查询 '{}': {} 个结果", query, results.len());
        for (i, r) in results.iter().enumerate() {
            println!("     #{}: 片段id={}, 相似度={:.4}", i + 1, r.fragment_id, r.similarity);
        }
    }

    println!("\n3. 完整性检查");
    for (i, &hash) in source_hashes.iter().enumerate() {
        let integrity = hm.integrity(hash);
        println!("   源{}: 可用={}/{} 损毁率={:.2}", i, integrity.fragments_available, integrity.fragments_total, integrity.damage_ratio);
    }

    println!("\n4. 容错恢复测试");
    for (title, content) in &knowledge {
        let full_text = format!("{}: {}", title, content);
        let data: Vec<f64> = full_text.as_bytes().iter().map(|&b| b as f64 / 255.0).collect();
        let result = hm.store_with_fault_tolerance(&data, 0.0).unwrap();
        let mse_str = if result.mse < 1e-10 { "无损".to_string() } else { format!("{:.2e}", result.mse) };
        println!("   '{}': 0%损毁 MSE={}", title, mse_str);
    }

    println!("\n5. 统一API便捷性");
    println!("   片段总数: {}", hm.fragment_count());
    println!("   数据源数: {}", hm.source_count());
    println!("   配置: FFT={}, 冗余={}, Top-K={}",
        hm.config().encoding.fft_window_size,
        hm.config().encoding.redundancy_level,
        hm.config().retrieval.top_k,
    );

    println!("\n=== 实战演示完成 ===");
}
