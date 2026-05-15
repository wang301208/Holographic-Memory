use holographic_memory::*;

fn main() {
    println!("=== Reed-Solomon 纠删码容错实战 ===\n");

    let data_shards = 4;
    let parity_shards = 2;
    let rs = ReedSolomon::new(data_shards, parity_shards).unwrap();
    println!("RS配置: {} 数据片 + {} 校验片 = {} 总片", data_shards, parity_shards, data_shards + parity_shards);
    println!("最大容错: {} 片丢失仍可恢复\n", parity_shards);

    let original: Vec<Vec<u8>> = (0..data_shards).map(|i| {
        vec![(i * 10 + 1) as u8, (i * 10 + 2) as u8, (i * 10 + 3) as u8, (i * 10 + 4) as u8]
    }).collect();

    println!("原始数据片:");
    for (i, shard) in original.iter().enumerate() {
        println!("  片{}: {:?}", i, shard);
    }

    let parity = rs.encode(&original).unwrap();
    println!("\n校验片:");
    for (i, p) in parity.iter().enumerate() {
        println!("  片{}: {:?}", data_shards + i, p);
    }

    let all_shards: Vec<Option<Vec<u8>>> = original.iter()
        .chain(parity.iter())
        .map(|s| Some(s.clone()))
        .collect();

    println!("\n--- 模拟数据损毁 ---");

    let damage_scenarios: Vec<Vec<usize>> = vec![
        vec![],
        vec![5],
        vec![4],
        vec![4, 5],
        vec![3, 5],
        vec![0, 4],
    ];

    for lost in &damage_scenarios {
        let mut shards: Vec<Option<Vec<u8>>> = all_shards.iter()
            .map(|s| s.clone())
            .collect();
        for &idx in lost {
            shards[idx] = None;
        }

        let lost_str = if lost.is_empty() { "无".to_string() } else {
            lost.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
        };

        print!("  丢失片[{}]: ", lost_str);

        match rs.reconstruct(&shards) {
            Ok(recovered) => {
                let data_ok = recovered[..data_shards] == original[..];
                if data_ok {
                    println!("恢复成功，数据完整");
                } else {
                    println!("恢复完成，但数据不一致!");
                }
            }
            Err(e) => {
                println!("恢复失败: {}", e);
            }
        }
    }

    println!("\n--- 超出容错能力 ---");
    let mut shards: Vec<Option<Vec<u8>>> = all_shards.iter().map(|s| s.clone()).collect();
    shards[0] = None;
    shards[4] = None;
    shards[5] = None;
    print!("  丢失片[0,4,5] ({}片 > {}容错): ", 3, parity_shards);
    match rs.reconstruct(&shards) {
        Ok(_) => println!("意外成功"),
        Err(e) => println!("预期失败: {}", e),
    }

    println!("\n--- 与全息记忆集成 ---");
    let config = HolographicConfig {
        encoding: EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 3,
            phase_modulation: false,
            normalize: true,
        },
        ..Default::default()
    };

    let mut hm = HolographicMemory::new(config)
        .with_reed_solomon(4, 2)
        .unwrap();

    let signal: Vec<f64> = (0..512).map(|i| {
        let t = i as f64 * 0.02;
        t.sin() + 0.5 * (t * 3.0).cos()
    }).collect();

    let result = hm.store_with_rs(&signal).unwrap();
    println!("存储: {} 片段 (RS纠删码保护)", result.total_fragments);
    println!("\n=== 完成 ===");
}
