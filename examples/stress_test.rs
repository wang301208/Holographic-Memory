use holographic_memory::*;
use std::time::Instant;

fn main() {
    println!("=== 全息记忆存储 - 大规模压力测试 ===\n");

    let config = HolographicConfig {
        encoding: EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        },
        ..Default::default()
    };

    let mut hm = HolographicMemory::new(config);

    let num_sources = 1000;
    let data_len = 256;

    println!("--- 存储压力测试 ({}) 条数据 ---", num_sources);
    let mut hashes = Vec::with_capacity(num_sources);
    let t0 = Instant::now();
    for i in 0..num_sources {
        let data: Vec<f64> = (0..data_len).map(|j| ((j as f64 + i as f64) * 0.02).sin()).collect();
        let result = hm.store(&data).unwrap();
        hashes.push(result.source_hash);
    }
    let store_time = t0.elapsed();
    println!("  存储 {} 条: {:?} ({:.0} 条/秒)", num_sources, store_time, num_sources as f64 / store_time.as_secs_f64());
    println!("  总片段数: {}", hm.fragment_count());

    println!("\n--- 检索压力测试 ---");
    let query_count = 500;
    let t1 = Instant::now();
    for i in 0..query_count {
        let hash = hashes[i % hashes.len()];
        let _ = hm.retrieve(hash, data_len);
    }
    let retrieve_time = t1.elapsed();
    println!("  检索 {} 次: {:?} ({:.0} 次/秒)", query_count, retrieve_time, query_count as f64 / retrieve_time.as_secs_f64());

    println!("\n--- 容错恢复压力测试 ---");
    let ft_count = 100;
    let t2 = Instant::now();
    for i in 0..ft_count {
        let data: Vec<f64> = (0..data_len).map(|j| ((j as f64 + i as f64) * 0.03).cos()).collect();
        let _ = hm.store_with_fault_tolerance(&data, 0.3);
    }
    let ft_time = t2.elapsed();
    println!("  容错测试 {} 次(30%损毁): {:?} ({:.0} 次/秒)", ft_count, ft_time, ft_count as f64 / ft_time.as_secs_f64());

    println!("\n--- 索引操作压力测试 ---");
    let mut idx = HolographicIndex::new();
    use ndarray::Array2;
    use num_complex::Complex64;

    let idx_count = 10_000;
    let t3 = Instant::now();
    for i in 0..idx_count {
        let frag = HologramFragment {
            id: i as u64 + 100_000,
            frequency_domain: Array2::from_shape_vec((1, 4), vec![Complex64::new(i as f64, 0.0); 4]).unwrap(),
            phase_key: PhaseKey::zero(4),
            redundancy_level: 1,
            metadata: FragmentMeta::new(9999, 1, 0),
        };
        idx.insert(frag);
    }
    let insert_time = t3.elapsed();
    println!("  索引插入 {} 条: {:?} ({:.0} 条/秒)", idx_count, insert_time, idx_count as f64 / insert_time.as_secs_f64());

    let t4 = Instant::now();
    for i in 0..idx_count {
        let _ = idx.get(i as u64 + 100_000);
    }
    let get_time = t4.elapsed();
    println!("  索引查询 {} 次: {:?} ({:.0} 次/秒)", idx_count, get_time, idx_count as f64 / get_time.as_secs_f64());

    println!("\n--- RS 纠删码压力测试 ---");
    let rs = ReedSolomon::new(8, 4).unwrap();
    let rs_count = 1000;
    let rs_data: Vec<Vec<u8>> = (0..8).map(|_| vec![42u8; 4096]).collect();
    let t5 = Instant::now();
    for _ in 0..rs_count {
        let _ = rs.encode(&rs_data);
    }
    let rs_time = t5.elapsed();
    println!("  RS编码 {} 次(8x4, 4KB/片): {:?} ({:.0} 次/秒)", rs_count, rs_time, rs_count as f64 / rs_time.as_secs_f64());

    println!("\n=== 压力测试完成 ===");
}
