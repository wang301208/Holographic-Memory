use std::time::Instant;

fn main() {
    println!("=== 全息记忆存储 - 性能基准测试 ===\n");

    bench_fft_encode();
    bench_fft_decode();
    bench_index_operations();
    bench_similarity_search();
    bench_parallel_encode();
    bench_fault_tolerance();
    bench_persistence();

    println!("\n=== 基准测试完成 ===");
}

fn bench_fft_encode() {
    use holographic_memory::*;

    let config = EncodingConfig {
        fft_window_size: 1024,
        overlap_ratio: 0.5,
        redundancy_level: 3,
        phase_modulation: false,
        normalize: true,
    };

    for &size in &[512, 2048, 8192] {
        let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.01).sin()).collect();
        let iters = if size <= 512 { 100 } else { 20 };

        let start = Instant::now();
        for _ in 0..iters {
            let mut encoder = FourierEncoder::new(config.clone());
            let _ = encoder.encode(&data);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / iters;

        println!("FFT编码  {}点: {:?}/op ({}次迭代)", size, per_op, iters);
    }
}

fn bench_fft_decode() {
    use holographic_memory::*;

    let config = EncodingConfig {
        fft_window_size: 1024,
        overlap_ratio: 0.5,
        redundancy_level: 2,
        phase_modulation: false,
        normalize: true,
    };

    let data: Vec<f64> = (0..2048).map(|i| (i as f64 * 0.01).sin()).collect();
    let mut encoder = FourierEncoder::new(config.clone());
    let encoded = encoder.encode(&data);

    let start = Instant::now();
    for _ in 0..50 {
        let mut dec = FourierEncoder::new(config.clone());
        let _ = dec.decode(&encoded.fragments, data.len());
    }
    let per_op = start.elapsed() / 50;
    println!("FFT解码 2048点: {:?}/op", per_op);
}

fn bench_index_operations() {
    use holographic_memory::*;
    use ndarray::Array2;
    use num_complex::Complex64;

    let mut index = HolographicIndex::new();

    let start = Instant::now();
    for i in 0..10000u64 {
        let freq: Vec<Complex64> = (0..64).map(|j| Complex64::new((j as f64 + i as f64).sin(), 0.0)).collect();
        let frag = HologramFragment {
            id: i,
            frequency_domain: Array2::from_shape_vec((1, 64), freq).unwrap(),
            phase_key: PhaseKey::zero(64),
            redundancy_level: 2,
            metadata: FragmentMeta::new(1, 10000, i as u32),
        };
        index.insert(frag);
    }
    let insert_time = start.elapsed();
    println!("索引插入 10000片段: {:?}", insert_time);

    let start = Instant::now();
    for i in 0..1000u64 {
        let _ = index.get(i);
    }
    let lookup_time = start.elapsed() / 1000;
    println!("索引查找: {:?}/op", lookup_time);
}

fn bench_similarity_search() {
    use holographic_memory::*;
    use ndarray::Array2;
    use num_complex::Complex64;

    let n = 500;
    let fragments: Vec<HologramFragment> = (0..n)
        .map(|i| {
            let freq: Vec<Complex64> = (0..128).map(|j| Complex64::new((j as f64 * i as f64 * 0.001).sin(), (j as f64).cos())).collect();
            HologramFragment {
                id: i,
                frequency_domain: Array2::from_shape_vec((1, 128), freq).unwrap(),
                phase_key: PhaseKey::zero(128),
                redundancy_level: 2,
                metadata: FragmentMeta::new(1, n as u32, i as u32),
            }
        })
        .collect();

    let matcher = SimilarityMatcher::new(0.0);
    let start = Instant::now();
    for _ in 0..10 {
        let _ = matcher.find_similar(&fragments[0], &fragments[1..], 10);
    }
    let per_op = start.elapsed() / 10;
    println!("相似度搜索 500片段: {:?}/op", per_op);
}

fn bench_parallel_encode() {
    use holographic_memory::*;

    let config = EncodingConfig {
        fft_window_size: 1024,
        overlap_ratio: 0.5,
        redundancy_level: 2,
        phase_modulation: false,
        normalize: true,
    };

    let data: Vec<f64> = (0..8192).map(|i| (i as f64 * 0.003).sin()).collect();

    let par_encoder = ParallelEncoder::new(config.clone());
    let start = Instant::now();
    for _ in 0..20 {
        let enc = ParallelEncoder::new(config.clone());
        let _ = enc.encode(&data);
    }
    let par_time = start.elapsed() / 20;

    let start = Instant::now();
    for _ in 0..20 {
        let mut enc = FourierEncoder::new(config.clone());
        let _ = enc.encode(&data);
    }
    let seq_time = start.elapsed() / 20;

    println!("并行编码 8192点: {:?}/op (串行: {:?})", par_time, seq_time);
}

fn bench_fault_tolerance() {
    use holographic_memory::*;

    let config = EncodingConfig {
        fft_window_size: 256,
        overlap_ratio: 0.5,
        redundancy_level: 3,
        phase_modulation: false,
        normalize: true,
    };

    let data: Vec<f64> = (0..2048).map(|i| (i as f64 * 0.01).sin()).collect();
    let mut encoder = FourierEncoder::new(config);
    let result = encoder.encode(&data);

    let weaver = RedundancyWeaver::new(3);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = weaver.weave(&result.fragments);
    }
    let weave_time = start.elapsed() / 100;

    let woven = weaver.weave(&result.fragments);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = weaver.unweave(&woven);
    }
    let unweave_time = start.elapsed() / 100;

    println!("冗余交织: {:?}/op, 解织: {:?}/op", weave_time, unweave_time);
}

fn bench_persistence() {
    use holographic_memory::*;

    let dir = std::env::temp_dir().join("holographic_bench");
    let _ = std::fs::remove_dir_all(&dir);

    let mut index = HolographicIndex::new();
    use ndarray::Array2;
    use num_complex::Complex64;
    for i in 0..100u64 {
        let freq: Vec<Complex64> = (0..64).map(|j| Complex64::new((j as f64 + i as f64).sin(), 0.0)).collect();
        let frag = HologramFragment {
            id: i,
            frequency_domain: Array2::from_shape_vec((1, 64), freq).unwrap(),
            phase_key: PhaseKey::zero(64),
            redundancy_level: 2,
            metadata: FragmentMeta::new(1, 100, i as u32),
        };
        index.insert(frag);
    }

    let engine = PersistenceEngine::new(&dir);
    let start = Instant::now();
    for _ in 0..50 {
        let _ = engine.save_index(&index, "bench.idx");
    }
    let save_time = start.elapsed() / 50;

    let start = Instant::now();
    for _ in 0..50 {
        let _ = engine.load_index("bench.idx");
    }
    let load_time = start.elapsed() / 50;

    println!("持久化保存 100片段: {:?}/op", save_time);
    println!("持久化加载 100片段: {:?}/op", load_time);
}
