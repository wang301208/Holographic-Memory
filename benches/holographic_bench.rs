use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use holographic_memory::*;

fn bench_fft_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_encode");
    for size in [256, 512, 1024, 2048] {
        let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.02).sin()).collect();
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            let config = EncodingConfig {
                fft_window_size: 256,
                overlap_ratio: 0.5,
                redundancy_level: 2,
                phase_modulation: false,
                normalize: true,
            };
            let mut encoder = FourierEncoder::new(config);
            b.iter(|| encoder.encode(data));
        });
    }
    group.finish();
}

fn bench_fft_decode(c: &mut Criterion) {
    let config = EncodingConfig {
        fft_window_size: 256,
        overlap_ratio: 0.5,
        redundancy_level: 2,
        phase_modulation: false,
        normalize: true,
    };
    let mut encoder = FourierEncoder::new(config);
    let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.02).sin()).collect();
    let encoded = encoder.encode(&data);

    c.bench_function("fft_decode_1024", |b| {
        b.iter(|| encoder.decode(&encoded.fragments, data.len()));
    });
}

fn bench_similarity_search(c: &mut Criterion) {
    let config = EncodingConfig {
        fft_window_size: 256,
        overlap_ratio: 0.5,
        redundancy_level: 2,
        phase_modulation: false,
        normalize: true,
    };
    let mut encoder = FourierEncoder::new(config);
    let candidates: Vec<HologramFragment> = (0..100)
        .map(|i| {
            let d: Vec<f64> = (0..256).map(|j| ((j as f64 + i as f64) * 0.02).sin()).collect();
            encoder.encode(&d).fragments.into_iter().next().unwrap()
        })
        .collect();

    let query: Vec<f64> = (0..256).map(|i| (i as f64 * 0.02).sin()).collect();
    let query_frag = encoder.encode(&query).fragments.into_iter().next().unwrap();

    c.bench_function("similarity_top10_100candidates", |b| {
        let matcher = SimilarityMatcher::new(0.0);
        b.iter(|| matcher.find_similar(&query_frag, &candidates, 10));
    });
}

fn bench_reed_solomon(c: &mut Criterion) {
    let mut group = c.benchmark_group("reed_solomon");

    group.bench_function("encode_8x4", |b| {
        let rs = ReedSolomon::new(8, 4).unwrap();
        let data: Vec<Vec<u8>> = (0..8).map(|_| vec![42u8; 1024]).collect();
        b.iter(|| rs.encode(&data));
    });

    group.bench_function("reconstruct_8x4_2lost", |b| {
        let rs = ReedSolomon::new(8, 4).unwrap();
        let data: Vec<Vec<u8>> = (0..8).map(|i| vec![(i * 3 + 7) as u8; 1024]).collect();
        let parity = rs.encode(&data).unwrap();
        let all: Vec<Option<Vec<u8>>> = data.iter()
            .chain(parity.iter())
            .map(|s| Some(s.clone()))
            .collect();
        let mut damaged = all.clone();
        damaged[0] = None;
        damaged[8] = None;
        b.iter(|| rs.reconstruct(&damaged));
    });

    group.finish();
}

fn bench_simd_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_ops");
    let va: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();
    let vb: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.02).cos()).collect();

    group.bench_function("dot_product_1024", |bencher| {
        let a = &va;
        let b = &vb;
        bencher.iter(|| SimdOps::dot_product(a, b));
    });

    group.bench_function("cosine_similarity_1024", |bencher| {
        let a = &va;
        let b = &vb;
        bencher.iter(|| SimdOps::cosine_similarity(a, b));
    });

    group.bench_function("hadamard_256", |bencher| {
        let v: Vec<f64> = (0..256).map(|i| (i as f64 * 0.03).sin()).collect();
        bencher.iter(|| SimdOps::hadamard_transform(&v));
    });

    group.finish();
}

fn bench_holographic_memory_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("hm_store");
    for size in [256, 512, 1024] {
        let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.02).sin()).collect();
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
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
            b.iter(|| hm.store(data));
        });
    }
    group.finish();
}

fn bench_redundancy_weave(c: &mut Criterion) {
    let config = EncodingConfig {
        fft_window_size: 256,
        overlap_ratio: 0.5,
        redundancy_level: 2,
        phase_modulation: false,
        normalize: true,
    };
    let mut encoder = FourierEncoder::new(config);
    let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.02).sin()).collect();
    let encoded = encoder.encode(&data);

    c.bench_function("redundancy_weave_r3", |b| {
        let weaver = RedundancyWeaver::new(3);
        b.iter(|| weaver.weave(&encoded.fragments));
    });
}

criterion_group!(
    benches,
    bench_fft_encode,
    bench_fft_decode,
    bench_similarity_search,
    bench_reed_solomon,
    bench_simd_ops,
    bench_holographic_memory_store,
    bench_redundancy_weave,
);

criterion_main!(benches);
