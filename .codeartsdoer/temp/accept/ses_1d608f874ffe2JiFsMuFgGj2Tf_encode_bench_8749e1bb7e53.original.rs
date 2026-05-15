use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holographic_memory::*;

fn bench_fft_encode(c: &mut Criterion) {
    let config = EncodingConfig {
        fft_window_size: 1024,
        overlap_ratio: 0.5,
        redundancy_level: 2,
        phase_modulation: false,
        normalize: true,
    };
    let data: Vec<f64> = (0..2048).map(|i| (i as f64 * 0.01).sin()).collect();

    c.bench_function("fft_encode_2048", |b| {
        b.iter(|| {
            let mut encoder = FourierEncoder::new(config.clone());
            encoder.encode(black_box(&data))
        })
    });
}

fn bench_fft_decode(c: &mut Criterion) {
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

    c.bench_function("fft_decode_2048", |b| {
        b.iter(|| {
            let mut dec = FourierEncoder::new(config.clone());
            dec.decode(black_box(&encoded.fragments), data.len())
        })
    });
}

fn bench_similarity(c: &mut Criterion) {
    let matcher = SimilarityMatcher::new(0.3);
    let query = create_bench_fragment(1, 256);
    let candidates: Vec<HologramFragment> = (0..100)
        .map(|i| create_bench_fragment(i + 2, 256))
        .collect();

    c.bench_function("similarity_top10_100_candidates", |b| {
        b.iter(|| {
            matcher.find_similar(black_box(&query), black_box(&candidates), 10)
        })
    });
}

fn create_bench_fragment(id: FragmentId, size: usize) -> HologramFragment {
    use ndarray::Array2;
    use num_complex::Complex64;
    let freq_data: Vec<Complex64> = (0..size)
        .map(|i| Complex64::new((i as f64 + id as f64).sin(), (i as f64).cos()))
        .collect();
    HologramFragment {
        id,
        frequency_domain: Array2::from_shape_vec((1, size), freq_data).unwrap(),
        phase_key: PhaseKey::zero(size),
        redundancy_level: 2,
        metadata: FragmentMeta::new(0, 1, 0),
    }
}

criterion_group!(benches, bench_fft_encode, bench_fft_decode, bench_similarity);
criterion_main!(benches);
