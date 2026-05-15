use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holographic_memory::*;

fn bench_decode_full(c: &mut Criterion) {
    let config = EncodingConfig {
        fft_window_size: 1024,
        overlap_ratio: 0.5,
        redundancy_level: 3,
        phase_modulation: true,
        normalize: true,
    };
    let data: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.005).sin()).collect();
    let mut encoder = FourierEncoder::new(config.clone());
    let encoded = encoder.encode(&data);

    c.bench_function("decode_4096_with_phase", |b| {
        b.iter(|| {
            let mut dec = FourierEncoder::new(config.clone());
            dec.decode(black_box(&encoded.fragments), data.len())
        })
    });
}

fn bench_decode_partial(c: &mut Criterion) {
    let config = EncodingConfig {
        fft_window_size: 512,
        overlap_ratio: 0.5,
        redundancy_level: 3,
        phase_modulation: false,
        normalize: true,
    };
    let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.02).cos()).collect();
    let mut encoder = FourierEncoder::new(config.clone());
    let encoded = encoder.encode(&data);

    let partial: Vec<HologramFragment> = encoded
        .fragments
        .iter()
        .step_by(2)
        .cloned()
        .collect();

    c.bench_function("decode_1024_partial_50pct", |b| {
        b.iter(|| {
            let mut dec = FourierEncoder::new(config.clone());
            dec.decode(black_box(&partial), data.len())
        })
    });
}

criterion_group!(benches, bench_decode_full, bench_decode_partial);
criterion_main!(benches);
