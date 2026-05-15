use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holographic_memory::*;

fn bench_retrieval(c: &mut Criterion) {
    let mut index = HolographicIndex::new();
    for i in 0..500u64 {
        index.insert(create_retrieval_fragment(i, 128));
    }

    c.bench_function("index_get_500_items", |b| {
        b.iter(|| {
            for i in 0..500u64 {
                black_box(index.get(i + 1));
            }
        })
    });

    let matcher = SimilarityMatcher::new(0.3);
    let query = create_retrieval_fragment(999, 128);
    let candidates: Vec<&HologramFragment> = index.all_fragments();

    c.bench_function("similarity_search_500", |b| {
        b.iter(|| {
            matcher.find_similar(black_box(&query), black_box(&candidates), 10)
        })
    });
}

fn create_retrieval_fragment(id: FragmentId, size: usize) -> HologramFragment {
    use ndarray::Array2;
    use num_complex::Complex64;
    let freq_data: Vec<Complex64> = (0..size)
        .map(|i| Complex64::new((i as f64 * id as f64 * 0.001).sin(), (i as f64).cos()))
        .collect();
    HologramFragment {
        id,
        frequency_domain: Array2::from_shape_vec((1, size), freq_data).unwrap(),
        phase_key: PhaseKey::zero(size),
        redundancy_level: 2,
        metadata: FragmentMeta::new(id, 1, 0),
    }
}

criterion_group!(benches, bench_retrieval);
criterion_main!(benches);
