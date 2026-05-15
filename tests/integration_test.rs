#[cfg(test)]
mod tests {
    use holographic_memory::*;

    #[test]
    fn test_config_default() {
        let config = HolographicConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.encoding.fft_window_size, 1024);
        assert_eq!(config.encoding.redundancy_level, 3);
    }

    #[test]
    fn test_config_validation() {
        let mut config = HolographicConfig::default();
        config.encoding.fft_window_size = 100;
        assert!(config.validate().is_err());

        config.encoding.fft_window_size = 1024;
        config.encoding.overlap_ratio = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_phase_key() {
        let pk = PhaseKey::zero(10);
        assert_eq!(pk.dimension, 10);
        assert!(pk.phases.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn test_fragment_meta() {
        let meta = FragmentMeta::new(12345, 5, 2);
        assert_eq!(meta.source_hash, 12345);
        assert_eq!(meta.fragment_count, 5);
        assert_eq!(meta.fragment_index, 2);
        assert!(meta.created_at > 0);
    }

    #[test]
    fn test_integrity_report() {
        let report = IntegrityReport::new(10, 5);
        assert!((report.damage_ratio - 0.5).abs() < 1e-10);
        assert!(report.recovery_possible);
    }

    #[test]
    fn test_integrity_report_beyond_threshold() {
        let report = IntegrityReport::new(10, 4);
        assert!((report.damage_ratio - 0.6).abs() < 1e-10);
        assert!(!report.recovery_possible);
    }

    #[test]
    fn test_holographic_index_insert_get() {
        let mut index = HolographicIndex::new();
        let fragment = create_test_fragment(1, 100);
        index.insert(fragment);
        assert!(index.get(1).is_some());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_holographic_index_remove() {
        let mut index = HolographicIndex::new();
        index.insert(create_test_fragment(1, 100));
        assert!(index.remove(1).is_some());
        assert!(index.is_empty());
    }

    #[test]
    fn test_holographic_index_by_source() {
        let mut index = HolographicIndex::new();
        let mut f1 = create_test_fragment(1, 100);
        f1.metadata.source_hash = 42;
        let mut f2 = create_test_fragment(2, 100);
        f2.metadata.source_hash = 42;
        index.insert(f1);
        index.insert(f2);
        let results = index.get_by_source(42);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_fourier_transform_roundtrip() {
        let mut transformer = FourierTransformer::new();
        let input: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let freq = transformer.forward(&input);
        let time = transformer.inverse(&freq);
        for (i, (expected, actual)) in input.iter().zip(time.iter()).enumerate() {
            assert!(
                (expected - actual.re).abs() < 1e-10,
                "位置 {} 不匹配: {} vs {}",
                i, expected, actual.re
            );
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        use num_complex::Complex64;
        use holographic_memory::foundation::math::cosine_similarity;
        let a: Vec<Complex64> = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10, "相似度应为1.0，实际为 {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        use num_complex::Complex64;
        use holographic_memory::foundation::math::cosine_similarity;
        let a: Vec<Complex64> = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let b: Vec<Complex64> = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "正交向量相似度应为0，实际为 {}", sim);
    }

    #[test]
    fn test_next_power_of_two() {
        use holographic_memory::foundation::math::next_power_of_two;
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(1024), 1024);
        assert_eq!(next_power_of_two(1025), 2048);
    }

    #[test]
    fn test_fourier_encoder_basic() {
        let config = EncodingConfig {
            fft_window_size: 64,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..32).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = encoder.encode(&data);
        assert!(!result.fragments.is_empty());
        assert_ne!(result.source_hash, 0);
    }

    #[test]
    fn test_fourier_encoder_decode_no_phase() {
        let config = EncodingConfig {
            fft_window_size: 64,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..32).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = encoder.encode(&data);
        let decoded = encoder.decode(&result.fragments, data.len());

        let mse: f64 = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        assert!(mse < 0.5, "均方误差过大: {}", mse);
    }

    #[test]
    fn test_fourier_encoder_decode_long_signal() {
        let config = EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..512).map(|i| (i as f64 * 0.02).sin()).collect();
        let result = encoder.encode(&data);
        let decoded = encoder.decode(&result.fragments, data.len());

        let inner_mse: f64 = data[64..448]
            .iter()
            .zip(decoded[64..448].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / 384.0;
        assert!(inner_mse < 0.05, "内部区域均方误差过大: {}", inner_mse);
    }

    #[test]
    fn test_similarity_matcher() {
        let matcher = SimilarityMatcher::new(0.3);
        let f1 = create_test_fragment(1, 16);
        let f2 = create_test_fragment(2, 16);
        let sim = matcher.similarity(&f1, &f2);
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_partial_recovery_engine() {
        let engine = PartialRecoveryEngine::new(3);
        assert!(engine.can_recover(6, 10));
        assert!(!engine.can_recover(3, 10));

        let confidence = engine.estimate_confidence(6, 10);
        assert!((confidence - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_redundancy_weaver() {
        let weaver = RedundancyWeaver::new(2);
        let fragments = vec![
            create_test_fragment(1, 16),
            create_test_fragment(2, 16),
            create_test_fragment(3, 16),
        ];
        let woven = weaver.weave(&fragments);
        assert!(woven.len() > fragments.len());
        let unwoven = weaver.unweave(&woven);
        assert_eq!(unwoven.len(), fragments.len());
    }

    #[test]
    fn test_memory_pool() {
        use holographic_memory::foundation::memory_pool::MemoryPool;
        let mut pool = MemoryPool::new(1024);
        let _ptr1 = pool.allocate(64);
        assert!(pool.total_used() > 0);
    }

    fn create_test_fragment(id: FragmentId, size: usize) -> HologramFragment {
        use ndarray::Array2;
        use num_complex::Complex64;

        let freq_data: Vec<Complex64> = (0..size)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        HologramFragment {
            id,
            frequency_domain: Array2::from_shape_vec((1, size), freq_data).unwrap(),
            phase_key: PhaseKey::zero(size),
            redundancy_level: 2,
            metadata: FragmentMeta::new(0, 1, 0),
        }
    }
}
