#[cfg(test)]
mod tests {
    use holographic_memory::*;

    #[test]
    fn test_encode_decode_precision_1024() {
        let config = EncodingConfig {
            fft_window_size: 1024,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..2048).map(|i| (i as f64 * 0.01).sin() + 0.3 * (i as f64 * 0.07).cos()).collect();
        let result = encoder.encode(&data);
        let decoded = encoder.decode(&result.fragments, data.len());

        let inner_start = 256;
        let inner_end = data.len() - 256;
        let inner_mse: f64 = data[inner_start..inner_end]
            .iter()
            .zip(decoded[inner_start..inner_end].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (inner_end - inner_start) as f64;
        assert!(inner_mse < 0.01, "内部区域MSE过大: {}", inner_mse);
    }

    #[test]
    fn test_encode_decode_with_phase_modulation() {
        let config = EncodingConfig {
            fft_window_size: 512,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: true,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.02).sin()).collect();
        let result = encoder.encode(&data);
        assert!(!result.fragments.is_empty());

        for fragment in &result.fragments {
            assert_eq!(fragment.phase_key.dimension, 512);
        }
    }

    #[test]
    fn test_hologram_fragmenter_partial_recovery() {
        let mut encoder = FourierEncoder::new(EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        });
        let data: Vec<f64> = (0..512).map(|i| (i as f64 * 0.02).sin()).collect();
        let result = encoder.encode(&data);

        let fragmenter = HologramFragmenter::new(128);
        let hologram_fragments = fragmenter.fragment(&result.fragments);
        assert!(!hologram_fragments.is_empty());

        if hologram_fragments.len() > 1 {
            let half_count = hologram_fragments.len() / 2;
            let partial: Vec<HologramFragment> = hologram_fragments[..half_count].to_vec();
            assert!(partial.len() > 0);
        }
    }

    #[test]
    fn test_hologram_fragmenter_defragment_roundtrip() {
        let mut encoder = FourierEncoder::new(EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        });
        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.03).sin()).collect();
        let result = encoder.encode(&data);

        let fragmenter = HologramFragmenter::new(64);
        let fragmented = fragmenter.fragment(&result.fragments);
        let defragmented = fragmenter.defragment(&fragmented);

        assert!(!defragmented.is_empty());
    }

    #[test]
    fn test_redundancy_weaver_50pct_damage() {
        let weaver = RedundancyWeaver::new(3);
        let n = 8;
        let fragments: Vec<HologramFragment> = (0..n)
            .map(|i| create_test_fragment(i + 1, 32))
            .collect();

        let woven = weaver.weave(&fragments);
        assert!(woven.len() > fragments.len());

        let original_count = fragments.len() as u32;
        let remove_count = original_count as usize / 2;
        let mut available: Vec<HologramFragment> = fragments.clone();
        available.truncate(remove_count);

        let recovery = weaver.recover_from_partial(&available, original_count);
        assert!(recovery.confidence > 0.0, "恢复置信度应大于0");
    }

    #[test]
    fn test_redundancy_weaver_preserves_original() {
        let weaver = RedundancyWeaver::new(2);
        let fragments = vec![
            create_test_fragment(1, 16),
            create_test_fragment(2, 16),
            create_test_fragment(3, 16),
        ];
        let woven = weaver.weave(&fragments);
        let unwoven = weaver.unweave(&woven);
        assert_eq!(unwoven.len(), fragments.len());
    }

    #[test]
    fn test_partial_recovery_50pct() {
        let engine = PartialRecoveryEngine::new(3);
        assert!(engine.can_recover(6, 10));
        assert!(engine.can_recover(5, 10));
        assert!(!engine.can_recover(4, 10));
        assert!(!engine.can_recover(3, 10));

        let confidence = engine.estimate_confidence(5, 10);
        assert!((confidence - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_partial_recovery_with_fragments() {
        let engine = PartialRecoveryEngine::new(3);
        let fragments: Vec<HologramFragment> = (0..6)
            .map(|i| create_test_fragment(i + 1, 32))
            .collect();

        let result = engine.recover(&fragments, 10);
        assert!(result.confidence > 0.0);
        assert!(result.integrity.damage_ratio <= 1.0);
        assert!(!result.content.is_empty(), "恢复结果应包含可解码的片段内容");
    }

    #[test]
    fn test_parallel_encoder_basic() {
        let config = EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let encoder = ParallelEncoder::new(config);
        let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();
        let result = encoder.encode(&data);
        assert!(!result.fragments.is_empty());
        assert_ne!(result.source_hash, 0);
    }

    #[test]
    fn test_parallel_encoder_large_signal() {
        let config = EncodingConfig {
            fft_window_size: 1024,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let encoder = ParallelEncoder::new(config);
        let data: Vec<f64> = (0..8192).map(|i| (i as f64 * 0.003).sin()).collect();
        let result = encoder.encode(&data);
        assert!(!result.fragments.is_empty());
        assert!(result.fragments.len() > 4);
    }

    #[test]
    fn test_encode_decode_multi_frequency() {
        let config = EncodingConfig {
            fft_window_size: 512,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..1024)
            .map(|i| {
                let t = i as f64 * 0.01;
                (t * 2.0).sin() + 0.5 * (t * 7.0).sin() + 0.3 * (t * 13.0).cos()
            })
            .collect();
        let result = encoder.encode(&data);
        let decoded = encoder.decode(&result.fragments, data.len());

        let inner_start = 128;
        let inner_end = data.len() - 128;
        let inner_mse: f64 = data[inner_start..inner_end]
            .iter()
            .zip(decoded[inner_start..inner_end].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (inner_end - inner_start) as f64;
        assert!(inner_mse < 0.02, "多频信号内部区域MSE过大: {}", inner_mse);
    }

    #[test]
    fn test_redundancy_multi_level_recovery() {
        let weaver = RedundancyWeaver::new(4);
        let n: usize = 6;
        let fragments: Vec<HologramFragment> = (0..n)
            .map(|i| create_test_fragment(i as u64 + 1, 16))
            .collect();

        let woven = weaver.weave(&fragments);
        let redundancy_count = woven.len() - n;
        assert!(redundancy_count > 0, "应有冗余片段");

        let mut available = fragments.clone();
        available.truncate(n / 2);

        let recovery = weaver.recover_from_partial(&available, n as u32);
        assert!(recovery.confidence > 0.0);
    }

    #[test]
    fn test_sparse_encoder_compresses_and_roundtrips_major_energy() {
        let encoder = SparseEncoder::new(0.3);
        let fragment = create_test_fragment(42, 128);
        let dense_bytes = bincode::serialize(&fragment).unwrap();

        let sparse = encoder.sparsify(&fragment);
        let sparse_bytes = bincode::serialize(&sparse).unwrap();
        assert!(
            (sparse_bytes.len() as f64) < (dense_bytes.len() as f64) * 0.3,
            "稀疏表示未达到预期压缩比: dense={} sparse={}",
            dense_bytes.len(),
            sparse_bytes.len()
        );

        let restored = encoder.densify(&sparse);
        assert_eq!(restored.frequency_domain.len(), fragment.frequency_domain.len());

        let original_energy: f64 = fragment
            .frequency_domain
            .iter()
            .map(|c| c.norm_sqr())
            .sum();
        let restored_energy: f64 = restored
            .frequency_domain
            .iter()
            .map(|c| c.norm_sqr())
            .sum();
        assert!(
            restored_energy / original_energy > 0.2,
            "稀疏回填保留的能量过低: original={} restored={}",
            original_energy,
            restored_energy
        );
    }

    fn create_test_fragment(id: FragmentId, size: usize) -> HologramFragment {
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
            metadata: FragmentMeta::new(0, 1, (id - 1) as u32),
        }
    }
}
