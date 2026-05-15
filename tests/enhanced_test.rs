#[cfg(test)]
mod tests {
    use holographic_memory::*;
    use ndarray::Array2;
    use num_complex::Complex64;

    fn create_test_fragment(id: FragmentId, size: usize, source: u64, idx: u32, count: u32) -> HologramFragment {
        let freq_data: Vec<Complex64> = (0..size)
            .map(|i| Complex64::new((i as f64 + id as f64).sin(), (i as f64).cos()))
            .collect();
        HologramFragment {
            id,
            frequency_domain: Array2::from_shape_vec((1, size), freq_data).unwrap(),
            phase_key: PhaseKey::zero(size),
            redundancy_level: 3,
            metadata: FragmentMeta::new(source, count, idx),
        }
    }

    #[test]
    fn test_parity_recovery_single_missing() {
        let weaver = RedundancyWeaver::new(3);
        let n = 6;
        let source = 12345;
        let fragments: Vec<HologramFragment> = (0..n)
            .map(|i| create_test_fragment(i as u64 + 1, 32, source, i as u32, n as u32))
            .collect();

        let woven = weaver.weave(&fragments);
        let parity_count = woven.iter().filter(|f| f.metadata.tags.iter().any(|t| t.starts_with("parity_L"))).count();
        assert!(parity_count >= 2, "应至少有2个校验片段");

        let mut available: Vec<HologramFragment> = fragments.clone();
        available.remove(2);

        let recovery = weaver.recover_from_partial(&available, n as u32);
        assert!(recovery.confidence >= 0.8, "单片段丢失恢复置信度应≥0.8: {}", recovery.confidence);
    }

    #[test]
    fn test_parity_recovery_two_missing() {
        let weaver = RedundancyWeaver::new(3);
        let n = 8;
        let source = 54321;
        let fragments: Vec<HologramFragment> = (0..n)
            .map(|i| create_test_fragment(i as u64 + 100, 32, source, i as u32, n as u32))
            .collect();

        let woven = weaver.weave(&fragments);
        let parity_fragments: Vec<&HologramFragment> = woven.iter()
            .filter(|f| f.metadata.tags.iter().any(|t| t.starts_with("parity_L")))
            .collect();

        let mut available = fragments.clone();
        available.remove(3);
        if available.len() > 4 {
            available.remove(4);
        }

        if parity_fragments.len() >= 2 {
            for &pf in &parity_fragments {
                available.push((*pf).clone());
            }
        }

        let recovery = weaver.recover_from_partial(&available, n as u32);
        assert!(recovery.confidence > 0.0, "多片段丢失应有部分恢复");
    }

    #[test]
    fn test_50pct_damage_confidence() {
        let weaver = RedundancyWeaver::new(3);
        let n = 10;
        let source = 99999;
        let fragments: Vec<HologramFragment> = (0..n)
            .map(|i| create_test_fragment(i as u64 + 200, 32, source, i as u32, n as u32))
            .collect();

        let woven = weaver.weave(&fragments);

        let mut available: Vec<HologramFragment> = fragments.clone();
        let half = n / 2;
        available.truncate(half);

        let parity_fragments: Vec<HologramFragment> = woven.iter()
            .filter(|f| f.metadata.tags.iter().any(|t| t.starts_with("parity_L")))
            .cloned()
            .collect();
        available.extend(parity_fragments);

        let recovery = weaver.recover_from_partial(&available, n as u32);
        assert!(recovery.confidence >= 0.5, "50%损毁+校验恢复置信度应≥0.5: {}", recovery.confidence);
    }

    #[test]
    fn test_large_scale_encode_decode() {
        let config = EncodingConfig {
            fft_window_size: 1024,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..8192).map(|i| (i as f64 * 0.003).sin()).collect();
        let result = encoder.encode(&data);
        assert!(result.fragments.len() > 4);

        let decoded = encoder.decode(&result.fragments, data.len());
        let inner_start = 512;
        let inner_end = data.len() - 512;
        let mse: f64 = data[inner_start..inner_end]
            .iter()
            .zip(decoded[inner_start..inner_end].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (inner_end - inner_start) as f64;
        assert!(mse < 0.01, "大规模信号MSE: {}", mse);
    }

    #[test]
    fn test_large_scale_index() {
        let mut index = HolographicIndex::new();
        let n = 1000;
        for i in 0..n {
            let frag = create_test_fragment(i as u64 + 1, 16, 77777, i as u32, n as u32);
            index.insert(frag);
        }
        assert_eq!(index.len(), n as usize);

        let integrity = index.integrity_check(77777);
        assert_eq!(integrity.fragments_available, n as u32);
        assert!((integrity.damage_ratio - 0.0).abs() < 1e-10);

        let by_source = index.get_by_source(77777);
        assert_eq!(by_source.len(), n as usize);
    }

    #[test]
    fn test_stress_encode_multiple_sources() {
        let config = EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let mut index = HolographicIndex::new();

        for source_id in 0..10u64 {
            let data: Vec<f64> = (0..512)
                .map(|i| (i as f64 * 0.01 + source_id as f64).sin())
                .collect();
            let result = encoder.encode(&data);
            for frag in result.fragments {
                index.insert(frag);
            }
        }

        assert!(index.len() > 10);
        assert_eq!(index.all_source_hashes().len(), 10);
    }

    #[test]
    fn test_fault_tolerance_sweep() {
        let config = EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 3,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin() + 0.5 * (i as f64 * 0.05).cos()).collect();
        let result = encoder.encode(&data);
        let total = result.fragments.len();

        let weaver = RedundancyWeaver::new(3);
        let _woven = weaver.weave(&result.fragments);

        for &damage_pct in &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
            let remove_count = (total as f64 * damage_pct) as usize;
            let available: Vec<HologramFragment> = result.fragments.iter()
                .skip(remove_count)
                .cloned()
                .collect();

            let decoded = encoder.decode(&available, data.len());

            let integrity = IntegrityReport::new(total as u32, available.len() as u32);
            assert!(integrity.damage_ratio <= 1.0);
            assert!(decoded.len() == data.len());
        }
    }

    #[test]
    fn test_parallel_vs_sequential_encode() {
        let data: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.005).sin()).collect();

        let config = EncodingConfig {
            fft_window_size: 512,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };

        let mut seq_encoder = FourierEncoder::new(config.clone());
        let seq_result = seq_encoder.encode(&data);

        let par_encoder = ParallelEncoder::new(config);
        let par_result = par_encoder.encode(&data);

        assert_eq!(seq_result.fragments.len(), par_result.fragments.len());
    }
}
