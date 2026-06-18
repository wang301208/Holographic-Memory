#[cfg(test)]
mod tests {
    use holographic_memory::*;

    #[test]
    fn test_hm_store() {
        let config = HolographicConfig::default();
        let mut hm = HolographicMemory::new(config);
        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.02).sin()).collect();
        let result = hm.store(&data).unwrap();
        assert!(result.fragment_count > 0);
        assert!(hm.fragment_count() > 0);
    }

    #[test]
    fn test_hm_retrieve() {
        let config = make_small_config();
        let mut hm = HolographicMemory::new(config);
        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.02).sin()).collect();
        let store = hm.store(&data).unwrap();
        let decoded = hm.retrieve(store.source_hash, data.len()).unwrap();
        assert_eq!(decoded.len(), data.len());
    }

    #[test]
    fn test_hm_search() {
        let config = make_small_config();
        let mut hm = HolographicMemory::new(config);
        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.02).sin()).collect();
        hm.store(&data).unwrap();
        let results = hm.search(&data, 5).unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_hm_integrity() {
        let config = HolographicConfig::default();
        let mut hm = HolographicMemory::new(config);
        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.02).sin()).collect();
        let store = hm.store(&data).unwrap();
        let integrity = hm.integrity(store.source_hash);
        assert!(integrity.fragments_available > 0);
    }

    #[test]
    fn test_hm_persistence() {
        let dir = std::env::temp_dir().join("hm_api_test");
        let _ = std::fs::remove_dir_all(&dir);
        let config = HolographicConfig::default();
        let mut hm = HolographicMemory::new(config).with_persistence(&dir);
        let data: Vec<f64> = (0..128).map(|i| (i as f64 * 0.05).sin()).collect();
        hm.store(&data).unwrap();
        hm.save().unwrap();

        let mut hm2 = HolographicMemory::new(HolographicConfig::default()).with_persistence(&dir);
        hm2.load().unwrap();
        assert!(hm2.fragment_count() > 0);
    }

    #[test]
    fn test_hm_fault_tolerance() {
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
        let mut hm = HolographicMemory::new(config);
        let data: Vec<f64> = (0..512).map(|i| (i as f64 * 0.02).sin()).collect();
        let result = hm.store_with_fault_tolerance(&data, 0.3).unwrap();
        assert!(result.total_fragments > 0);
        assert!(result.integrity.damage_ratio <= 1.0);
    }

    #[test]
    fn test_hm_fault_tolerance_50pct_recovers_signal() {
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
        let mut hm = HolographicMemory::new(config);
        let data: Vec<f64> = (0..1024)
            .map(|i| {
                let t = i as f64 * 0.02;
                t.sin() + 0.5 * (t * 3.0).cos()
            })
            .collect();

        let result = hm.store_with_fault_tolerance(&data, 0.5).unwrap();

        assert!(result.integrity.recovery_possible);
        assert_eq!(result.damage_ratio, 0.5);
        assert!(result.mse < 0.2, "50% 损毁恢复 MSE 过大: {}", result.mse);
    }

    #[test]
    fn test_hm_compression_report_reaches_03x() {
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
        let mut hm = HolographicMemory::new(config);
        let data: Vec<f64> = (0..1024)
            .map(|i| {
                let t = i as f64 * 0.02;
                t.sin() + 0.5 * (t * 3.0).cos()
            })
            .collect();

        let report = hm.compression_report(&data, 0.3).unwrap();

        assert!(report.fragment_count > 0);
        assert!(report.dense_bytes > report.sparse_bytes);
        assert!(
            report.compression_ratio < 0.3,
            "压缩比未达到 0.3x: {:?}",
            report
        );
        assert!(report.retained_energy_ratio > 0.2);
    }

    #[test]
    fn test_hm_associate_returns_related_fragments() {
        let config = HolographicConfig {
            encoding: EncodingConfig {
                fft_window_size: 256,
                overlap_ratio: 0.5,
                redundancy_level: 3,
                phase_modulation: false,
                normalize: true,
            },
            retrieval: RetrievalConfig {
                top_k: 5,
                similarity_threshold: 0.0,
                max_association_hops: 2,
                enable_partial_recovery: true,
            },
            ..Default::default()
        };
        let mut hm = HolographicMemory::new(config);

        let base: Vec<f64> = (0..512)
            .map(|i| {
                let t = i as f64 * 0.02;
                (t * 2.0).sin() + 0.3 * (t * 5.0).cos()
            })
            .collect();
        let related: Vec<f64> = (0..512)
            .map(|i| {
                let t = i as f64 * 0.02;
                (t * 2.0).sin() + 0.3 * (t * 5.0).cos() + 0.02 * (t * 11.0).sin()
            })
            .collect();
        let unrelated: Vec<f64> = (0..512)
            .map(|i| (i as f64 * 0.11).cos())
            .collect();

        hm.store(&base).unwrap();
        hm.store(&related).unwrap();
        hm.store(&unrelated).unwrap();

        let results = hm.associate(&base, 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|item| item.similarity > 0.5));
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_holo_error() {
        let err = HoloError::Encode("测试".to_string());
        assert!(err.to_string().contains("测试"));
    }

    #[test]
    fn test_phase_key_thread_safe() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        let keys: Arc<Mutex<Vec<PhaseKey>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let keys = Arc::clone(&keys);
            handles.push(thread::spawn(move || {
                let pk = PhaseKey::random(10);
                keys.lock().unwrap().push(pk);
            }));
        }
        for h in handles { h.join().unwrap(); }
        assert_eq!(keys.lock().unwrap().len(), 4);
    }

    #[test]
    fn test_config_from_str() {
        let toml = r#"
[encoding]
fft_window_size = 512
overlap_ratio = 0.25
redundancy_level = 2
phase_modulation = false
normalize = true
[storage]
data_dir = "./test_data"
max_segment_size = 33554432
auto_compact = true
sync_on_write = false
[retrieval]
top_k = 5
similarity_threshold = 0.5
max_association_hops = 2
enable_partial_recovery = false
"#;
        let config = HolographicConfig::load_from_str(toml).unwrap();
        assert_eq!(config.encoding.fft_window_size, 512);
        assert_eq!(config.retrieval.top_k, 5);
    }

    fn make_small_config() -> HolographicConfig {
        HolographicConfig {
            encoding: EncodingConfig {
                fft_window_size: 256,
                overlap_ratio: 0.5,
                redundancy_level: 2,
                phase_modulation: false,
                normalize: true,
            },
            ..Default::default()
        }
    }
}
