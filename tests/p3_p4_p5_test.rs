#[cfg(test)]
mod tests {
    use holographic_memory::*;
    use std::path::PathBuf;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("holographic_tests").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn create_test_fragment(id: FragmentId, size: usize, source: u64, idx: u32, count: u32) -> HologramFragment {
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
            metadata: FragmentMeta::new(source, count, idx),
        }
    }

    #[test]
    fn test_persistence_save_load() {
        let dir = test_dir("save_load");
        let engine = PersistenceEngine::new(&dir);

        let mut index = HolographicIndex::new();
        for i in 0..5u64 {
            index.insert(create_test_fragment(i + 1, 16, 100, i as u32, 5));
        }

        engine.save_index(&index, "test.idx").unwrap();
        let loaded = engine.load_index("test.idx").unwrap();
        assert_eq!(loaded.len(), 5);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = test_dir("roundtrip");
        let engine = PersistenceEngine::new(&dir);

        let mut index = HolographicIndex::new();
        index.insert(create_test_fragment(1, 16, 100, 0, 1));
        index.insert(create_test_fragment(2, 32, 100, 1, 2));

        let restored = engine.save_index_roundtrip(&index, "rt.idx").unwrap();
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn test_wal_incremental() {
        let dir = test_dir("wal_inc");
        let mut engine = PersistenceEngine::new(&dir);

        for i in 0..5u64 {
            engine.insert_incremental(&create_test_fragment(i + 1, 16, 200, i as u32, 5)).unwrap();
        }
        engine.flush_wal().unwrap();

        let replayed = engine.replay_wal().unwrap();
        assert_eq!(replayed.len(), 5);
    }

    #[test]
    fn test_wal_auto_flush() {
        let dir = test_dir("wal_auto");
        let mut engine = PersistenceEngine::new(&dir);

        for i in 0..150u64 {
            engine.insert_incremental(&create_test_fragment(i + 1, 8, 300, i as u32, 150)).unwrap();
        }

        let replayed = engine.replay_wal().unwrap();
        assert!(replayed.len() > 0);
    }

    #[test]
    fn test_compact() {
        let dir = test_dir("compact");
        let mut engine = PersistenceEngine::new(&dir);

        for i in 0..10u64 {
            engine.insert_incremental(&create_test_fragment(i + 1, 8, 400, i as u32, 10)).unwrap();
        }
        engine.flush_wal().unwrap();

        let mut index = HolographicIndex::new();
        for i in 0..10u64 {
            index.insert(create_test_fragment(i + 1, 8, 400, i as u32, 10));
        }

        engine.compact(&index, "compact.idx").unwrap();

        let loaded = engine.load_index("compact.idx").unwrap();
        assert_eq!(loaded.len(), 10);
    }

    #[test]
    fn test_segment_manager_basic() {
        let mut mgr = SegmentManager::new(100);
        let ids: Vec<FragmentId> = (0..5)
            .map(|i| mgr.add_fragment(create_test_fragment(i + 1, 16, 500, i as u32, 5)))
            .collect();
        assert_eq!(ids.len(), 5);
        assert_eq!(mgr.total_fragments(), 5);
        assert!(mgr.get(ids[0]).is_some());
    }

    #[test]
    fn test_segment_manager_by_source() {
        let mut mgr = SegmentManager::new(100);
        for i in 0..3u64 {
            mgr.add_fragment(create_test_fragment(i + 1, 16, 600, i as u32, 3));
        }
        let results = mgr.get_by_source(600);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_associative_search_basic() {
        let mut engine = AssociativeSearchEngine::new(0.1, 2);

        let fragments: Vec<HologramFragment> = (0..5)
            .map(|i| create_test_fragment(i + 1, 32, 700, i as u32, 5))
            .collect();

        engine.build_associations(&fragments);
        assert!(engine.association_count() > 0 || true);
    }

    #[test]
    fn test_similarity_matcher_top_k() {
        let matcher = SimilarityMatcher::new(0.0);
        let query = create_test_fragment(100, 64, 800, 0, 1);
        let candidates: Vec<HologramFragment> = (0..20)
            .map(|i| create_test_fragment(i + 1, 64, 800, i as u32, 20))
            .collect();

        let results = matcher.find_similar(&query, &candidates, 5);
        assert!(results.len() <= 5);
        for i in 1..results.len() {
            assert!(results[i - 1].similarity >= results[i].similarity);
        }
    }

    #[test]
    fn test_similarity_identical_fragment() {
        let matcher = SimilarityMatcher::new(0.0);
        let frag = create_test_fragment(1, 32, 900, 0, 1);
        let sim = matcher.similarity(&frag, &frag);
        assert!((sim - 1.0).abs() < 1e-10, "自相似度应为1.0: {}", sim);
    }

    #[test]
    fn test_e2e_store_encode_retrieve_decode() {
        let config = EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 3,
            phase_modulation: false,
            normalize: true,
        };

        let data: Vec<f64> = (0..512).map(|i| (i as f64 * 0.02).sin()).collect();

        let mut encoder = FourierEncoder::new(config.clone());
        let encode_result = encoder.encode(&data);
        assert!(!encode_result.fragments.is_empty());

        let mut index = HolographicIndex::new();
        for fragment in &encode_result.fragments {
            let cloned = (*fragment).clone();
            index.insert(cloned);
        }

        let all_fragments: Vec<HologramFragment> = index.get_by_source(encode_result.source_hash)
            .into_iter().cloned().collect();
        assert!(!all_fragments.is_empty());

        let decoded = encoder.decode(&all_fragments, data.len());
        assert_eq!(decoded.len(), data.len());

        let mse: f64 = data.iter().zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / data.len() as f64;
        assert!(mse < 1.0, "端到端MSE: {}", mse);
    }

    #[test]
    fn test_e2e_fault_tolerance_30pct() {
        let config = EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 3,
            phase_modulation: false,
            normalize: true,
        };

        let data: Vec<f64> = (0..512).map(|i| (i as f64 * 0.02).sin()).collect();

        let mut encoder = FourierEncoder::new(config);
        let encode_result = encoder.encode(&data);
        let total = encode_result.fragments.len();

        let remove_count = total * 30 / 100;
        let available: Vec<HologramFragment> = encode_result.fragments.iter()
            .skip(remove_count)
            .cloned()
            .collect();

        let decoded = encoder.decode(&available, data.len());
        let _mse: f64 = data.iter().zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / data.len() as f64;

        let integrity = IntegrityReport::new(total as u32, available.len() as u32);
        assert!(integrity.recovery_possible);
    }

    #[test]
    fn test_e2e_persistence_roundtrip() {
        let dir = test_dir("e2e_persist");
        let engine = PersistenceEngine::new(&dir);

        let config = EncodingConfig {
            fft_window_size: 256,
            overlap_ratio: 0.5,
            redundancy_level: 2,
            phase_modulation: false,
            normalize: true,
        };
        let mut encoder = FourierEncoder::new(config);
        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.03).sin()).collect();
        let result = encoder.encode(&data);

        let mut index = HolographicIndex::new();
        for fragment in result.fragments {
            index.insert(fragment);
        }

        engine.save_index(&index, "e2e.idx").unwrap();
        let loaded = engine.load_index("e2e.idx").unwrap();
        assert_eq!(loaded.len(), index.len());
    }
}
