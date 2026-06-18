use holographic_memory::*;
use std::path::PathBuf;

fn temp_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("C:/tmp/holo_backend_test_{}/{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

fn signal_data(len: usize) -> Vec<f64> {
    (0..len).map(|i| {
        let t = i as f64 * 0.02;
        t.sin() + 0.5 * (t * 3.0).cos()
    }).collect()
}

fn small_config() -> HolographicConfig {
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

#[test]
fn test_tiered_index_store_retrieve() {
    let dir = temp_dir("tiered_store");
    {
        let tiered_config = TieredConfig {
            l0_capacity: 100,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 2,
            demote_after_access: false,
        };

        let mut hm = HolographicMemory::new(small_config())
            .with_tiered_index(tiered_config)
            .unwrap();

        let data = signal_data(256);
        let result = hm.store(&data).unwrap();
        assert!(result.fragment_count > 0);

        let decoded = hm.retrieve(result.source_hash, data.len()).unwrap();
        assert_eq!(decoded.len(), data.len());
    }
    cleanup(&dir);
}

#[test]
fn test_tiered_index_multiple_sources() {
    let dir = temp_dir("tiered_multi");
    {
        let tiered_config = TieredConfig {
            l0_capacity: 200,
            l1_memtable_capacity: 200,
            l1_dir: dir.join("l1"),
            promote_threshold: 2,
            demote_after_access: false,
        };

        let mut hm = HolographicMemory::new(small_config())
            .with_tiered_index(tiered_config)
            .unwrap();

        let data1 = signal_data(256);
        let data2: Vec<f64> = (0..256).map(|i| (i as f64 * 0.03).cos()).collect();

        let r1 = hm.store(&data1).unwrap();
        let r2 = hm.store(&data2).unwrap();
        assert_ne!(r1.source_hash, r2.source_hash);

        let d1 = hm.retrieve(r1.source_hash, 256).unwrap();
        let d2 = hm.retrieve(r2.source_hash, 256).unwrap();
        assert_eq!(d1.len(), 256);
        assert_eq!(d2.len(), 256);
    }
    cleanup(&dir);
}

#[test]
fn test_rs_store_basic() {
    let mut hm = HolographicMemory::new(small_config())
        .with_reed_solomon(4, 2)
        .unwrap();

    let data = signal_data(256);
    let result = hm.store_with_rs(&data).unwrap();
    assert!(result.fragment_count > 0);
    assert!(hm.rs_codec().is_some());
}

#[test]
fn test_rs_config_validation() {
    let result = HolographicMemory::new(small_config())
        .with_reed_solomon(0, 2);
    assert!(result.is_err());
}

#[test]
fn test_mmap_roundtrip() {
    let dir = temp_dir("mmap_rt");
    let mmap_dir = temp_dir("mmap_files");
    {
        let config = small_config();
        let mut hm = HolographicMemory::new(config)
            .with_persistence(&dir)
            .with_mmap(&mmap_dir);

        let data = signal_data(256);
        let store = hm.store(&data).unwrap();
        assert!(store.fragment_count > 0);

        hm.save_mmap("test.mmap").unwrap();

        let mut hm2 = HolographicMemory::new(small_config())
            .with_mmap(&mmap_dir);
        hm2.load_mmap("test.mmap").unwrap();
        assert!(hm2.fragment_count() > 0);

        let decoded = hm2.retrieve(store.source_hash, data.len()).unwrap();
        assert_eq!(decoded.len(), data.len());
    }
    cleanup(&dir);
    cleanup(&mmap_dir);
}

#[test]
fn test_tiered_with_rs() {
    let dir = temp_dir("tiered_rs");
    {
        let tiered_config = TieredConfig {
            l0_capacity: 100,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 2,
            demote_after_access: false,
        };

        let mut hm = HolographicMemory::new(small_config())
            .with_tiered_index(tiered_config)
            .unwrap()
            .with_reed_solomon(3, 2)
            .unwrap();

        let data = signal_data(256);
        let result = hm.store_with_rs(&data).unwrap();
        assert!(result.fragment_count > 0);
        assert!(hm.rs_codec().is_some());
    }
    cleanup(&dir);
}

#[test]
fn test_full_stack_simple() {
    let dir = temp_dir("full_simple");
    let mmap_dir = temp_dir("full_mmap");
    {
        let config = small_config();
        let mut hm = HolographicMemory::new(config)
            .with_persistence(&dir)
            .with_mmap(&mmap_dir);

        let data = signal_data(512);
        let store_result = hm.store(&data).unwrap();

        let integrity = hm.integrity(store_result.source_hash);
        assert!(integrity.fragments_total > 0 || integrity.fragments_available > 0);

        let decoded = hm.retrieve(store_result.source_hash, data.len()).unwrap();
        let mse: f64 = data.iter().zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / data.len() as f64;
        assert!(mse < 1.0, "MSE应小于1.0，实际: {}", mse);

        hm.save().unwrap();
        hm.save_mmap("full.mmap").unwrap();

        let can = hm.can_recover(store_result.total_fragments, store_result.total_fragments);
        assert!(can);
    }
    cleanup(&dir);
    cleanup(&mmap_dir);
}

#[test]
fn test_fault_tolerance_tiered() {
    let dir = temp_dir("fault_tiered");
    {
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

        let tiered_config = TieredConfig {
            l0_capacity: 200,
            l1_memtable_capacity: 200,
            l1_dir: dir.join("l1"),
            promote_threshold: 3,
            demote_after_access: false,
        };

        let mut hm = HolographicMemory::new(config)
            .with_tiered_index(tiered_config)
            .unwrap();

        let data = signal_data(512);
        let result = hm.store_with_fault_tolerance(&data, 0.2).unwrap();
        assert!(result.total_fragments > 0);
        assert!(result.integrity.damage_ratio <= 1.0);
    }
    cleanup(&dir);
}

#[test]
fn test_recover_and_decode_with_50pct_original_damage() {
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
    let data = signal_data(1024);
    let store = hm.store(&data).unwrap();
    let fragments = hm.all_fragments_pub();

    let mut original: Vec<HologramFragment> = fragments
        .iter()
        .filter(|f| !f.metadata.tags.iter().any(|t| t.starts_with("redundancy_L") || t.starts_with("parity_L")))
        .cloned()
        .collect();
    original.sort_by_key(|f| f.metadata.fragment_index);

    let parity_and_redundancy: Vec<HologramFragment> = fragments
        .iter()
        .filter(|f| f.metadata.tags.iter().any(|t| t.starts_with("redundancy_L") || t.starts_with("parity_L")))
        .cloned()
        .collect();

    let mut available: Vec<HologramFragment> = original
        .iter()
        .take(original.len() / 2)
        .cloned()
        .collect();
    available.extend(parity_and_redundancy);

    let decoded = hm
        .recover_and_decode(&available, store.fragment_count as u32, data.len())
        .unwrap();

    let inner_start = 256;
    let inner_end = data.len() - 256;
    let mse: f64 = data[inner_start..inner_end]
        .iter()
        .zip(decoded[inner_start..inner_end].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / (inner_end - inner_start) as f64;

    assert!(mse < 0.2, "50%损坏恢复后的内部区间 MSE 过大: {}", mse);
}

#[test]
fn test_sparse_index_stores_compressed_fragments_and_retrieves_by_source() {
    let config = small_config();
    let mut encoder = FourierEncoder::new(config.encoding);
    let data = signal_data(512);
    let encoded = encoder.encode(&data);
    assert!(!encoded.fragments.is_empty());

    let mut sparse_index = SparseIndex::new(0.3);
    let dense_bytes: usize = encoded
        .fragments
        .iter()
        .map(|fragment| bincode::serialize(fragment).unwrap().len())
        .sum();

    for fragment in encoded.fragments.clone() {
        sparse_index.insert(fragment);
    }

    let sparse_bytes: usize = sparse_index
        .get_sparse_by_source(encoded.source_hash)
        .iter()
        .map(|fragment| bincode::serialize(fragment).unwrap().len())
        .sum();

    assert_eq!(sparse_index.len(), encoded.fragments.len());
    assert!(
        (sparse_bytes as f64) < (dense_bytes as f64) * 0.3,
        "稀疏索引未达到 0.3x: dense={} sparse={}",
        dense_bytes,
        sparse_bytes
    );

    let restored = sparse_index.get_by_source(encoded.source_hash);
    assert_eq!(restored.len(), encoded.fragments.len());
    assert!(restored.iter().all(|fragment| fragment.metadata.source_hash == encoded.source_hash));
}

#[test]
fn test_holographic_memory_can_use_sparse_index_backend() {
    let mut hm = HolographicMemory::new(small_config()).with_sparse_index(0.3);
    let data = signal_data(512);
    let store = hm.store(&data).unwrap();

    assert!(store.total_fragments > 0);
    assert_eq!(hm.fragment_count(), store.total_fragments);

    let integrity = hm.integrity(store.source_hash);
    assert_eq!(integrity.fragments_available as usize, store.total_fragments);

    let results = hm.search(&data, 5).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_error_paths() {
    let hm = HolographicMemory::new(small_config());
    let result = hm.save_mmap("test.mmap");
    assert!(result.is_err());
}

#[test]
fn test_mmap_without_config_errors() {
    let hm = HolographicMemory::new(small_config());
    let result = hm.save_mmap("test.mmap");
    assert!(result.is_err());
}

#[test]
fn test_rs_without_config_errors() {
    let mut hm = HolographicMemory::new(small_config());
    let data = signal_data(256);
    let result = hm.store_with_rs(&data);
    assert!(result.is_err());
}
