use holographic_memory::*;
use ndarray::Array2;
use num_complex::Complex64;
use std::path::PathBuf;

fn make_frag(id: u64, source: u64) -> HologramFragment {
    HologramFragment {
        id,
        frequency_domain: Array2::from_shape_vec((1, 1), vec![Complex64::new(id as f64, 0.0)]).unwrap(),
        phase_key: PhaseKey::zero(1),
        redundancy_level: 0,
        metadata: FragmentMeta::new(source, 3, (id % 3) as u32),
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("C:/tmp/holo_tiered_test_{}/{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_tiered_insert_and_get() {
    let dir = temp_dir("basic");
    {
        let config = TieredConfig {
            l0_capacity: 100,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 3,
            demote_after_access: false,
        };
        let mut idx = TieredIndex::new(config).unwrap();
        idx.insert(make_frag(1, 100)).unwrap();
        idx.insert(make_frag(2, 100)).unwrap();
        idx.insert(make_frag(3, 200)).unwrap();

        assert_eq!(idx.get(1).unwrap().unwrap().id, 1);
        assert_eq!(idx.get(2).unwrap().unwrap().id, 2);
        assert!(idx.get(99).unwrap().is_none());
    }
    cleanup(&dir);
}

#[test]
fn test_tiered_eviction_to_l1() {
    let dir = temp_dir("evict");
    {
        let config = TieredConfig {
            l0_capacity: 3,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 100,
            demote_after_access: false,
        };
        let mut idx = TieredIndex::new(config).unwrap();
        for i in 1..=5u64 {
            idx.insert(make_frag(i, 100)).unwrap();
        }

        for i in 1..=5u64 {
            let frag = idx.get(i).unwrap();
            assert!(frag.is_some(), "片段{}应存在", i);
        }
        assert_eq!(idx.len(), 5);
    }
    cleanup(&dir);
}

#[test]
fn test_tiered_locate() {
    let dir = temp_dir("locate");
    {
        let config = TieredConfig {
            l0_capacity: 100,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 100,
            demote_after_access: false,
        };
        let mut idx = TieredIndex::new(config).unwrap();
        idx.insert(make_frag(1, 100)).unwrap();
        assert_eq!(idx.locate(1).unwrap(), Some(Layer::L0));
        assert_eq!(idx.locate(99).unwrap(), None);
    }
    cleanup(&dir);
}

#[test]
fn test_tiered_promote_demote() {
    let dir = temp_dir("promote");
    {
        let config = TieredConfig {
            l0_capacity: 100,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 100,
            demote_after_access: false,
        };
        let mut idx = TieredIndex::new(config).unwrap();
        idx.insert(make_frag(1, 100)).unwrap();
        idx.insert_at(make_frag(2, 100), Layer::L1).unwrap();

        assert_eq!(idx.locate(1).unwrap(), Some(Layer::L0));
        assert_eq!(idx.locate(2).unwrap(), Some(Layer::L1));

        let demoted = idx.demote(1).unwrap();
        assert!(demoted);
        assert_eq!(idx.locate(1).unwrap(), Some(Layer::L1));

        let promoted = idx.promote(1).unwrap();
        assert!(promoted);
        assert_eq!(idx.locate(1).unwrap(), Some(Layer::L0));
    }
    cleanup(&dir);
}

#[test]
fn test_tiered_get_by_source() {
    let dir = temp_dir("source");
    {
        let config = TieredConfig {
            l0_capacity: 100,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 100,
            demote_after_access: false,
        };
        let mut idx = TieredIndex::new(config).unwrap();
        idx.insert(make_frag(1, 100)).unwrap();
        idx.insert(make_frag(2, 100)).unwrap();
        idx.insert(make_frag(3, 200)).unwrap();

        let frags = idx.get_by_source(100).unwrap();
        assert_eq!(frags.len(), 2);
    }
    cleanup(&dir);
}

#[test]
fn test_tiered_remove() {
    let dir = temp_dir("remove");
    {
        let config = TieredConfig {
            l0_capacity: 100,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 100,
            demote_after_access: false,
        };
        let mut idx = TieredIndex::new(config).unwrap();
        idx.insert(make_frag(1, 100)).unwrap();
        let removed = idx.remove(1).unwrap().unwrap();
        assert_eq!(removed.id, 1);
        assert!(idx.get(1).unwrap().is_none());
    }
    cleanup(&dir);
}

#[test]
fn test_tiered_stats() {
    let dir = temp_dir("stats");
    {
        let config = TieredConfig {
            l0_capacity: 100,
            l1_memtable_capacity: 100,
            l1_dir: dir.join("l1"),
            promote_threshold: 3,
            demote_after_access: false,
        };
        let mut idx = TieredIndex::new(config).unwrap();
        idx.insert(make_frag(1, 100)).unwrap();
        idx.insert(make_frag(2, 100)).unwrap();

        let stats = idx.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.l0_entries, 2);
        let display = format!("{}", stats);
        assert!(display.contains("分层索引统计"));
    }
    cleanup(&dir);
}

#[test]
fn test_tiered_flush_and_compact() {
    let dir = temp_dir("flush");
    {
        let config = TieredConfig {
            l0_capacity: 2,
            l1_memtable_capacity: 2,
            l1_dir: dir.join("l1"),
            promote_threshold: 100,
            demote_after_access: false,
        };
        let mut idx = TieredIndex::new(config).unwrap();
        for i in 1..=6u64 {
            idx.insert(make_frag(i, 100)).unwrap();
        }
        idx.flush().unwrap();
        idx.compact().unwrap();
        assert_eq!(idx.len(), 6);
    }
    cleanup(&dir);
}
