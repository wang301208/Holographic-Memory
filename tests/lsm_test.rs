use holographic_memory::*;
use ndarray::Array2;
use num_complex::Complex64;
use std::path::PathBuf;

fn make_fragment(id: u64, source_hash: u64, fragment_count: u32, fragment_index: u32) -> HologramFragment {
    HologramFragment {
        id,
        frequency_domain: Array2::from_shape_vec((1, 1), vec![Complex64::new(id as f64, 0.0)]).unwrap(),
        phase_key: PhaseKey::zero(1),
        redundancy_level: 0,
        metadata: FragmentMeta::new(source_hash, fragment_count, fragment_index),
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("C:/tmp/holo_lsm_test_{}/{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_lsm_insert_and_get() {
    let dir = temp_dir("basic");
    {
        let mut lsm = LsmIndex::open(&dir).unwrap();
        let f1 = make_fragment(1, 100, 2, 0);
        let f2 = make_fragment(2, 100, 2, 1);
        lsm.insert(f1).unwrap();
        lsm.insert(f2).unwrap();

        let got = lsm.get(1).unwrap().unwrap();
        assert_eq!(got.id, 1);
        let got2 = lsm.get(2).unwrap().unwrap();
        assert_eq!(got2.id, 2);
        assert!(lsm.get(99).unwrap().is_none());
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_get_by_source() {
    let dir = temp_dir("source");
    {
        let mut lsm = LsmIndex::open(&dir).unwrap();
        lsm.insert(make_fragment(1, 100, 2, 0)).unwrap();
        lsm.insert(make_fragment(2, 100, 2, 1)).unwrap();
        lsm.insert(make_fragment(3, 200, 1, 0)).unwrap();

        let frags = lsm.get_by_source(100).unwrap();
        assert_eq!(frags.len(), 2);
        let frags2 = lsm.get_by_source(200).unwrap();
        assert_eq!(frags2.len(), 1);
        let frags3 = lsm.get_by_source(999).unwrap();
        assert_eq!(frags3.len(), 0);
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_remove() {
    let dir = temp_dir("remove");
    {
        let mut lsm = LsmIndex::open(&dir).unwrap();
        lsm.insert(make_fragment(1, 100, 1, 0)).unwrap();
        lsm.insert(make_fragment(2, 200, 1, 0)).unwrap();

        let removed = lsm.remove(1).unwrap().unwrap();
        assert_eq!(removed.id, 1);
        assert!(lsm.get(1).unwrap().is_none());
        assert!(lsm.get(2).unwrap().is_some());
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_flush_to_sstable() {
    let dir = temp_dir("flush");
    {
        let mut lsm = LsmIndex::open_with_capacity(&dir, 4).unwrap();
        for i in 1..=10u64 {
            lsm.insert(make_fragment(i, 100, 10, (i - 1) as u32)).unwrap();
        }
        lsm.flush().unwrap();

        let stats = lsm.stats();
        assert_eq!(stats.total_entries, 10);

        for i in 1..=10u64 {
            assert!(lsm.get(i).unwrap().is_some(), "片段{}未找到", i);
        }
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_reopen() {
    let dir = temp_dir("reopen");
    {
        let mut lsm = LsmIndex::open_with_capacity(&dir, 4).unwrap();
        for i in 1..=6u64 {
            lsm.insert(make_fragment(i, 100, 6, (i - 1) as u32)).unwrap();
        }
        lsm.flush().unwrap();
        drop(lsm);
    }
    {
        let lsm = LsmIndex::open(&dir).unwrap();
        for i in 1..=6u64 {
            let frag = lsm.get(i).unwrap();
            assert!(frag.is_some(), "重开后片段{}未找到", i);
            assert_eq!(frag.unwrap().id, i);
        }
        assert_eq!(lsm.len(), 6);
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_integrity_check() {
    let dir = temp_dir("integrity");
    {
        let mut lsm = LsmIndex::open(&dir).unwrap();
        lsm.insert(make_fragment(1, 100, 4, 0)).unwrap();
        lsm.insert(make_fragment(2, 100, 4, 1)).unwrap();

        let report = lsm.integrity_check(100).unwrap();
        assert_eq!(report.fragments_total, 4);
        assert_eq!(report.fragments_available, 2);
        assert_eq!(report.damage_ratio, 0.5);
        assert!(report.recovery_possible);
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_all_fragments_and_source_hashes() {
    let dir = temp_dir("all");
    {
        let mut lsm = LsmIndex::open(&dir).unwrap();
        lsm.insert(make_fragment(1, 100, 1, 0)).unwrap();
        lsm.insert(make_fragment(2, 200, 1, 0)).unwrap();
        lsm.insert(make_fragment(3, 300, 1, 0)).unwrap();

        let all = lsm.all_fragments().unwrap();
        assert_eq!(all.len(), 3);

        let hashes = lsm.all_source_hashes();
        assert_eq!(hashes.len(), 3);
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_compact() {
    let dir = temp_dir("compact");
    {
        let mut lsm = LsmIndex::open_with_capacity(&dir, 2).unwrap();
        for i in 1..=10u64 {
            lsm.insert(make_fragment(i, 100, 10, (i - 1) as u32)).unwrap();
        }
        lsm.compact().unwrap();

        for i in 1..=10u64 {
            assert!(lsm.get(i).unwrap().is_some());
        }
        assert_eq!(lsm.len(), 10);
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_batch_insert() {
    let dir = temp_dir("batch");
    {
        let mut lsm = LsmIndex::open(&dir).unwrap();
        let frags: Vec<HologramFragment> = (1..=5u64)
            .map(|i| make_fragment(i, 100, 5, (i - 1) as u32))
            .collect();
        let ids = lsm.insert_batch(frags).unwrap();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(lsm.len(), 5);
    }
    cleanup(&dir);
}

#[test]
fn test_lsm_stats_display() {
    let dir = temp_dir("stats");
    {
        let mut lsm = LsmIndex::open(&dir).unwrap();
        lsm.insert(make_fragment(1, 100, 1, 0)).unwrap();
        let stats = lsm.stats();
        let display = format!("{}", stats);
        assert!(display.contains("LSM索引统计"));
        assert!(display.contains("MemTable"));
    }
    cleanup(&dir);
}
