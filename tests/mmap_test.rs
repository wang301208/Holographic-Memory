use holographic_memory::*;
use ndarray::Array2;
use num_complex::Complex64;
use std::path::PathBuf;

fn make_fragment(id: u64, source_hash: u64) -> HologramFragment {
    HologramFragment {
        id,
        frequency_domain: Array2::from_shape_vec((2, 2), vec![
            Complex64::new(id as f64, 1.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap(),
        phase_key: PhaseKey::zero(2),
        redundancy_level: 1,
        metadata: FragmentMeta::new(source_hash, 3, (id % 3) as u32),
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("C:/tmp/holo_mmap_test_{}/{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_mmap_write_and_read() {
    let dir = temp_dir("basic");
    {
        let mp = MmapPersistence::new(&dir);
        let frags = vec![make_fragment(1, 100), make_fragment(2, 100), make_fragment(3, 200)];
        mp.write(&frags, "test.holo").unwrap();

        let read_frags = mp.read_fragments("test.holo").unwrap();
        assert_eq!(read_frags.len(), 3);
        assert_eq!(read_frags[0].id, 1);
        assert_eq!(read_frags[1].id, 2);
        assert_eq!(read_frags[2].id, 3);
    }
    cleanup(&dir);
}

#[test]
fn test_mmap_read_index() {
    let dir = temp_dir("index");
    {
        let mp = MmapPersistence::new(&dir);
        let frags = vec![make_fragment(1, 100), make_fragment(2, 100), make_fragment(3, 200)];
        mp.write(&frags, "idx.holo").unwrap();

        let index = mp.read_index("idx.holo").unwrap();
        assert_eq!(index.len(), 3);
        assert!(index.get(1).is_some());
        assert!(index.get(2).is_some());
        assert!(index.get(3).is_some());
    }
    cleanup(&dir);
}

#[test]
fn test_mmap_write_from_index() {
    let dir = temp_dir("from_index");
    {
        let mut index = HolographicIndex::new();
        index.insert(make_fragment(10, 500));
        index.insert(make_fragment(11, 500));
        index.insert(make_fragment(12, 600));

        let mp = MmapPersistence::new(&dir);
        mp.write_index(&index, "snap.holo").unwrap();

        let loaded = mp.read_index("snap.holo").unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.get(10).unwrap().metadata.source_hash, 500);
    }
    cleanup(&dir);
}

#[test]
fn test_mmap_zero_copy_access() {
    let dir = temp_dir("zerocopy");
    {
        let mp = MmapPersistence::new(&dir);
        let frags = vec![make_fragment(1, 100)];
        mp.write(&frags, "zc.holo").unwrap();

        let reader = mp.read("zc.holo").unwrap();
        assert!(reader.data_len() > 0);
        let slice = reader.as_slice();
        assert_eq!(slice.len(), reader.data_len());
    }
    cleanup(&dir);
}

#[test]
fn test_mmap_file_size() {
    let dir = temp_dir("size");
    {
        let mp = MmapPersistence::new(&dir);
        let frags = vec![make_fragment(1, 100), make_fragment(2, 100)];
        mp.write(&frags, "sz.holo").unwrap();

        let size = mp.file_size("sz.holo").unwrap();
        assert!(size > 16);
    }
    cleanup(&dir);
}

#[test]
fn test_mmap_exists() {
    let dir = temp_dir("exists");
    {
        let mp = MmapPersistence::new(&dir);
        assert!(!mp.exists("no.holo"));

        let frags = vec![make_fragment(1, 100)];
        mp.write(&frags, "yes.holo").unwrap();
        assert!(mp.exists("yes.holo"));
    }
    cleanup(&dir);
}

#[test]
fn test_mmap_invalid_format() {
    let dir = temp_dir("invalid");
    {
        let mp = MmapPersistence::new(&dir);
        let path = dir.join("bad.holo");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, b"INVALIDXXsome data here").unwrap();

        let result = mp.read("bad.holo");
        assert!(result.is_err());
    }
    cleanup(&dir);
}

#[test]
fn test_mmap_roundtrip_preserves_data() {
    let dir = temp_dir("roundtrip");
    {
        let mp = MmapPersistence::new(&dir);
        let orig = vec![make_fragment(1, 100), make_fragment(2, 100), make_fragment(3, 300)];
        mp.write(&orig, "rt.holo").unwrap();

        let loaded = mp.read_fragments("rt.holo").unwrap();
        assert_eq!(loaded.len(), orig.len());
        for (o, l) in orig.iter().zip(loaded.iter()) {
            assert_eq!(o.id, l.id);
            assert_eq!(o.metadata.source_hash, l.metadata.source_hash);
            assert_eq!(o.metadata.fragment_count, l.metadata.fragment_count);
            assert_eq!(o.redundancy_level, l.redundancy_level);
        }
    }
    cleanup(&dir);
}

#[test]
fn test_mmap_empty_fragments() {
    let dir = temp_dir("empty");
    {
        let mp = MmapPersistence::new(&dir);
        let frags: Vec<HologramFragment> = vec![];
        mp.write(&frags, "empty.holo").unwrap();

        let loaded = mp.read_fragments("empty.holo").unwrap();
        assert!(loaded.is_empty());
    }
    cleanup(&dir);
}
