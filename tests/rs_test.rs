use holographic_memory::*;

fn make_data_shards(n: usize, block_len: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| vec![(i as u8 + 1).wrapping_mul(0x11); block_len]).collect()
}

#[test]
fn test_rs_encode_basic() {
    let rs = ReedSolomon::new(3, 2).unwrap();
    let data = make_data_shards(3, 4);
    let parity = rs.encode(&data).unwrap();
    assert_eq!(parity.len(), 2);
    assert_eq!(parity[0].len(), 4);
}

#[test]
fn test_rs_verify_valid() {
    let rs = ReedSolomon::new(3, 2).unwrap();
    let data = make_data_shards(3, 8);
    let parity = rs.encode(&data).unwrap();
    assert!(rs.verify(&data, &parity));
}

#[test]
fn test_rs_reconstruct_no_loss() {
    let rs = ReedSolomon::new(3, 2).unwrap();
    let data = make_data_shards(3, 8);
    let parity = rs.encode(&data).unwrap();

    let mut shards: Vec<Option<Vec<u8>>> = data.into_iter().map(Some).collect();
    for p in parity {
        shards.push(Some(p));
    }

    let reconstructed = rs.reconstruct(&shards).unwrap();
    for i in 0..3 {
        assert_eq!(shards[i].as_ref().unwrap(), &reconstructed[i]);
    }
}

#[test]
fn test_rs_reconstruct_one_lost_data() {
    let rs = ReedSolomon::new(3, 2).unwrap();
    let data = make_data_shards(3, 8);
    let parity = rs.encode(&data).unwrap();

    let mut shards: Vec<Option<Vec<u8>>> = data.into_iter().map(Some).collect();
    for p in parity {
        shards.push(Some(p));
    }

    let original = shards[1].as_ref().unwrap().clone();
    shards[1] = None;

    let mut shards_mut = shards;
    rs.reconstruct_data(&mut shards_mut).unwrap();
    assert_eq!(shards_mut[1].as_ref().unwrap(), &original);
}

#[test]
fn test_rs_reconstruct_two_lost_data() {
    let rs = ReedSolomon::new(4, 3).unwrap();
    let data = make_data_shards(4, 16);
    let parity = rs.encode(&data).unwrap();

    let mut shards: Vec<Option<Vec<u8>>> = data.into_iter().map(Some).collect();
    for p in parity {
        shards.push(Some(p));
    }

    let orig0 = shards[0].as_ref().unwrap().clone();
    let orig2 = shards[2].as_ref().unwrap().clone();
    shards[0] = None;
    shards[2] = None;

    let mut shards_mut = shards;
    rs.reconstruct_data(&mut shards_mut).unwrap();
    assert_eq!(shards_mut[0].as_ref().unwrap(), &orig0);
    assert_eq!(shards_mut[2].as_ref().unwrap(), &orig2);
}

#[test]
fn test_rs_reconstruct_lost_parity() {
    let rs = ReedSolomon::new(3, 2).unwrap();
    let data = make_data_shards(3, 8);
    let parity = rs.encode(&data).unwrap();

    let mut shards: Vec<Option<Vec<u8>>> = data.into_iter().map(Some).collect();
    for p in parity {
        shards.push(Some(p));
    }

    shards[4] = None;

    let mut shards_mut = shards;
    rs.reconstruct_data(&mut shards_mut).unwrap();
    for i in 0..3 {
        assert!(shards_mut[i].is_some());
    }
}

#[test]
fn test_rs_insufficient_shards_error() {
    let rs = ReedSolomon::new(4, 2).unwrap();
    let data = make_data_shards(4, 8);
    let parity = rs.encode(&data).unwrap();

    let mut shards: Vec<Option<Vec<u8>>> = data.into_iter().map(Some).collect();
    for p in parity {
        shards.push(Some(p));
    }

    shards[0] = None;
    shards[1] = None;
    shards[2] = None;

    let result = rs.reconstruct(&shards);
    assert!(result.is_err());
}

#[test]
fn test_rs_error_cases() {
    assert!(ReedSolomon::new(0, 2).is_err());
    assert!(ReedSolomon::new(3, 0).is_err());
    assert!(ReedSolomon::new(200, 60).is_err());
}

#[test]
fn test_rs_tolerance_and_ratio() {
    let rs = ReedSolomon::new(4, 2).unwrap();
    assert_eq!(rs.erasure_tolerance(), 2);
    assert!((rs.max_recoverable_damage_ratio() - 2.0 / 6.0).abs() < 1e-10);
}

#[test]
fn test_rs_display() {
    let rs = ReedSolomon::new(4, 2).unwrap();
    let s = format!("{}", rs);
    assert!(s.contains("Reed-Solomon"));
    assert!(s.contains("4/6"));
}

#[test]
fn test_rs_large_block() {
    let rs = ReedSolomon::new(3, 2).unwrap();
    let data: Vec<Vec<u8>> = (0..3).map(|i| vec![(i as u8).wrapping_add(42); 1024]).collect();
    let parity = rs.encode(&data).unwrap();
    assert!(rs.verify(&data, &parity));

    let mut shards: Vec<Option<Vec<u8>>> = data.into_iter().map(Some).collect();
    for p in parity { shards.push(Some(p)); }
    let orig0 = shards[0].as_ref().unwrap().clone();
    let orig1 = shards[1].as_ref().unwrap().clone();
    shards[0] = None;
    shards[1] = None;

    let mut shards_mut = shards;
    rs.reconstruct_data(&mut shards_mut).unwrap();
    assert_eq!(shards_mut[0].as_ref().unwrap(), &orig0);
    assert_eq!(shards_mut[1].as_ref().unwrap(), &orig1);
}
