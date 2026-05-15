use holographic_memory::*;
use approx::assert_abs_diff_eq;

#[test]
fn test_simd_dot_product() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    assert_abs_diff_eq!(SimdOps::dot_product(&a, &b), 70.0);
}

#[test]
fn test_simd_add_sub() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let sum = SimdOps::add(&a, &b);
    let diff = SimdOps::sub(&b, &a);
    assert_eq!(sum, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    assert_eq!(diff, vec![9.0, 18.0, 27.0, 36.0, 45.0]);
}

#[test]
fn test_simd_scale() {
    let a = vec![1.0, 2.0, 3.0];
    assert_eq!(SimdOps::scale(&a, 3.0), vec![3.0, 6.0, 9.0]);
}

#[test]
fn test_simd_norm() {
    let a = vec![3.0, 4.0];
    assert_abs_diff_eq!(SimdOps::norm(&a), 5.0);
    assert_abs_diff_eq!(SimdOps::norm_squared(&a), 25.0);
}

#[test]
fn test_simd_cosine_similarity() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let c = vec![1.0, 0.0];
    assert_abs_diff_eq!(SimdOps::cosine_similarity(&a, &b), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(SimdOps::cosine_similarity(&a, &c), 1.0, epsilon = 1e-10);
}

#[test]
fn test_simd_mul_elementwise() {
    let a = vec![2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(SimdOps::mul_elementwise(&a, &b), vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_hadamard_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let transformed = SimdOps::hadamard_transform(&data);
    let recovered = SimdOps::inverse_hadamard(&transformed);
    for i in 0..4 {
        assert_abs_diff_eq!(recovered[i], data[i], epsilon = 1e-10);
    }
}

#[test]
fn test_walsh_hadamard_encode() {
    let data = vec![1.0, 0.0, 1.0, 0.0];
    let encoded = SimdOps::walsh_hadamard_encode(&data, 1);
    assert_eq!(encoded.len(), 4);
}

#[test]
fn test_simd_large_vector() {
    let n = 1024;
    let a: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
    let dot = SimdOps::dot_product(&a, &b);
    let sum = SimdOps::add(&a, &b);
    assert!(dot.is_finite());
    assert_eq!(sum.len(), n);
}
