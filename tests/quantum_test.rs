use holographic_memory::*;
use approx::assert_abs_diff_eq;

#[test]
fn test_quantum_encode_decode_roundtrip() {
    let mut encoder = QuantumEncoder::new(8);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let state = encoder.encode_superposition(&data);
    let recovered = encoder.decode_measurement(&state);

    for i in 0..8 {
        assert_abs_diff_eq!(recovered[i], data[i], epsilon = 0.1);
    }
}

#[test]
fn test_superposition_normalized() {
    let state = SuperpositionState::new(vec![
        num_complex::Complex64::new(3.0, 0.0),
        num_complex::Complex64::new(4.0, 0.0),
    ]);
    let normed = state.normalized();
    assert_abs_diff_eq!(normed.norm, 1.0, epsilon = 1e-10);
}

#[test]
fn test_superposition_overlap_and_fidelity() {
    let a = SuperpositionState::new(vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
    ]);
    let b = SuperpositionState::new(vec![
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(1.0, 0.0),
    ]);
    let same = SuperpositionState::new(vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
    ]);

    assert_abs_diff_eq!(a.fidelity(&b), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(a.fidelity(&same), 1.0, epsilon = 1e-10);
}

#[test]
fn test_superposition_entropy() {
    let uniform = SuperpositionState::new(vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(1.0, 0.0),
    ]);
    let pure = SuperpositionState::new(vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
    ]);

    assert!(uniform.entropy() > pure.entropy());
    assert_abs_diff_eq!(uniform.entropy(), 4.0_f64.ln(), epsilon = 1e-10);
}

#[test]
fn test_interference_pattern() {
    let encoder = QuantumEncoder::new(4);
    let a = SuperpositionState::new(vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
    ]);
    let b = SuperpositionState::new(vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(-1.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
    ]);

    let (result, pattern) = encoder.interfere(&a, &b);
    assert!(result.amplitudes[0].norm() > 0.0);
    assert!(pattern.constructive.len() > 0 || pattern.destructive.len() > 0);
}

#[test]
fn test_encode_with_phases() {
    let mut encoder = QuantumEncoder::new(4);
    let data = vec![1.0, 0.0, -1.0, 0.0];
    let encoded = encoder.encode_with_phases(&data, vec!["a".into(), "b".into()]);

    assert_eq!(encoded.phases.len(), encoded.state.amplitudes.len());
    assert_eq!(encoded.basis_labels.len(), 2);
}

#[test]
fn test_phase_interference() {
    let mut encoder = QuantumEncoder::new(4);
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![1.0, 2.0, 3.0, 4.0];
    let enc_a = encoder.encode_with_phases(&data_a, vec![]);
    let enc_b = encoder.encode_with_phases(&data_b, vec![]);

    let coherence = encoder.phase_interference(&enc_a, &enc_b);
    assert_abs_diff_eq!(coherence, 1.0, epsilon = 0.01);
}

#[test]
fn test_grover_amplify() {
    let encoder = QuantumEncoder::new(4);
    let state = SuperpositionState::new(vec![
        num_complex::Complex64::new(0.5, 0.0),
        num_complex::Complex64::new(0.5, 0.0),
        num_complex::Complex64::new(0.5, 0.0),
        num_complex::Complex64::new(0.5, 0.0),
    ]);

    let amplified = encoder.grover_amplify(&state, &[0]);
    assert!(amplified.amplitudes[0].norm() > state.amplitudes[0].norm());
}

#[test]
fn test_display_formatting() {
    let state = SuperpositionState::new(vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(0.0, 1.0),
    ]);
    let s = format!("{}", state);
    assert!(s.contains("叠加态"));
}
