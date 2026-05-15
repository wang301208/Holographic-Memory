#![allow(clippy::needless_range_loop)]

use num_complex::Complex64;

use crate::foundation::math::FourierTransformer;

pub struct QuantumEncoder {
    dimension: usize,
    noise_scale: f64,
    transformer: FourierTransformer,
}

#[derive(Debug, Clone)]
pub struct SuperpositionState {
    pub amplitudes: Vec<Complex64>,
    pub dimension: usize,
    pub norm: f64,
}

#[derive(Debug, Clone)]
pub struct InterferencePattern {
    pub constructive: Vec<usize>,
    pub destructive: Vec<usize>,
    pub coherence: f64,
}

#[derive(Debug, Clone)]
pub struct QuantumEncodedData {
    pub state: SuperpositionState,
    pub phases: Vec<f64>,
    pub basis_labels: Vec<String>,
}

impl SuperpositionState {
    pub fn new(amplitudes: Vec<Complex64>) -> Self {
        let dimension = amplitudes.len();
        let norm = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        Self { amplitudes, dimension, norm }
    }

    pub fn normalized(&self) -> Self {
        if self.norm < 1e-15 {
            return Self::new(vec![Complex64::new(0.0, 0.0); self.dimension]);
        }
        let scaled: Vec<Complex64> = self.amplitudes.iter()
            .map(|a| a / self.norm)
            .collect();
        Self::new(scaled)
    }

    pub fn overlap(&self, other: &SuperpositionState) -> Complex64 {
        let min_len = self.amplitudes.len().min(other.amplitudes.len());
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..min_len {
            sum += self.amplitudes[i].conj() * other.amplitudes[i];
        }
        sum
    }

    pub fn fidelity(&self, other: &SuperpositionState) -> f64 {
        let overlap = self.overlap(other);
        overlap.norm_sqr()
    }

    pub fn probability_distribution(&self) -> Vec<f64> {
        let total: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if total < 1e-15 {
            return vec![0.0; self.dimension];
        }
        self.amplitudes.iter().map(|a| a.norm_sqr() / total).collect()
    }

    pub fn entropy(&self) -> f64 {
        let probs = self.probability_distribution();
        let mut h = 0.0;
        for &p in &probs {
            if p > 1e-15 {
                h -= p * p.ln();
            }
        }
        h
    }
}

impl QuantumEncoder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            noise_scale: 0.01,
            transformer: FourierTransformer::new(),
        }
    }

    pub fn with_noise(dimension: usize, noise_scale: f64) -> Self {
        Self {
            dimension,
            noise_scale,
            transformer: FourierTransformer::new(),
        }
    }

    pub fn encode_superposition(&mut self, data: &[f64]) -> SuperpositionState {
        let n = self.dimension.max(data.len()).next_power_of_two();
        let mut padded = vec![0.0f64; n];
        for (i, &v) in data.iter().enumerate() {
            padded[i] = v;
        }

        let freq = self.transformer.forward(&padded);
        let amplitudes: Vec<Complex64> = freq.iter()
            .map(|&c| {
                let noise_re = self.noise_scale * (c.re * 0.1).sin();
                let noise_im = self.noise_scale * (c.im * 0.1).cos();
                c + Complex64::new(noise_re, noise_im)
            })
            .collect();

        SuperpositionState::new(amplitudes)
    }

    pub fn decode_measurement(&mut self, state: &SuperpositionState) -> Vec<f64> {
        let freq: Vec<Complex64> = state.amplitudes.iter()
            .map(|&a| a - Complex64::new(
                self.noise_scale * (a.re * 0.1).sin(),
                self.noise_scale * (a.im * 0.1).cos(),
            ))
            .collect();

        self.transformer.inverse(&freq).iter().map(|c| c.re).collect()
    }

    pub fn interfere(&self, state_a: &SuperpositionState, state_b: &SuperpositionState) -> (SuperpositionState, InterferencePattern) {
        let n = state_a.amplitudes.len().max(state_b.amplitudes.len());
        let mut result = vec![Complex64::new(0.0, 0.0); n];

        for i in 0..n {
            let a = if i < state_a.amplitudes.len() { state_a.amplitudes[i] } else { Complex64::new(0.0, 0.0) };
            let b = if i < state_b.amplitudes.len() { state_b.amplitudes[i] } else { Complex64::new(0.0, 0.0) };
            result[i] = a + b;
        }

        let mut constructive = Vec::new();
        let mut destructive = Vec::new();

        for i in 0..n {
            let a_mag = if i < state_a.amplitudes.len() { state_a.amplitudes[i].norm() } else { 0.0 };
            let b_mag = if i < state_b.amplitudes.len() { state_b.amplitudes[i].norm() } else { 0.0 };
            let combined = result[i].norm();

            if combined > (a_mag + b_mag) * 0.7 {
                constructive.push(i);
            } else if combined < (a_mag + b_mag) * 0.3 {
                destructive.push(i);
            }
        }

        let coherence = if constructive.len() + destructive.len() > 0 {
            constructive.len() as f64 / (constructive.len() + destructive.len()) as f64
        } else {
            0.5
        };

        let interfered = SuperpositionState::new(result);
        let pattern = InterferencePattern {
            constructive,
            destructive,
            coherence,
        };

        (interfered, pattern)
    }

    pub fn encode_with_phases(&mut self, data: &[f64], labels: Vec<String>) -> QuantumEncodedData {
        let state = self.encode_superposition(data);
        let phases: Vec<f64> = state.amplitudes.iter()
            .map(|a| a.arg())
            .collect();
        QuantumEncodedData {
            state,
            phases,
            basis_labels: labels,
        }
    }

    pub fn phase_interference(&self, encoded_a: &QuantumEncodedData, encoded_b: &QuantumEncodedData) -> f64 {
        let n = encoded_a.phases.len().min(encoded_b.phases.len());
        if n == 0 { return 0.0; }

        let mut coherence: f64 = 0.0;
        for i in 0..n {
            let diff = encoded_a.phases[i] - encoded_b.phases[i];
            coherence += diff.cos();
        }
        coherence / n as f64
    }

    pub fn grover_amplify(&self, state: &SuperpositionState, target_indices: &[usize]) -> SuperpositionState {
        let n = state.amplitudes.len();
        if n == 0 || target_indices.is_empty() {
            return state.clone();
        }

        let target_set: std::collections::HashSet<usize> = target_indices.iter().copied().collect();

        let mut reflected: Vec<Complex64> = state.amplitudes.iter().enumerate()
            .map(|(i, &a)| {
                if target_set.contains(&i) { a } else { -a }
            })
            .collect();

        let mean: Complex64 = reflected.iter().sum::<Complex64>() / n as f64;
        for r in reflected.iter_mut() {
            *r = Complex64::new(2.0, 0.0) * mean - *r;
        }

        SuperpositionState::new(reflected)
    }

    pub fn dimension(&self) -> usize { self.dimension }
    pub fn noise_scale(&self) -> f64 { self.noise_scale }
}

impl std::fmt::Display for SuperpositionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "叠加态[维度={}, 范数={:.4}, 熵={:.4}]", self.dimension, self.norm, self.entropy())
    }
}

impl std::fmt::Display for InterferencePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "干涉[建设性={}, 破坏性={}, 相干度={:.4}]",
            self.constructive.len(), self.destructive.len(), self.coherence)
    }
}

impl std::fmt::Display for QuantumEncodedData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "量子编码[{}, 基底数={}]", self.state, self.basis_labels.len())
    }
}
