use ndarray::Array2;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

pub type FragmentId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HologramFragment {
    pub id: FragmentId,
    pub frequency_domain: Array2<Complex64>,
    pub phase_key: PhaseKey,
    pub redundancy_level: u8,
    pub metadata: FragmentMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseKey {
    pub phases: Vec<f64>,
    pub dimension: usize,
}

impl PhaseKey {
    pub fn new(phases: Vec<f64>) -> Self {
        let dimension = phases.len();
        Self { phases, dimension }
    }

    pub fn zero(dimension: usize) -> Self {
        Self {
            phases: vec![0.0; dimension],
            dimension,
        }
    }

    pub fn random(dimension: usize) -> Self {
        let phases: Vec<f64> = (0..dimension)
            .map(|_| rand_simple())
            .collect();
        Self { phases, dimension }
    }
}

fn rand_simple() -> f64 {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(0) };
    }

    SEED.with(|seed| {
        let mut s = seed.get();
        if s == 0 {
            s = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
        }
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed.set(s);
        (s >> 33) as f64 / (1u64 << 31) as f64 * std::f64::consts::TAU
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentMeta {
    pub created_at: u64,
    pub source_hash: u64,
    pub fragment_count: u32,
    pub fragment_index: u32,
    pub tags: Vec<String>,
}

impl FragmentMeta {
    pub fn new(source_hash: u64, fragment_count: u32, fragment_index: u32) -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            created_at,
            source_hash,
            fragment_count,
            fragment_index,
            tags: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub content: Vec<u8>,
    pub confidence: f64,
    pub associations: Vec<AssociatedItem>,
    pub integrity: IntegrityReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociatedItem {
    pub fragment_id: FragmentId,
    pub similarity: f64,
    pub metadata: FragmentMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub fragments_total: u32,
    pub fragments_available: u32,
    pub damage_ratio: f64,
    pub recovery_possible: bool,
}

impl IntegrityReport {
    pub fn new(fragments_total: u32, fragments_available: u32) -> Self {
        let damage_ratio = if fragments_total > 0 {
            1.0 - fragments_available as f64 / fragments_total as f64
        } else {
            0.0
        };
        let recovery_possible = damage_ratio <= 0.5;
        Self {
            fragments_total,
            fragments_available,
            damage_ratio,
            recovery_possible,
        }
    }
}
