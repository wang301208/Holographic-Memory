use ndarray::Array2;
use num_complex::Complex64;

use crate::types::{FragmentId, FragmentMeta, HologramFragment};

pub struct RedundancyWeaver {
    redundancy_level: u8,
}

impl RedundancyWeaver {
    pub fn new(redundancy_level: u8) -> Self {
        Self {
            redundancy_level: redundancy_level.max(1),
        }
    }

    pub fn weave(&self, fragments: &[HologramFragment]) -> Vec<HologramFragment> {
        if fragments.is_empty() {
            return Vec::new();
        }

        let n = fragments.len();
        let mut result = fragments.to_vec();
        let mut id_counter: FragmentId = fragments.iter().map(|f| f.id).max().unwrap_or(0) + 1;

        for level in 1..self.redundancy_level {
            let parity = compute_parity(fragments, level);

            let source_hash = fragments[0].metadata.source_hash;
            let fragment_count = fragments[0].metadata.fragment_count;

            let mut parity_fragment = fragments[0].clone();
            parity_fragment.id = id_counter;
            id_counter += 1;
            parity_fragment.frequency_domain = parity;
            parity_fragment.redundancy_level = self.redundancy_level;
            parity_fragment.metadata = FragmentMeta::new(source_hash, fragment_count, level as u32 - 1);
            parity_fragment.metadata.tags.clear();
            parity_fragment.metadata.tags.push(format!("parity_L{}", level));
            parity_fragment.metadata.tags.push(format!("data_count_{}", n));

            for (i, frag) in fragments.iter().enumerate() {
                parity_fragment.metadata.tags.push(format!("src_{}_{}", i, frag.id));
            }

            result.push(parity_fragment);
        }

        for level in 1..self.redundancy_level {
            for i in 0..n {
                let j = (i + level as usize) % n;
                let combined = combine_fragments(&fragments[i], &fragments[j], level);

                let mut redundant = fragments[i].clone();
                redundant.id = id_counter;
                id_counter += 1;
                redundant.frequency_domain = combined;
                redundant.redundancy_level = self.redundancy_level;
                redundant.metadata = FragmentMeta::new(
                    fragments[i].metadata.source_hash,
                    fragments[i].metadata.fragment_count,
                    fragments[i].metadata.fragment_index,
                );
                redundant.metadata.tags.push(format!("redundancy_L{}", level));
                redundant.metadata.tags.push(format!("source_pair_{}_{}", i, j));

                result.push(redundant);
            }
        }

        result
    }

    pub fn unweave(&self, all_fragments: &[HologramFragment]) -> Vec<HologramFragment> {
        all_fragments
            .iter()
            .filter(|f| {
                !f.metadata.tags.iter().any(|t| t.starts_with("redundancy_L") || t.starts_with("parity_L"))
            })
            .cloned()
            .collect()
    }

    pub fn recover_from_partial(
        &self,
        available: &[HologramFragment],
        total_fragment_count: u32,
    ) -> RecoveryResult {
        let original_fragments: Vec<&HologramFragment> = available
            .iter()
            .filter(|f| {
                !f.metadata.tags.iter().any(|t| t.starts_with("redundancy_L") || t.starts_with("parity_L"))
            })
            .collect();

        let parity_fragments: Vec<&HologramFragment> = available
            .iter()
            .filter(|f| f.metadata.tags.iter().any(|t| t.starts_with("parity_L")))
            .collect();

        let redundant_fragments: Vec<&HologramFragment> = available
            .iter()
            .filter(|f| f.metadata.tags.iter().any(|t| t.starts_with("redundancy_L")))
            .collect();

        let missing_count = total_fragment_count as usize - original_fragments.len();
        let mut recovered = original_fragments.iter().map(|&f| f.clone()).collect::<Vec<_>>();
        let mut recovered_count = 0usize;

        if missing_count > 0 {
            let mut known_ids: std::collections::HashSet<FragmentId> = original_fragments
                .iter()
                .map(|f| f.id)
                .collect();
            let mut known_indices: std::collections::HashSet<u32> = original_fragments
                .iter()
                .map(|f| f.metadata.fragment_index)
                .collect();

            if !parity_fragments.is_empty() {
                let missing_ids: Vec<FragmentId> = (0..total_fragment_count as usize)
                    .filter_map(|idx| {
                        if !known_indices.contains(&(idx as u32)) {
                            find_fragment_id_by_index(available, idx as u32)
                        } else {
                            None
                        }
                    })
                    .collect();

                if missing_ids.len() <= parity_fragments.len() {
                    for &parity_frag in &parity_fragments {
                        let level = extract_parity_level(&parity_frag.metadata.tags);
                        if let Some(data_count) = extract_data_count(&parity_frag.metadata.tags) {
                            let source_ids: Vec<(usize, FragmentId)> = extract_source_ids(&parity_frag.metadata.tags);
                            let missing_in_this: Vec<(usize, FragmentId)> = source_ids.iter()
                                .filter(|(_, id)| !known_ids.contains(id))
                                .cloned()
                                .collect();

                            if missing_in_this.len() == 1 {
                                let (missing_idx, _) = missing_in_this[0];
                                let mut restored = recover_from_parity(
                                    parity_frag,
                                    &recovered,
                                    level,
                                    data_count,
                                    missing_idx,
                                );
                                restored.metadata.tags.retain(|t| !t.starts_with("parity_L") && !t.starts_with("data_count") && !t.starts_with("src_"));
                                if !known_ids.contains(&restored.id) {
                                    known_ids.insert(restored.id);
                                    known_indices.insert(restored.metadata.fragment_index);
                                    recovered.push(restored);
                                    recovered_count += 1;
                                }
                            }
                        }
                    }
                }
            }

            if recovered.len() < total_fragment_count as usize && !redundant_fragments.is_empty() {
                let mut changed = true;
                while changed {
                    changed = false;

                    for &redundant in &redundant_fragments {
                        let source_idx = redundant.metadata.fragment_index;
                        if known_indices.contains(&source_idx) {
                            continue;
                        }

                        if let Some(pair_info) = find_source_pair(&redundant.metadata.tags) {
                            let (pair_a, pair_b) = pair_info;
                            let other_idx = if pair_a == source_idx as usize { pair_b } else { pair_a };

                            if known_indices.contains(&(other_idx as u32)) {
                                let level = extract_redundancy_level(&redundant.metadata.tags);
                                let restored = extract_from_redundant(redundant, level);
                                if !known_indices.contains(&restored.metadata.fragment_index) {
                                    known_ids.insert(restored.id);
                                    known_indices.insert(restored.metadata.fragment_index);
                                    recovered.push(restored);
                                    recovered_count += 1;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        let confidence = if total_fragment_count > 0 {
            (recovered.len() as f64 / total_fragment_count as f64).min(1.0)
        } else {
            1.0
        };

        let total_with_redundancy = total_fragment_count as usize * self.redundancy_level as usize;
        let damage_ratio = if total_with_redundancy > 0 {
            1.0 - available.len() as f64 / total_with_redundancy as f64
        } else {
            0.0
        };

        RecoveryResult {
            fragments: recovered,
            confidence,
            damage_ratio,
            recovered_count,
        }
    }
}

pub struct RecoveryResult {
    pub fragments: Vec<HologramFragment>,
    pub confidence: f64,
    pub damage_ratio: f64,
    pub recovered_count: usize,
}

fn compute_parity(fragments: &[HologramFragment], level: u8) -> Array2<Complex64> {
    if fragments.is_empty() {
        return Array2::from_elem((1, 1), Complex64::new(0.0, 0.0));
    }

    let max_rows = fragments.iter().map(|f| f.frequency_domain.nrows()).max().unwrap_or(1);
    let max_cols = fragments.iter().map(|f| f.frequency_domain.ncols()).max().unwrap_or(1);

    let mut parity = Array2::from_elem((max_rows, max_cols), Complex64::new(0.0, 0.0));
    let root = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * level as f64 / (fragments.len() + 1) as f64);

    for (i, fragment) in fragments.iter().enumerate() {
        let coeff = root.powi(i as i32 + 1);
        for ((r, c), &val) in fragment.frequency_domain.indexed_iter() {
            parity[[r, c]] += val * coeff;
        }
    }

    parity
}

fn recover_from_parity(
    parity_frag: &HologramFragment,
    known_fragments: &[HologramFragment],
    level: u8,
    data_count: usize,
    missing_idx: usize,
) -> HologramFragment {
    let n = data_count;
    let root = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * level as f64 / (n + 1) as f64);
    let missing_coeff = root.powi(missing_idx as i32 + 1);

    let mut recovered = parity_frag.clone();

    for known in known_fragments {
        let known_idx = known.metadata.fragment_index as usize;
        if known_idx >= n {
            continue;
        }
        let coeff = root.powi(known_idx as i32 + 1);
        for ((r, c), &val) in known.frequency_domain.indexed_iter() {
            if r < recovered.frequency_domain.nrows() && c < recovered.frequency_domain.ncols() {
                recovered.frequency_domain[[r, c]] -= val * coeff;
            }
        }
    }

    for val in recovered.frequency_domain.iter_mut() {
        *val /= missing_coeff;
    }

    recovered.metadata.fragment_index = missing_idx as u32;
    recovered
}

fn combine_fragments(a: &HologramFragment, b: &HologramFragment, level: u8) -> Array2<Complex64> {
    let (rows_a, cols_a) = a.frequency_domain.dim();
    let (rows_b, cols_b) = b.frequency_domain.dim();

    let max_rows = rows_a.max(rows_b);
    let max_cols = cols_a.max(cols_b);

    let mut combined = Array2::from_elem((max_rows, max_cols), Complex64::new(0.0, 0.0));

    for ((i, j), &val) in a.frequency_domain.indexed_iter() {
        combined[[i, j]] += val * Complex64::new(0.5, 0.0);
    }

    let phase = std::f64::consts::FRAC_PI_4 * level as f64;
    let rot = Complex64::new(phase.cos(), phase.sin());
    for ((i, j), &val) in b.frequency_domain.indexed_iter() {
        combined[[i, j]] += val * Complex64::new(0.5, 0.0) * rot;
    }

    combined
}

fn extract_redundancy_level(tags: &[String]) -> u8 {
    for tag in tags {
        if let Some(suffix) = tag.strip_prefix("redundancy_L") {
            if let Ok(level) = suffix.parse::<u8>() {
                return level;
            }
        }
    }
    0
}

fn extract_parity_level(tags: &[String]) -> u8 {
    for tag in tags {
        if let Some(suffix) = tag.strip_prefix("parity_L") {
            if let Ok(level) = suffix.parse::<u8>() {
                return level;
            }
        }
    }
    0
}

fn extract_data_count(tags: &[String]) -> Option<usize> {
    for tag in tags {
        if let Some(suffix) = tag.strip_prefix("data_count_") {
            if let Ok(count) = suffix.parse::<usize>() {
                return Some(count);
            }
        }
    }
    None
}

fn extract_source_ids(tags: &[String]) -> Vec<(usize, FragmentId)> {
    let mut result = Vec::new();
    for tag in tags {
        if let Some(suffix) = tag.strip_prefix("src_") {
            let parts: Vec<&str> = suffix.split('_').collect();
            if parts.len() == 2 {
                if let (Ok(idx), Ok(id)) = (parts[0].parse::<usize>(), parts[1].parse::<FragmentId>()) {
                    result.push((idx, id));
                }
            }
        }
    }
    result
}

fn find_fragment_id_by_index(fragments: &[HologramFragment], index: u32) -> Option<FragmentId> {
    fragments.iter()
        .filter(|f| !f.metadata.tags.iter().any(|t| t.starts_with("redundancy_L") || t.starts_with("parity_L")))
        .find(|f| f.metadata.fragment_index == index)
        .map(|f| f.id)
}

fn find_source_pair(tags: &[String]) -> Option<(usize, usize)> {
    for tag in tags {
        if let Some(suffix) = tag.strip_prefix("source_pair_") {
            let parts: Vec<&str> = suffix.split('_').collect();
            if parts.len() == 2 {
                if let (Ok(a), Ok(b)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    return Some((a, b));
                }
            }
        }
    }
    None
}

fn extract_from_redundant(redundant: &HologramFragment, level: u8) -> HologramFragment {
    let mut restored = redundant.clone();

    let phase = std::f64::consts::FRAC_PI_4 * level as f64;
    let rot_conj = Complex64::new(phase.cos(), -phase.sin());

    for ((_i, _j), val) in restored.frequency_domain.indexed_iter_mut() {
        let a_component = *val * Complex64::new(2.0, 0.0);
        let b_estimated = a_component * rot_conj;
        *val = b_estimated;
    }

    restored.metadata.tags.retain(|t| !t.starts_with("redundancy_L") && !t.starts_with("source_pair_"));
    restored
}
