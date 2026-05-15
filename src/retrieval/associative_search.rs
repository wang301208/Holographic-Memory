use std::collections::{HashMap, HashSet};

use crate::retrieval::similarity_matcher::SimilarityMatcher;
use crate::types::{AssociatedItem, FragmentId, FragmentMeta, HologramFragment};

pub struct AssociativeSearchEngine {
    matcher: SimilarityMatcher,
    association_graph: HashMap<FragmentId, Vec<AssociationEdge>>,
    max_hops: u32,
}

struct AssociationEdge {
    target_id: FragmentId,
    weight: f64,
}

impl AssociativeSearchEngine {
    pub fn new(threshold: f64, max_hops: u32) -> Self {
        Self {
            matcher: SimilarityMatcher::new(threshold),
            association_graph: HashMap::new(),
            max_hops,
        }
    }

    pub fn build_associations(&mut self, fragments: &[HologramFragment]) {
        self.association_graph.clear();

        for fragment in fragments.iter() {
            let similar = self.matcher.find_similar(fragment, fragments, 10);

            let edges: Vec<AssociationEdge> = similar
                .iter()
                .filter(|item| item.fragment_id != fragment.id)
                .map(|item| AssociationEdge {
                    target_id: item.fragment_id,
                    weight: item.similarity,
                })
                .collect();

            if !edges.is_empty() {
                self.association_graph.insert(fragment.id, edges);
            }

            for item in &similar {
                if item.fragment_id == fragment.id {
                    continue;
                }
                self.association_graph
                    .entry(item.fragment_id)
                    .or_default()
                    .push(AssociationEdge {
                        target_id: fragment.id,
                        weight: item.similarity,
                    });
            }
        }
    }

    pub fn search(
        &self,
        query: &HologramFragment,
        candidates: &[HologramFragment],
        top_k: usize,
    ) -> Vec<AssociatedItem> {
        let direct = self.matcher.find_similar(query, candidates, top_k);

        let mut associated = Vec::new();
        let mut visited: HashSet<FragmentId> = direct.iter().map(|d| d.fragment_id).collect();
        visited.insert(query.id);

        for item in &direct {
            self.search_hops(item.fragment_id, item.similarity, 1, &mut visited, &mut associated);
        }

        associated.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        associated.truncate(top_k);

        let mut result = direct;
        result.extend(associated);
        result.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(top_k);
        result
    }

    fn search_hops(
        &self,
        from_id: FragmentId,
        from_similarity: f64,
        current_hop: u32,
        visited: &mut HashSet<FragmentId>,
        results: &mut Vec<AssociatedItem>,
    ) {
        if current_hop >= self.max_hops {
            return;
        }

        if let Some(edges) = self.association_graph.get(&from_id) {
            for edge in edges {
                if visited.contains(&edge.target_id) {
                    continue;
                }
                visited.insert(edge.target_id);
                let decay = 0.5_f64.powi(current_hop as i32);
                let sim = from_similarity * edge.weight * decay;
                results.push(AssociatedItem {
                    fragment_id: edge.target_id,
                    similarity: sim,
                    metadata: FragmentMeta::new(0, 0, 0),
                });
                self.search_hops(edge.target_id, sim, current_hop + 1, visited, results);
            }
        }
    }

    pub fn association_count(&self) -> usize {
        self.association_graph.values().map(|v| v.len()).sum::<usize>() / 2
    }
}

impl Default for AssociativeSearchEngine {
    fn default() -> Self {
        Self::new(0.3, 3)
    }
}
