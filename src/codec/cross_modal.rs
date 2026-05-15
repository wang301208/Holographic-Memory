use num_complex::Complex64;
use std::collections::HashMap;

use crate::foundation::math::cosine_similarity;
use crate::types::{AssociatedItem, FragmentId, HologramFragment};

/// 模态类型标识
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Structured,
    Custom(String),
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modality::Text => write!(f, "text"),
            Modality::Image => write!(f, "image"),
            Modality::Audio => write!(f, "audio"),
            Modality::Video => write!(f, "video"),
            Modality::Structured => write!(f, "structured"),
            Modality::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

/// 模态编码器 trait：将不同模态的输入统一映射到频域向量
pub trait ModalityEncoder: Send + Sync {
    fn modality(&self) -> Modality;
    fn encode(&self, input: &[f64]) -> Vec<f64>;
    fn embedding_dim(&self) -> usize;
}

/// 文本模态编码器：基于频域特征提取
pub struct TextModalityEncoder {
    window_size: usize,
}

impl TextModalityEncoder {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl ModalityEncoder for TextModalityEncoder {
    fn modality(&self) -> Modality {
        Modality::Text
    }

    fn encode(&self, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut features = Vec::with_capacity(self.window_size);

        for i in 0..self.window_size.min(n) {
            let t = i as f64 * 2.0 * std::f64::consts::PI / self.window_size as f64;
            let re: f64 = input.iter().enumerate().map(|(k, &x)| x * (t * k as f64).cos()).sum();
            let im: f64 = input.iter().enumerate().map(|(k, &x)| -x * (t * k as f64).sin()).sum();
            features.push((re * re + im * im).sqrt());
        }

        let max_val = features.iter().cloned().fold(0.0f64, f64::max);
        if max_val > 0.0 {
            for f in features.iter_mut() {
                *f /= max_val;
            }
        }

        features
    }

    fn embedding_dim(&self) -> usize {
        self.window_size
    }
}

/// 图像模态编码器：基于空间频域特征
pub struct ImageModalityEncoder {
    patch_size: usize,
}

impl ImageModalityEncoder {
    pub fn new(patch_size: usize) -> Self {
        Self { patch_size }
    }
}

impl ModalityEncoder for ImageModalityEncoder {
    fn modality(&self) -> Modality {
        Modality::Image
    }

    fn encode(&self, input: &[f64]) -> Vec<f64> {
        let patch_dim = self.patch_size * self.patch_size;
        let num_patches = input.len() / patch_dim;
        if num_patches == 0 {
            return vec![0.0; self.patch_size];
        }

        let mut patch_features = Vec::with_capacity(num_patches * self.patch_size);

        for p in 0..num_patches {
            let start = p * patch_dim;
            let end = (start + patch_dim).min(input.len());
            let patch = &input[start..end];

            for freq in 0..self.patch_size.min(patch.len()) {
                let t = freq as f64 * 2.0 * std::f64::consts::PI / patch.len() as f64;
                let re: f64 = patch.iter().enumerate().map(|(k, &x)| x * (t * k as f64).cos()).sum();
                let im: f64 = patch.iter().enumerate().map(|(k, &x)| -x * (t * k as f64).sin()).sum();
                patch_features.push((re * re + im * im).sqrt());
            }
        }

        let max_val = patch_features.iter().cloned().fold(0.0f64, f64::max);
        if max_val > 0.0 {
            for f in patch_features.iter_mut() {
                *f /= max_val;
            }
        }

        patch_features
    }

    fn embedding_dim(&self) -> usize {
        self.patch_size
    }
}

/// 跨模态映射：将源模态频域映射到目标模态频域
#[derive(Debug, Clone)]
pub struct CrossModalMapping {
    pub source_modality: Modality,
    pub target_modality: Modality,
    pub transform_matrix: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl CrossModalMapping {
    pub fn new(source: Modality, target: Modality, source_dim: usize, target_dim: usize) -> Self {
        let transform_matrix = (0..target_dim)
            .map(|i| {
                (0..source_dim)
                    .map(|j| {
                        let seed = ((i + 1) * 7 + (j + 1) * 13) as f64;
                        (seed * 0.618).fract() * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect();

        let bias = (0..target_dim)
            .map(|i| ((i + 1) as f64 * 0.314).fract() * 0.1)
            .collect();

        Self {
            source_modality: source,
            target_modality: target,
            transform_matrix,
            bias,
        }
    }

    pub fn apply(&self, source_freq: &[f64]) -> Vec<f64> {
        self.transform_matrix
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| {
                let dot: f64 = row.iter().zip(source_freq.iter()).map(|(w, &x)| w * x).sum();
                dot + b
            })
            .collect()
    }

    pub fn source_dim(&self) -> usize {
        self.transform_matrix.first().map(|r| r.len()).unwrap_or(0)
    }

    pub fn target_dim(&self) -> usize {
        self.transform_matrix.len()
    }
}

/// 跨模态联想结果
#[derive(Debug, Clone)]
pub struct CrossModalAssociation {
    pub source_id: FragmentId,
    pub source_modality: Modality,
    pub target_id: FragmentId,
    pub target_modality: Modality,
    pub bridge_confidence: f64,
    pub source_freq: Vec<Complex64>,
    pub projected_freq: Vec<Complex64>,
}

/// 跨模态联想引擎：实现不同模态间的频域桥接
pub struct CrossModalReasoner {
    mappings: HashMap<(Modality, Modality), CrossModalMapping>,
    #[allow(dead_code)]
    modality_registry: HashMap<Modality, FragmentId>,
}

impl CrossModalReasoner {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
            modality_registry: HashMap::new(),
        }
    }

    /// 注册跨模态映射
    pub fn register_mapping(&mut self, mapping: CrossModalMapping) {
        let key = (mapping.source_modality.clone(), mapping.target_modality.clone());
        self.mappings.insert(key, mapping);
    }

    /// 创建文本↔图像双向映射
    pub fn register_text_image_bridge(&mut self, text_dim: usize, image_dim: usize) {
        self.register_mapping(CrossModalMapping::new(
            Modality::Text, Modality::Image, text_dim, image_dim,
        ));
        self.register_mapping(CrossModalMapping::new(
            Modality::Image, Modality::Text, image_dim, text_dim,
        ));
    }

    /// 跨模态联想：从源模态片段检索目标模态片段
    pub fn cross_modal_search(
        &self,
        source_fragment: &HologramFragment,
        source_modality: &Modality,
        target_fragments: &[HologramFragment],
        target_modality: &Modality,
        top_k: usize,
    ) -> Vec<CrossModalAssociation> {
        let key = (source_modality.clone(), target_modality.clone());
        let mapping = match self.mappings.get(&key) {
            Some(m) => m,
            None => return Vec::new(),
        };

        let source_freq: Vec<f64> = source_fragment.frequency_domain.iter()
            .map(|c| c.norm())
            .collect();

        let projected = mapping.apply(&source_freq);

        let projected_complex: Vec<Complex64> = projected
            .iter()
            .enumerate()
            .map(|(i, &mag)| {
                let phase = i as f64 * 0.1;
                Complex64::new(mag * phase.cos(), mag * phase.sin())
            })
            .collect();

        let source_complex: Vec<Complex64> = source_fragment.frequency_domain.iter().copied().collect();

        let mut scored: Vec<(usize, f64)> = target_fragments
            .iter()
            .enumerate()
            .map(|(idx, frag)| {
                let target_freq: Vec<Complex64> = frag.frequency_domain.iter().copied().collect();
                let sim = cosine_similarity(&projected_complex, &target_freq);
                (idx, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(idx, confidence)| CrossModalAssociation {
                source_id: source_fragment.id,
                source_modality: source_modality.clone(),
                target_id: target_fragments[idx].id,
                target_modality: target_modality.clone(),
                bridge_confidence: confidence,
                source_freq: source_complex.clone(),
                projected_freq: projected_complex.clone(),
            })
            .collect()
    }

    /// 跨模态关联检索：返回 AssociatedItem 格式（兼容现有接口）
    pub fn cross_modal_associations(
        &self,
        source_fragment: &HologramFragment,
        source_modality: &Modality,
        target_fragments: &[HologramFragment],
        target_modality: &Modality,
        top_k: usize,
    ) -> Vec<AssociatedItem> {
        self.cross_modal_search(source_fragment, source_modality, target_fragments, target_modality, top_k)
            .into_iter()
            .map(|assoc| AssociatedItem {
                fragment_id: assoc.target_id,
                similarity: assoc.bridge_confidence,
                metadata: crate::types::FragmentMeta::new(0, 0, 0),
            })
            .collect()
    }

    /// 获取已注册的映射数量
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }
}

impl Default for CrossModalReasoner {
    fn default() -> Self {
        Self::new()
    }
}
