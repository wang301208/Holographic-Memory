use std::collections::BTreeMap;
use std::hash::Hasher;

use crate::types::FragmentId;

/// 一致性哈希环错误
#[derive(Debug, thiserror::Error)]
pub enum RingError {
    #[error("环为空，无可用节点")]
    EmptyRing,
    #[error("节点已存在: {0}")]
    NodeExists(String),
    #[error("节点不存在: {0}")]
    NodeNotFound(String),
}

/// 虚拟节点（虚拟节点映射到物理节点）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VirtualNode {
    pub hash: u64,
    pub node_id: String,
}

impl PartialOrd for VirtualNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VirtualNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash.cmp(&other.hash)
    }
}

/// 一致性哈希环
///
/// 使用虚拟节点实现负载均衡，支持动态节点加入/离开。
/// 数据分片通过 SipHash 映射到环上，顺时针查找最近的节点。
pub struct ConsistentHashRing {
    ring: BTreeMap<u64, String>,
    nodes: Vec<String>,
    virtual_node_count: usize,
}

impl ConsistentHashRing {
    pub fn new(virtual_node_count: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            nodes: Vec::new(),
            virtual_node_count,
        }
    }

    /// 计算键的哈希值（SipHash-2-4）
    pub fn hash_key(key: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write(key.as_bytes());
        hasher.finish()
    }

    /// 计算虚拟节点的哈希值
    fn hash_virtual_node(node_id: &str, replica_index: usize) -> u64 {
        let key = format!("{}:vn:{}", node_id, replica_index);
        Self::hash_key(&key)
    }

    /// 添加节点到哈希环
    pub fn add_node(&mut self, node_id: &str) -> Result<(), RingError> {
        if self.nodes.contains(&node_id.to_string()) {
            return Err(RingError::NodeExists(node_id.to_string()));
        }

        for i in 0..self.virtual_node_count {
            let hash = Self::hash_virtual_node(node_id, i);
            self.ring.insert(hash, node_id.to_string());
        }

        self.nodes.push(node_id.to_string());
        Ok(())
    }

    /// 从哈希环移除节点
    pub fn remove_node(&mut self, node_id: &str) -> Result<(), RingError> {
        if !self.nodes.contains(&node_id.to_string()) {
            return Err(RingError::NodeNotFound(node_id.to_string()));
        }

        for i in 0..self.virtual_node_count {
            let hash = Self::hash_virtual_node(node_id, i);
            self.ring.remove(&hash);
        }

        self.nodes.retain(|n| n != node_id);
        Ok(())
    }

    /// 查找键所属的主节点（顺时针第一个虚拟节点）
    pub fn find_node(&self, key: &str) -> Result<&str, RingError> {
        if self.ring.is_empty() {
            return Err(RingError::EmptyRing);
        }

        let hash = Self::hash_key(key);

        match self.ring.range(hash..).next() {
            Some((_, node_id)) => Ok(node_id),
            None => Ok(self.ring.iter().next().map(|(_, n)| n.as_str()).unwrap()),
        }
    }

    /// 查找键的 N 个副本节点（用于冗余存储）
    pub fn find_replicas(&self, key: &str, replica_count: usize) -> Vec<String> {
        if self.ring.is_empty() || replica_count == 0 {
            return Vec::new();
        }

        let hash = Self::hash_key(key);
        let mut replicas = Vec::with_capacity(replica_count);
        let mut seen = std::collections::HashSet::new();

        let total = self.ring.len();
        let start = self.ring.range(hash..);

        for (_, node_id) in start.chain(self.ring.iter()) {
            if seen.len() >= replica_count {
                break;
            }
            if seen.insert(node_id.clone()) {
                replicas.push(node_id.clone());
            }
            if seen.len() >= total {
                break;
            }
        }

        replicas
    }

    /// 查找片段 ID 所属的节点
    pub fn find_node_for_fragment(&self, fragment_id: FragmentId) -> Result<&str, RingError> {
        let key = format!("frag:{:?}", fragment_id);
        self.find_node(&key)
    }

    /// 获取所有节点
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 获取虚拟节点总数
    pub fn virtual_node_count_total(&self) -> usize {
        self.ring.len()
    }

    /// 计算节点负载分布（每个物理节点负责的虚拟节点数）
    pub fn load_distribution(&self) -> BTreeMap<String, usize> {
        let mut dist = BTreeMap::new();
        for node_id in &self.nodes {
            dist.insert(node_id.clone(), 0);
        }
        for node_id in self.ring.values() {
            *dist.entry(node_id.clone()).or_insert(0) += 1;
        }
        dist
    }

    /// 计算迁移计划：当新节点加入时，需要从哪些节点迁移哪些哈希范围
    pub fn compute_migration_plan(&self, new_node_id: &str) -> Vec<(String, u64, u64)> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let mut plan = Vec::new();
        let _new_node_hash = Self::hash_virtual_node(new_node_id, 0);

        for i in 0..self.virtual_node_count {
            let vhash = Self::hash_virtual_node(new_node_id, i);

            if let Some((_, source_node)) = self.ring.range(vhash..).next()
                .or_else(|| self.ring.iter().next())
            {
                if source_node != new_node_id {
                    let next_vhash = Self::hash_virtual_node(new_node_id, (i + 1) % self.virtual_node_count);
                    plan.push((source_node.clone(), vhash, next_vhash));
                }
            }
        }

        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_remove_node() {
        let mut ring = ConsistentHashRing::new(150);
        ring.add_node("node1").unwrap();
        ring.add_node("node2").unwrap();
        assert_eq!(ring.node_count(), 2);
        ring.remove_node("node1").unwrap();
        assert_eq!(ring.node_count(), 1);
    }

    #[test]
    fn test_find_node() {
        let mut ring = ConsistentHashRing::new(150);
        ring.add_node("node1").unwrap();
        let node = ring.find_node("test_key").unwrap();
        assert_eq!(node, "node1");
    }

    #[test]
    fn test_find_replicas() {
        let mut ring = ConsistentHashRing::new(150);
        ring.add_node("node1").unwrap();
        ring.add_node("node2").unwrap();
        ring.add_node("node3").unwrap();
        let replicas = ring.find_replicas("key1", 2);
        assert!(replicas.len() <= 2);
        assert!(replicas.len() >= 1);
    }

    #[test]
    fn test_load_distribution() {
        let mut ring = ConsistentHashRing::new(150);
        ring.add_node("node1").unwrap();
        ring.add_node("node2").unwrap();
        let dist = ring.load_distribution();
        assert_eq!(dist.values().sum::<usize>(), 300);
    }

    #[test]
    fn test_duplicate_node_error() {
        let mut ring = ConsistentHashRing::new(150);
        ring.add_node("node1").unwrap();
        assert!(ring.add_node("node1").is_err());
    }
}
