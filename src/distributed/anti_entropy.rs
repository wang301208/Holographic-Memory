use std::collections::HashMap;

use crate::types::FragmentId;

/// Merkle 树节点
#[derive(Debug, Clone)]
pub struct MerkleNode {
    pub hash: u64,
    pub left: Option<Box<MerkleNode>>,
    pub right: Option<Box<MerkleNode>>,
    pub range_start: u64,
    pub range_end: u64,
}

impl MerkleNode {
    pub fn leaf(hash: u64, range_start: u64, range_end: u64) -> Self {
        Self {
            hash,
            left: None,
            right: None,
            range_start,
            range_end,
        }
    }

    pub fn internal(left: MerkleNode, right: MerkleNode) -> Self {
        let combined = left.hash.wrapping_mul(31).wrapping_add(right.hash);
        Self {
            hash: combined,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            range_start: 0,
            range_end: 0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

/// Merkle 树：用于高效检测副本间差异
pub struct MerkleTree {
    root: Option<MerkleNode>,
    bucket_size: usize,
}

impl MerkleTree {
    pub fn new(bucket_size: usize) -> Self {
        Self {
            root: None,
            bucket_size,
        }
    }

    /// 从片段摘要构建 Merkle 树
    pub fn build_from_digests(&mut self, digests: &[(u64, u64)]) {
        if digests.is_empty() {
            self.root = None;
            return;
        }

        let mut sorted = digests.to_vec();
        sorted.sort_by_key(|(id, _)| *id);

        let buckets: Vec<Vec<(u64, u64)>> = sorted.chunks(self.bucket_size)
            .map(|c| c.to_vec())
            .collect();

        let leaves: Vec<MerkleNode> = buckets.iter().map(|bucket| {
            let combined_hash: u64 = bucket.iter()
                .map(|(_, h)| *h)
                .fold(0u64, |acc, h| acc.wrapping_mul(31).wrapping_add(h));

            let range_start = bucket.first().map(|(id, _)| *id).unwrap_or(0);
            let range_end = bucket.last().map(|(id, _)| *id).unwrap_or(0);

            MerkleNode::leaf(combined_hash, range_start, range_end)
        }).collect();

        self.root = Some(Self::build_tree(&leaves));
    }

    fn build_tree(nodes: &[MerkleNode]) -> MerkleNode {
        if nodes.len() == 1 {
            return nodes[0].clone();
        }

        let mid = nodes.len() / 2;
        let left = Self::build_tree(&nodes[..mid]);
        let right = Self::build_tree(&nodes[mid..]);
        MerkleNode::internal(left, right)
    }

    pub fn root_hash(&self) -> u64 {
        self.root.as_ref().map(|n| n.hash).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// 与另一棵 Merkle 树比较，返回差异报告
    pub fn diff(&self, other: &MerkleTree) -> DiffReport {
        match (&self.root, &other.root) {
            (None, None) => DiffReport { identical: true, differing_ranges: Vec::new() },
            (Some(_), None) => DiffReport {
                identical: false,
                differing_ranges: vec![(0, u64::MAX)],
            },
            (None, Some(_)) => DiffReport {
                identical: false,
                differing_ranges: vec![(0, u64::MAX)],
            },
            (Some(a), Some(b)) => {
                let mut ranges = Vec::new();
                Self::diff_nodes(a, b, &mut ranges);
                DiffReport {
                    identical: ranges.is_empty(),
                    differing_ranges: ranges,
                }
            }
        }
    }

    fn diff_nodes(a: &MerkleNode, b: &MerkleNode, ranges: &mut Vec<(u64, u64)>) {
        if a.hash == b.hash {
            return;
        }

        if a.is_leaf() || b.is_leaf() {
            let start = a.range_start.min(b.range_start);
            let end = a.range_end.max(b.range_end);
            ranges.push((start, end));
            return;
        }

        if let (Some(ref a_left), Some(ref a_right)) = (&a.left, &a.right) {
            if let (Some(ref b_left), Some(ref b_right)) = (&b.left, &b.right) {
                Self::diff_nodes(a_left, b_left, ranges);
                Self::diff_nodes(a_right, b_right, ranges);
            }
        }
    }
}

/// 差异报告
#[derive(Debug, Clone)]
pub struct DiffReport {
    pub identical: bool,
    pub differing_ranges: Vec<(u64, u64)>,
}

/// 同步动作
#[derive(Debug, Clone)]
pub enum SyncAction {
    Send { fragment_ids: Vec<FragmentId> },
    Request { fragment_ids: Vec<FragmentId> },
    Skip,
}

/// 修复任务状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
}

/// 修复任务
#[derive(Debug, Clone)]
pub struct RepairTask {
    pub task_id: u64,
    pub source_node: String,
    pub target_node: String,
    pub fragment_ids: Vec<FragmentId>,
    pub status: RepairStatus,
    pub differing_ranges: Vec<(u64, u64)>,
}

/// 反熵修复引擎
///
/// 基于 Merkle 树的差异检测 + 增量修复：
/// 1. 各节点维护本地数据的 Merkle 树
/// 2. 定期与副本节点交换 Merkle 树根哈希
/// 3. 根哈希不同时递归比较子树，定位差异范围
/// 4. 对差异范围执行增量同步
pub struct AntiEntropyRepair {
    local_tree: MerkleTree,
    repair_tasks: HashMap<u64, RepairTask>,
    next_task_id: u64,
    max_concurrent_repairs: usize,
    local_digests: Vec<(u64, u64)>,
}

impl AntiEntropyRepair {
    pub fn new(bucket_size: usize, max_concurrent_repairs: usize) -> Self {
        Self {
            local_tree: MerkleTree::new(bucket_size),
            repair_tasks: HashMap::new(),
            next_task_id: 1,
            max_concurrent_repairs,
            local_digests: Vec::new(),
        }
    }

    /// 更新本地摘要数据
    pub fn update_digests(&mut self, digests: Vec<(u64, u64)>) {
        self.local_digests = digests;
        self.local_tree.build_from_digests(&self.local_digests);
    }

    /// 获取本地 Merkle 树根哈希
    pub fn root_hash(&self) -> u64 {
        self.local_tree.root_hash()
    }

    /// 与远程节点的 Merkle 树比较，生成差异报告
    pub fn compare_with(&self, remote_tree: &MerkleTree) -> DiffReport {
        self.local_tree.diff(remote_tree)
    }

    /// 创建修复任务
    pub fn create_repair_task(
        &mut self,
        source_node: String,
        target_node: String,
        differing_ranges: Vec<(u64, u64)>,
    ) -> u64 {
        let task_id = self.next_task_id;
        self.next_task_id += 1;

        let fragment_ids: Vec<FragmentId> = differing_ranges.iter()
            .flat_map(|(start, end)| {
                self.local_digests.iter()
                    .filter(|(id, _)| *id >= *start && *id <= *end)
                    .map(|(id, _)| *id as FragmentId)
                    .collect::<Vec<_>>()
            })
            .collect();

        self.repair_tasks.insert(task_id, RepairTask {
            task_id,
            source_node,
            target_node,
            fragment_ids,
            status: RepairStatus::Pending,
            differing_ranges,
        });

        task_id
    }

    /// 开始修复任务
    pub fn start_repair(&mut self, task_id: u64) -> Result<(), String> {
        let in_progress = self.repair_tasks.values()
            .filter(|t| t.status == RepairStatus::InProgress)
            .count();

        if in_progress >= self.max_concurrent_repairs {
            return Err("已达到最大并发修复数".to_string());
        }

        if let Some(task) = self.repair_tasks.get_mut(&task_id) {
            if task.status == RepairStatus::Pending {
                task.status = RepairStatus::InProgress;
                Ok(())
            } else {
                Err(format!("任务状态不是 Pending: {:?}", task.status))
            }
        } else {
            Err("任务不存在".to_string())
        }
    }

    /// 完成修复任务
    pub fn complete_repair(&mut self, task_id: u64) {
        if let Some(task) = self.repair_tasks.get_mut(&task_id) {
            task.status = RepairStatus::Completed;
        }
    }

    /// 标记修复任务失败
    pub fn fail_repair(&mut self, task_id: u64, reason: String) {
        if let Some(task) = self.repair_tasks.get_mut(&task_id) {
            task.status = RepairStatus::Failed(reason);
        }
    }

    /// 获取待处理修复任务
    pub fn pending_tasks(&self) -> Vec<&RepairTask> {
        self.repair_tasks.values()
            .filter(|t| t.status == RepairStatus::Pending)
            .collect()
    }

    /// 获取进行中修复任务
    pub fn in_progress_tasks(&self) -> Vec<&RepairTask> {
        self.repair_tasks.values()
            .filter(|t| t.status == RepairStatus::InProgress)
            .collect()
    }

    /// 生成同步动作
    pub fn compute_sync_actions(&self, diff: &DiffReport, remote_has_newer: bool) -> Vec<SyncAction> {
        if diff.identical {
            return vec![SyncAction::Skip];
        }

        diff.differing_ranges.iter().map(|(start, end)| {
            let ids: Vec<FragmentId> = self.local_digests.iter()
                .filter(|(id, _)| *id >= *start && *id <= *end)
                .map(|(id, _)| *id as FragmentId)
                .collect();

            if remote_has_newer {
                SyncAction::Request { fragment_ids: ids }
            } else {
                SyncAction::Send { fragment_ids: ids }
            }
        }).collect()
    }

    pub fn task_count(&self) -> usize {
        self.repair_tasks.len()
    }

    pub fn completed_task_count(&self) -> usize {
        self.repair_tasks.values().filter(|t| t.status == RepairStatus::Completed).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_identical() {
        let digests = vec![(1, 100), (2, 200), (3, 300)];
        let mut tree1 = MerkleTree::new(2);
        let mut tree2 = MerkleTree::new(2);
        tree1.build_from_digests(&digests);
        tree2.build_from_digests(&digests);
        let report = tree1.diff(&tree2);
        assert!(report.identical);
    }

    #[test]
    fn test_merkle_tree_different() {
        let digests1 = vec![(1, 100), (2, 200), (3, 300)];
        let digests2 = vec![(1, 100), (2, 999), (3, 300)];
        let mut tree1 = MerkleTree::new(2);
        let mut tree2 = MerkleTree::new(2);
        tree1.build_from_digests(&digests1);
        tree2.build_from_digests(&digests2);
        let report = tree1.diff(&tree2);
        assert!(!report.identical);
    }

    #[test]
    fn test_anti_entropy_create_task() {
        let mut ae = AntiEntropyRepair::new(10, 2);
        ae.update_digests(vec![(1, 100), (2, 200)]);
        let task_id = ae.create_repair_task(
            "node1".to_string(),
            "node2".to_string(),
            vec![(1, 2)],
        );
        assert_eq!(ae.task_count(), 1);
        ae.start_repair(task_id).unwrap();
        ae.complete_repair(task_id);
        assert_eq!(ae.completed_task_count(), 1);
    }

    #[test]
    fn test_max_concurrent_repairs() {
        let mut ae = AntiEntropyRepair::new(10, 1);
        ae.update_digests(vec![(1, 100), (2, 200)]);
        let t1 = ae.create_repair_task("n1".into(), "n2".into(), vec![(1, 1)]);
        let t2 = ae.create_repair_task("n2".into(), "n3".into(), vec![(2, 2)]);
        ae.start_repair(t1).unwrap();
        assert!(ae.start_repair(t2).is_err());
    }
}
