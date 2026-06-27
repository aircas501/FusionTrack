

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from models.cross_view_association import CrossViewAssociator


class FrameLevelCrossViewAssociator:

    
    def __init__(self,
                 base_associator: CrossViewAssociator,
                 voting_strategy: str = "sliding_window",
                 window_size: int = 10,
                 min_votes: int = 5,
                 consistency_weight: float = 0.3):

        self.base_associator = base_associator
        self.voting_strategy = voting_strategy
        self.window_size = window_size
        self.min_votes = min_votes
        self.consistency_weight = consistency_weight
        
        # 历史记录 [(frame_id, associations, confidence_scores), ...]
        self.history = []
        self.prev_associations = None
        
    def associate_frame(self,
                       frame_id: int,
                       view_detections: Dict[str, List[Tuple[int, torch.Tensor]]],
                       view_names: List[str],
                       view_bboxes: Dict[str, torch.Tensor] = None) -> Dict[Tuple[str, int], int]:

        # 1. 提取特征和ID
        view_features = {}
        view_ids = {}
        
        for view_name in view_names:
            if view_name not in view_detections or len(view_detections[view_name]) == 0:
                continue
            
            detections = view_detections[view_name]
            
            # 分离track_id和feature
            track_ids = [det[0] for det in detections]
            features = torch.stack([det[1] for det in detections])  # (N, C)
            
            view_features[view_name] = features
            view_ids[view_name] = track_ids
        
        # 2. 使用基础关联器获取当前帧的关联（传递bboxes）
        current_associations = self.base_associator.associate(
            view_features, view_ids, view_names, view_bboxes
        )
        
        # 3. 根据策略决定最终关联
        if self.voting_strategy == "none":
            final_associations = current_associations
        elif self.voting_strategy == "sliding_window":
            final_associations = self._sliding_window_vote(
                frame_id, current_associations, view_features
            )
        elif self.voting_strategy == "weighted":
            final_associations = self._weighted_vote(
                frame_id, current_associations, view_features
            )
        elif self.voting_strategy == "temporal_consistency":
            final_associations = self._temporal_consistency(
                frame_id, current_associations, view_features
            )
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        # 4. 更新历史
        self.prev_associations = final_associations
        
        return final_associations
    
    def _sliding_window_vote(self,
                             frame_id: int,
                             current_associations:  Dict,
                             view_features: Dict) -> Dict:
        """
        滑动窗口投票
        
        在窗口内统计每对关联出现的次数，选择出现次数最多的
        """
        # 添加到历史
        self.history.append((frame_id, current_associations, view_features))
        
        # 保持窗口大小
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # 如果历史不够，直接返回当前关联
        if len(self.history) < self.min_votes:
            return current_associations
        
        # 统计每对关联出现的次数
        pair_votes = defaultdict(int)
        
        for _, assoc, _ in self.history:
            pairs = self._extract_pairs(assoc)
            for pair in pairs:
                pair_votes[pair] += 1
        
        # 根据投票重建关联
        final_associations = self._rebuild_from_votes(pair_votes)
        
        return final_associations
    
    def _weighted_vote(self,
                      frame_id: int,
                      current_associations: Dict,
                      view_features: Dict) -> Dict:
        """
        加权投票
        
        使用特征相似度作为权重
        """
        # 添加到历史
        self.history.append((frame_id, current_associations, view_features))
        
        # 保持窗口大小
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        if len(self.history) < self.min_votes:
            return current_associations
        
        # 计算加权投票
        pair_scores = defaultdict(float)
        
        for fid, assoc, features in self.history:
            pairs = self._extract_pairs(assoc)
            
            for pair in pairs:
                # 计算这对的特征相似度作为权重
                weight = self._compute_pair_similarity(pair, features)
                pair_scores[pair] += weight
        
        # 根据得分重建关联（阈值为平均得分）
        if len(pair_scores) > 0:
            threshold = np.mean(list(pair_scores.values()))
            valid_pairs = [pair for pair, score in pair_scores.items() 
                          if score >= threshold]
            
            if len(valid_pairs) > 0:
                return self._rebuild_from_votes(
                    {pair: 1 for pair in valid_pairs}
                )
        
        return current_associations
    
    def _temporal_consistency(self,
                             frame_id: int,
                             current_associations: Dict,
                             view_features: Dict) -> Dict:
        """
        时序一致性约束
        
        优先保持与前一帧的关联一致
        """
        if self.prev_associations is None:
            return current_associations
        
        # 对于与前一帧一致的关联，优先保留
        consistent_associations = {}
        
        for key, global_id in current_associations.items():
            if key in self.prev_associations:
                prev_global_id = self.prev_associations[key]
                
                # 检查是否一致
                if global_id == prev_global_id:
                    # 一致，直接保留
                    consistent_associations[key] = global_id
                else:
                    # 不一致，根据权重决定
                    # 随机数 < consistency_weight 时保持一致
                    if np.random.random() < self.consistency_weight:
                        consistent_associations[key] = prev_global_id
                    else:
                        consistent_associations[key] = global_id
            else:
                # 新出现的目标，使用当前关联
                consistent_associations[key] = global_id
        
        return consistent_associations
    
    def _extract_pairs(self, associations: Dict) -> List[Tuple]:
        """
        从关联中提取配对
        
        Example:
            associations = {
                ('view1', 0): 0,
                ('view2', 3): 0,
                ('view1', 1): 1,
                ('view2', 5): 1,
            }
            
            pairs = [
                (('view1', 0), ('view2', 3)),  # global_id=0
                (('view1', 1), ('view2', 5)),  # global_id=1
            ]
        """
        # 按global_id分组
        global_groups = defaultdict(list)
        for key, global_id in associations.items():
            global_groups[global_id].append(key)
        
        # 提取配对（每个组内两两配对）
        pairs = []
        for global_id, group in global_groups.items():
            if len(group) < 2:
                continue
            
            # 两两配对
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # 排序以保证一致性
                    pair = tuple(sorted([group[i], group[j]]))
                    pairs.append(pair)
        
        return pairs
    
    def _compute_pair_similarity(self, 
                                 pair: Tuple, 
                                 view_features: Dict) -> float:
        """
        计算配对的特征相似度
        
        Args:
            pair: ((view1, tid1), (view2, tid2))
            view_features: {view_name: tensor}
            
        Returns:
            similarity: [0, 1]
        """
        key1, key2 = pair
        view1, tid1 = key1
        view2, tid2 = key2
        
        if view1 not in view_features or view2 not in view_features:
            return 1.0  # 默认权重
        
        try:
            # 获取特征
            feats1 = view_features[view1]  # (N1, C)
            feats2 = view_features[view2]  # (N2, C)
            
            # 找到对应的特征
            # 注意：这里tid是在detections列表中的索引
            if tid1 >= len(feats1) or tid2 >= len(feats2):
                return 1.0
            
            feat1 = feats1[tid1]  # (C,)
            feat2 = feats2[tid2]  # (C,)
            
            # 计算余弦相似度
            feat1_np = feat1.cpu().numpy()
            feat2_np = feat2.cpu().numpy()
            
            similarity = np.dot(feat1_np, feat2_np) / (
                np.linalg.norm(feat1_np) * np.linalg.norm(feat2_np) + 1e-6
            )
            
            return max(0, similarity)
            
        except Exception as e:
            # 出错时返回默认权重
            return 1.0
    
    def _rebuild_from_votes(self, pair_votes: Dict) -> Dict:

        # 1. 过滤低票配对
        threshold = self.min_votes
        valid_pairs = [pair for pair, votes in pair_votes.items() 
                      if votes >= threshold]
        
        # 2. 处理空数据情况
        if len(valid_pairs) == 0:
            # 没有足够的投票，返回最新的关联
            if len(self.history) > 0:
                return self.history[-1][1]
            else:
                return {}
        
        # 3. 准备并查集
        from models.loss.matrix_loss import UnionFind
        
        # 收集所有的key
        all_keys = set()
        for pair in valid_pairs:
            all_keys.add(pair[0])
            all_keys.add(pair[1])
        
        # 创建key到索引的映射 (用于并查集内部逻辑)
        key_to_idx = {key: idx for idx, key in enumerate(all_keys)}
        idx_to_key = {idx: key for key, idx in key_to_idx.items()}
        
        # 初始化并查集
        uf = UnionFind(len(all_keys))
        
        # 4. 执行合并操作
        for pair in valid_pairs:
            idx1 = key_to_idx[pair[0]]
            idx2 = key_to_idx[pair[1]]
            uf.union(idx1, idx2)
        
        # 5. 生成关联 (核心修改部分)
        groups = uf.get_groups()
        associations = {}
        
        # 遍历每一个并查集生成的组
        for group_indices in groups.values():
            # 将索引转换回实际的 keys 列表
            # 例如 cluster_keys = [('c001', 10), ('c002', 17), ('c004', 52)]
            cluster_keys = [idx_to_key[idx] for idx in group_indices]
            
            # --- 关键修改开始 ---
            
            # 找到该组中“最小”的 key。
            # Python tuple 比较机制：先比 cam_id (string)，再比 track_id (int)。
            # 这符合“最先出现的ID”逻辑，c001 会比 c002 小。
            leader_key = min(cluster_keys)
            
            # 直接提取 leader_key 中的原始 ID 作为 Global ID
            # 假设 key 的结构是 (camera_id, track_id)，取下标 1
            # 这样 ('c001', 10) 所在的组，Global ID 就固定为 10
            target_global_id = leader_key[1]
            
            # 将该组内所有轨迹统一映射到这个 ID
            for key in cluster_keys:
                associations[key] = target_global_id
                
            # --- 关键修改结束 ---
        
        return associations
    
    def reset(self):
        """重置历史记录"""
        self.history = []
        self.prev_associations = None


def build_frame_level_associator(config: dict, neighbor_filter=None) -> FrameLevelCrossViewAssociator:

    from models.cross_view_association import build_cross_view_associator
    
    # 创建基础关联器（传递邻居筛选器）
    base_associator = build_cross_view_associator(config, neighbor_filter=neighbor_filter)
    
    # 创建帧级关联器
    return FrameLevelCrossViewAssociator(
        base_associator=base_associator,
        voting_strategy=config.get("VOTING_STRATEGY", "sliding_window"),
        window_size=config.get("WINDOW_SIZE", 10),
        min_votes=config.get("MIN_VOTES", 5),
        consistency_weight=config.get("CONSISTENCY_WEIGHT", 0.3)
    )

