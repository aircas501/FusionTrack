
import torch
import numpy as np


class NeighborFilter:

    
    def __init__(
        self,
        k_top_similarities=5,
        min_neighbor_match_ratio=0.5,
        neighbor_sim_threshold=0.6,
        distance_metric="cosine",
        spatial_radius=100,  # 像素
    ):

        self.k_top_similarities = k_top_similarities
        self.min_neighbor_match_ratio = min_neighbor_match_ratio
        self.neighbor_sim_threshold = neighbor_sim_threshold
        self.distance_metric = distance_metric
        self.spatial_radius = spatial_radius
    
    def filter_association(
        self,
        candidate_pairs,
        view1_features,
        view2_features,
        view1_ids,
        view2_ids,
        view1_bboxes=None,
        view2_bboxes=None,
    ):

        if len(candidate_pairs) == 0:
            return [], {}
        
        # 归一化特征
        view1_features_norm = torch.nn.functional.normalize(view1_features, p=2, dim=1)
        view2_features_norm = torch.nn.functional.normalize(view2_features, p=2, dim=1)
        
        filtered_pairs = []
        rejection_reasons = {}
        
        for idx1, idx2 in candidate_pairs:
            # 1. 找到A和B的空间邻域内的所有目标
            if view1_bboxes is None or view2_bboxes is None:
                # 如果没有bbox信息，接受该匹配
                filtered_pairs.append((idx1, idx2))
                continue
            
            # 获取空间邻域内的所有目标索引
            neighbors_a_indices = self._find_spatial_neighbors(
                idx1, view1_bboxes, exclude_self=True
            )
            neighbors_b_indices = self._find_spatial_neighbors(
                idx2, view2_bboxes, exclude_self=True
            )
            
            # 如果邻居数量不足，跳过筛选（接受该匹配）
            if len(neighbors_a_indices) == 0 or len(neighbors_b_indices) == 0:
                filtered_pairs.append((idx1, idx2))
                continue
            
            # 2. 提取邻居特征
            neighbor_features_a = view1_features_norm[neighbors_a_indices]  # (Na, D)
            neighbor_features_b = view2_features_norm[neighbors_b_indices]  # (Nb, D)
            
            # 3. 计算邻居间的相似度矩阵
            if self.distance_metric == "cosine":
                # 余弦相似度 (Na, Nb)
                neighbor_sim_matrix = torch.mm(neighbor_features_a, neighbor_features_b.T)
            elif self.distance_metric == "euclidean":
                # 欧氏距离 -> 相似度
                dist = torch.cdist(neighbor_features_a, neighbor_features_b, p=2)
                neighbor_sim_matrix = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
            # 4. 对于每个A的邻居，找出相似度最高的前K个B的邻居
            # neighbor_sim_matrix: (Na, Nb)
            num_a_neighbors = neighbor_sim_matrix.shape[0]
            num_b_neighbors = neighbor_sim_matrix.shape[1]
            
            # 对于每一行（A的每个邻居），找出最高的前K个相似度
            k_actual = min(self.k_top_similarities, num_b_neighbors)
            top_k_sims_per_a, _ = torch.topk(neighbor_sim_matrix, k=k_actual, dim=1)  # (Na, K)
            
            # 5. 统计高相似度匹配的比例
            # 计算所有top-K相似度中超过阈值的比例
            high_sim_count = (top_k_sims_per_a > self.neighbor_sim_threshold).sum().item()
            total_count = num_a_neighbors * k_actual
            match_ratio = high_sim_count / total_count if total_count > 0 else 0.0
            
            # 6. 决策
            if match_ratio >= self.min_neighbor_match_ratio:
                filtered_pairs.append((idx1, idx2))
            else:
                rejection_reasons[(idx1, idx2)] = (
                    f"Neighbor match ratio too low: {match_ratio:.3f} < {self.min_neighbor_match_ratio:.3f} "
                    f"(Na={num_a_neighbors}, Nb={num_b_neighbors}, top-{k_actual})"
                )
        
        return filtered_pairs, rejection_reasons
    
    def _find_spatial_neighbors(self, target_idx, bboxes, exclude_self=True):

        # 计算目标中心点
        target_bbox = bboxes[target_idx]
        # 假设bbox格式可能是[x, y, w, h]或[cx, cy, w, h]
        # 统一转换为中心点坐标
        if target_bbox[2] > 0 and target_bbox[3] > 0:  # 有宽高信息
            target_center = target_bbox[:2] + target_bbox[2:] / 2.0  # (cx, cy)
        else:
            target_center = target_bbox[:2]  # 已经是中心点
        
        # 计算所有目标的中心点
        all_centers = bboxes[:, :2] + bboxes[:, 2:] / 2.0  # (N, 2)
        
        # 计算到目标中心的距离
        distances = torch.norm(all_centers - target_center, p=2, dim=1)  # (N,)
        
        # 找到空间邻域内的目标
        mask = (distances <= self.spatial_radius)
        
        # 排除自身
        if exclude_self:
            mask[target_idx] = False
        
        # 返回邻居索引
        neighbor_indices = torch.where(mask)[0].cpu().tolist()
        
        return neighbor_indices
    
    def get_statistics(self, rejection_reasons):

        stats = {
            "total_rejected": len(rejection_reasons),
            "rejection_reasons": {}
        }
        
        # 统计拒绝原因
        for reason in rejection_reasons.values():
            if reason not in stats["rejection_reasons"]:
                stats["rejection_reasons"][reason] = 0
            stats["rejection_reasons"][reason] += 1
        
        return stats


def build_neighbor_filter(config):

    if not config.get("USE_NEIGHBOR_FILTER", False):
        return None
    
    return NeighborFilter(
        k_top_similarities=config.get("NEIGHBOR_K_TOP", 5),
        min_neighbor_match_ratio=config.get("NEIGHBOR_MIN_MATCH_RATIO", 0.5),
        neighbor_sim_threshold=config.get("NEIGHBOR_SIM_THRESHOLD", 0.6),
        distance_metric=config.get("NEIGHBOR_DISTANCE_METRIC", "cosine"),
        spatial_radius=config.get("NEIGHBOR_SPATIAL_RADIUS", 100),
    )
