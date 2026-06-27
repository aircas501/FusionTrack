

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from models.loss.matrix_loss import euclidean_dist, cosine_dist, normalize, UnionFind


class CrossViewAssociator:

    def __init__(self, 
                 strategy: str = "hierarchical_clustering",
                 distance_metric: str = "euclidean",
                 threshold: float = 0.5,
                 normalize_features: bool = True,
                 enforce_one_to_one: bool = True,
                 neighbor_filter = None):
        self.strategy = strategy
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.normalize_features = normalize_features
        self.enforce_one_to_one = enforce_one_to_one
        self.neighbor_filter = neighbor_filter
        
    def associate(self, 
                  view_features: Dict[str, torch.Tensor],
                  view_ids: Dict[str, List[int]],
                  view_names: List[str],
                  view_bboxes: Optional[Dict[str, torch.Tensor]] = None) -> Dict[Tuple[str, int], int]:
        import time

        # 1. 构建全局特征矩阵和索引映射
        all_features, idx_to_view_id, view_idx_bounds = self._build_global_matrix(
            view_features, view_ids, view_names
        )
        
        if len(all_features) == 0:
            return {}
        
        # 2. 计算距离矩阵
        dist_matrix = self._compute_distance_matrix(all_features)
        #print("assosication 68" + str(time.perf_counter()))        
        # 3. 根据策略进行关联
        if self.strategy == "pairwise_hungarian":
            matches = self._pairwise_hungarian_matching(
                dist_matrix, view_idx_bounds, view_names,
                view_features, view_ids, view_bboxes  # 传递邻居筛选所需参数
            )
        elif self.strategy == "hierarchical_clustering":
            matches = self._hierarchical_clustering(
                dist_matrix, view_idx_bounds
            )
        elif self.strategy == "greedy_matching":
            matches = self._greedy_matching(
                dist_matrix, view_idx_bounds
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        #print("assosication 85" + str(time.perf_counter()))        
        # 4. 生成全局ID映射
        mapping = self._generate_global_mapping(
            matches, idx_to_view_id, view_names
        )
        
        return mapping
    
    def _build_global_matrix(self, 
                            view_features: Dict[str, torch.Tensor],
                            view_ids: Dict[str, List[int]],
                            view_names: List[str]) -> Tuple[torch.Tensor, Dict, List]:

        all_features = []
        idx_to_view_id = {}
        view_idx_bounds = []
        
        global_idx = 0
        for view_name in view_names:
            if view_name not in view_features or view_name not in view_ids:
                view_idx_bounds.append(0)
                continue
                
            features = view_features[view_name]  # (N_view, C)
            ids = view_ids[view_name]
            
            # 确保特征是2D
            if features.dim() == 1:
                features = features.unsqueeze(0)
            elif features.dim() > 2:
                features = features.squeeze()
                if features.dim() == 1:
                    features = features.unsqueeze(0)
            
            view_idx_bounds.append(len(features))
            
            for i, feat in enumerate(features):
                all_features.append(feat.cpu())
                idx_to_view_id[global_idx] = (view_name, ids[i])
                global_idx += 1
        
        if len(all_features) == 0:
            return torch.tensor([]), idx_to_view_id, view_idx_bounds
        
        all_features = torch.stack(all_features, dim=0)  # (N_total, C)
        
        # 归一化
        if self.normalize_features:
            all_features = normalize(all_features, axis=-1)
        
        return all_features, idx_to_view_id, view_idx_bounds
    
    def _compute_distance_matrix(self, features: torch.Tensor) -> np.ndarray:
        """
        计算距离矩阵
        
        Args:
            features: (N, C) 特征矩阵
            
        Returns:
            dist_matrix: (N, N) numpy数组
        """
        if self.distance_metric == "euclidean":
            dist_matrix = euclidean_dist(features, features)
        elif self.distance_metric == "cosine":
            dist_matrix = cosine_dist(features, features)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return dist_matrix.cpu().numpy()
    
    def _pairwise_hungarian_matching(self, 
                                     dist_matrix: np.ndarray,
                                     view_idx_bounds: List[int],
                                     view_names: List[str],
                                     view_features: Optional[Dict[str, torch.Tensor]] = None,
                                     view_ids: Optional[Dict[str, List[int]]] = None,
                                     view_bboxes: Optional[Dict[str, torch.Tensor]] = None) -> List[List[int]]:

        import time
        n_total = dist_matrix.shape[0]
        uf = UnionFind(n_total)
        
        # 计算视角起始索引
        view_start_indices = [0]
        for count in view_idx_bounds[:-1]:
            view_start_indices.append(view_start_indices[-1] + count)
        
        # 统计邻居筛选信息（用于调试）
        neighbor_filter_stats = {
            "total_pairs_before_filter": 0,
            "total_pairs_after_filter": 0,
            "total_rejected": 0,
            "rejection_details": []
        }
        
        # 两两视角间进行匈牙利匹配
        n_views = len(view_idx_bounds)
        for i in range(n_views):
            for j in range(i + 1, n_views):
                if view_idx_bounds[i] == 0 or view_idx_bounds[j] == 0:
                    continue
                
                # 提取两个视角间的距离子矩阵
                start_i = view_start_indices[i]
                end_i = start_i + view_idx_bounds[i]
                start_j = view_start_indices[j]
                end_j = start_j + view_idx_bounds[j]
                
                sub_matrix = dist_matrix[start_i:end_i, start_j:end_j].copy()
                
                # 超过阈值的设为无穷大
                sub_matrix[sub_matrix > self.threshold] = 1e9
                
                # 如果所有距离都超过阈值，跳过
                if np.all(sub_matrix >= 1e9):
                    continue
                
                # 匈牙利算法求解最优匹配
                #print("ph 215" + str(time.perf_counter()))

                row_ind, col_ind = linear_sum_assignment(sub_matrix)
                #print("ph 218" + str(time.perf_counter()))
                
                # ==================== 应用邻居筛选 ====================
                # 收集通过阈值和TopK约束的候选配对
                candidate_pairs = []
                for r, c in zip(row_ind, col_ind):
                    real_r = start_i + r
                    real_c = start_j + c
                    
                    if dist_matrix[real_r, real_c] <= self.threshold:
                        # 转换为视角内的局部索引（相对于每个视角的起始）
                        candidate_pairs.append((r, c))
                
                neighbor_filter_stats["total_pairs_before_filter"] += len(candidate_pairs)
                
                # 应用邻居筛选
                if self.neighbor_filter is not None and view_features is not None and view_bboxes is not None:
                    # 准备当前两个视角的数据
                    view1_name = view_names[i]
                    view2_name = view_names[j]
                    
                    if view1_name in view_features and view2_name in view_features:
                        view1_features = view_features[view1_name]
                        view2_features = view_features[view2_name]
                        view1_ids = view_ids[view1_name] if view_ids else list(range(len(view1_features)))
                        view2_ids = view_ids[view2_name] if view_ids else list(range(len(view2_features)))
                        view1_bboxes = view_bboxes.get(view1_name, None)
                        view2_bboxes = view_bboxes.get(view2_name, None)
                        
                        # 调用邻居筛选
                        filtered_pairs, rejection_reasons = self.neighbor_filter.filter_association(
                            candidate_pairs=candidate_pairs,
                            view1_features=view1_features,
                            view2_features=view2_features,
                            view1_ids=view1_ids,
                            view2_ids=view2_ids,
                            view1_bboxes=view1_bboxes,
                            view2_bboxes=view2_bboxes
                        )
                        
                        # 更新统计信息
                        neighbor_filter_stats["total_pairs_after_filter"] += len(filtered_pairs)
                        neighbor_filter_stats["total_rejected"] += len(rejection_reasons)
                        for pair, reason in rejection_reasons.items():
                            neighbor_filter_stats["rejection_details"].append({
                                "view1": view1_name,
                                "view2": view2_name,
                                "pair": pair,
                                "reason": reason
                            })
                        
                        # 更新候选配对（只保留通过筛选的）
                        candidate_pairs = filtered_pairs
                
                neighbor_filter_stats["total_pairs_after_filter"] += len(candidate_pairs)
                
                # ==================== 应用一对一约束和合并 ====================
                # 将匹配结果加入并查集
                for r, c in candidate_pairs:
                    real_r = start_i + r
                    real_c = start_j + c
                    
                    # 检查一对一约束
                    if self.enforce_one_to_one:
                        # 确保real_r和real_c都还没有被分配
                        root_r = uf.find(real_r)
                        root_c = uf.find(real_c)
                        
                        # 检查合并后是否会违反一对一约束
                        if root_r != root_c:
                            # 获取两个组的所有成员
                            group_r = [k for k in range(n_total) if uf.find(k) == root_r]
                            group_c = [k for k in range(n_total) if uf.find(k) == root_c]
                            
                            # 检查合并后每个视角最多只有一个目标
                            merged_group = group_r + group_c
                            view_counts = self._count_targets_per_view(
                                merged_group, view_start_indices, view_idx_bounds
                            )
                            
                            # 如果任何视角有多于1个目标，跳过这次合并
                            if all(count <= 1 for count in view_counts.values()):
                                uf.union(real_r, real_c)
                    else:
                        uf.union(real_r, real_c)
                #print("ph 250" + str(time.perf_counter()))
        #print("ph 251" + str(time.perf_counter()))
        
        # 打印邻居筛选统计信息
        if self.neighbor_filter is not None:
            print(f"\n{'='*80}")
            print(f"[NeighborFilter] 邻居筛选统计:")
            print(f"  - 筛选前候选配对数: {neighbor_filter_stats['total_pairs_before_filter']}")
            print(f"  - 筛选后保留配对数: {neighbor_filter_stats['total_pairs_after_filter']}")
            print(f"  - 被拒绝的配对数: {neighbor_filter_stats['total_rejected']}")
            if neighbor_filter_stats['total_pairs_before_filter'] > 0:
                rejection_rate = neighbor_filter_stats['total_rejected'] / neighbor_filter_stats['total_pairs_before_filter'] * 100
                print(f"  - 拒绝率: {rejection_rate:.2f}%")
            print(f"{'='*80}\n")
        
        # 从并查集提取匹配组
        groups = uf.get_groups()
        matches = [sorted(group) for group in groups.values()]
        #print("ph 255" + str(time.perf_counter()))

        return matches
    
    def _count_targets_per_view(self, 
                               indices: List[int],
                               view_start_indices: List[int],
                               view_idx_bounds: List[int]) -> Dict[int, int]:
        view_counts = {}
        
        for idx in indices:
            # 找到idx属于哪个视角
            view_idx = self._get_view_index(idx, view_start_indices, view_idx_bounds)
            if view_idx != -1:
                view_counts[view_idx] = view_counts.get(view_idx, 0) + 1
        
        return view_counts
    
    def _get_view_index(self, 
                       global_idx: int,
                       view_start_indices: List[int],
                       view_idx_bounds: List[int]) -> int:

        for i, (start, count) in enumerate(zip(view_start_indices, view_idx_bounds)):
            if start <= global_idx < start + count:
                return i
        return -1
    
    def _hierarchical_clustering(self,
                                dist_matrix: np.ndarray,
                                view_idx_bounds: List[int]) -> List[List[int]]:

        n_total = dist_matrix.shape[0]
        
        if n_total == 0:
            return []
        
        # 计算视角起始索引
        view_start_indices = [0]
        for count in view_idx_bounds[:-1]:
            view_start_indices.append(view_start_indices[-1] + count)
        
        # ⭐ 优化1：预计算每个索引的视角映射（避免重复调用_get_view_index）
        idx_to_view = {}
        for idx in range(n_total):
            view_idx = self._get_view_index(idx, view_start_indices, view_idx_bounds)
            idx_to_view[idx] = view_idx
        
        # ⭐ 优化2：预计算每个视角的所有目标索引（加速查找）
        view_to_indices = {}
        for view_idx in range(len(view_idx_bounds)):
            start = view_start_indices[view_idx]
            end = start + view_idx_bounds[view_idx]
            view_to_indices[view_idx] = set(range(start, end))
        
        # 初始化动态距离矩阵（会在聚类过程中更新）
        dynamic_dist = dist_matrix.copy()
        
        # ==================== 规则1：同视角内距离设为无穷大 ====================
        for i, (start, count) in enumerate(zip(view_start_indices, view_idx_bounds)):
            end = start + count
            dynamic_dist[start:end, start:end] = np.inf
        
        # 对角线设为无穷大（自己与自己）
        np.fill_diagonal(dynamic_dist, np.inf)
        
        # 初始化：每个目标是一个独立的簇
        clusters = {i: [i] for i in range(n_total)}
        active_clusters = set(range(n_total))
        
        # 记录每个全局索引属于哪个簇
        idx_to_cluster = {i: i for i in range(n_total)}
        
        # 记录每个簇占据了哪些视角（用于一对一约束）
        cluster_views = {idx: {idx_to_view[idx]} for idx in range(n_total)}
        
        # ⭐ 优化3：缓存簇间距离（避免重复计算）
        cluster_dist_cache = {}
        
        def get_cluster_distance(ci, cj):
            """获取两个簇之间的最小距离（Single Linkage）"""
            key = (min(ci, cj), max(ci, cj))
            if key in cluster_dist_cache:
                return cluster_dist_cache[key]
            
            min_d = np.inf
            for idx_i in clusters[ci]:
                for idx_j in clusters[cj]:
                    d = dynamic_dist[idx_i, idx_j]
                    if d < min_d:
                        min_d = d
            
            cluster_dist_cache[key] = min_d
            return min_d
        
        # ==================== 迭代聚类过程 ====================
        iteration = 0
        max_iterations = n_total * 2  # 防止无限循环
        
        while iteration < max_iterations:
            iteration += 1
            
            # 在活跃簇之间找到最小距离
            min_dist = np.inf
            best_pair = None
            
            # ⭐ 优化4：只遍历活跃簇对
            active_list = list(active_clusters)
            for i, ci in enumerate(active_list):
                for cj in active_list[i+1:]:
                    cluster_dist = get_cluster_distance(ci, cj)
                    
                    if cluster_dist < min_dist:
                        min_dist = cluster_dist
                        best_pair = (ci, cj)
            
            # 终止条件：没有可合并的簇对
            if min_dist > self.threshold or np.isinf(min_dist) or best_pair is None:
                break
            
            ci, cj = best_pair
            
            # ==================== 检查一对一约束 ====================
            merged_views = cluster_views[ci] | cluster_views[cj]
            
            # 如果有视角重复，则不能合并（违反一对一约束）
            if len(merged_views) < len(cluster_views[ci]) + len(cluster_views[cj]):
                # 违反一对一约束，屏蔽这对簇之间的所有距离
                for idx_i in clusters[ci]:
                    for idx_j in clusters[cj]:
                        dynamic_dist[idx_i, idx_j] = np.inf
                        dynamic_dist[idx_j, idx_i] = np.inf
                
                # ⭐ 清除相关缓存
                key = (min(ci, cj), max(ci, cj))
                cluster_dist_cache[key] = np.inf
                continue
            
            # ==================== 执行合并 ====================
            # 合并簇cj到簇ci
            clusters[ci].extend(clusters[cj])
            cluster_views[ci] = merged_views
            
            # 更新idx_to_cluster映射
            for idx in clusters[cj]:
                idx_to_cluster[idx] = ci
            
            # ⭐ 清除与cj相关的所有缓存
            keys_to_remove = [k for k in cluster_dist_cache.keys() if ci in k or cj in k]
            for k in keys_to_remove:
                del cluster_dist_cache[k]
            
            # 移除簇cj
            del clusters[cj]
            del cluster_views[cj]
            active_clusters.remove(cj)
            
            # ==================== 规则2：更新距离矩阵 ====================
            # ⭐ 优化5：使用预计算的映射，避免重复调用_get_view_index
            
            # 对于ci中的每个目标
            for idx_i in clusters[ci]:
                view_i = idx_to_view[idx_i]
                
                # 对于已关联的每个视角
                for assoc_view in cluster_views[ci]:
                    if assoc_view == view_i:
                        continue
                    
                    # ⭐ 使用预计算的视角索引集合
                    for idx_other in view_to_indices[assoc_view]:
                        # 如果idx_other不在同一个簇中，屏蔽距离
                        if idx_to_cluster[idx_other] != ci:
                            dynamic_dist[idx_i, idx_other] = np.inf
                            dynamic_dist[idx_other, idx_i] = np.inf
        
        # ==================== 提取最终匹配结果 ====================
        matches = [sorted(cluster) for cluster in clusters.values()]
        
        return matches
    
    
    def _can_merge_clusters(self,
                           cluster1: List[int],
                           cluster2: List[int],
                           view_start_indices: List[int],
                           view_idx_bounds: List[int]) -> bool:

        merged = cluster1 + cluster2
        
        # 检查大小
        if len(merged) > len(view_idx_bounds):
            return False
        
        # 检查每个视角最多一个目标
        view_counts = self._count_targets_per_view(
            merged, view_start_indices, view_idx_bounds
        )
        
        return all(count <= 1 for count in view_counts.values())
    
    def _greedy_matching(self,
                        dist_matrix: np.ndarray,
                        view_idx_bounds: List[int]) -> List[List[int]]:

        n_total = dist_matrix.shape[0]
        uf = UnionFind(n_total)
        
        # 计算视角起始索引
        view_start_indices = [0]
        for count in view_idx_bounds[:-1]:
            view_start_indices.append(view_start_indices[-1] + count)
        
        # 获取所有跨视角的距离对 (dist, i, j)
        pairs = []
        for i in range(n_total):
            view_i = self._get_view_index(i, view_start_indices, view_idx_bounds)
            for j in range(i + 1, n_total):
                view_j = self._get_view_index(j, view_start_indices, view_idx_bounds)
                
                # 只考虑跨视角的对
                if view_i != view_j and dist_matrix[i, j] <= self.threshold:
                    pairs.append((dist_matrix[i, j], i, j))
        
        # 按距离排序
        pairs.sort(key=lambda x: x[0])
        
        # 贪心匹配
        for dist, i, j in pairs:
            root_i = uf.find(i)
            root_j = uf.find(j)
            
            if root_i != root_j:
                # 检查合并约束
                group_i = [k for k in range(n_total) if uf.find(k) == root_i]
                group_j = [k for k in range(n_total) if uf.find(k) == root_j]
                
                if self._can_merge_clusters(
                    group_i, group_j,
                    view_start_indices, view_idx_bounds
                ):
                    uf.union(i, j)
        
        # 提取匹配组
        groups = uf.get_groups()
        matches = [sorted(group) for group in groups.values()]
        
        return matches
    
    def _generate_global_mapping(self,
                                matches: List[List[int]],
                                idx_to_view_id: Dict[int, Tuple[str, int]],
                                view_names: List[str]) -> Dict[Tuple[str, int], int]:
        """
        生成全局ID映射（修改版：继承最小ID）
        """
        mapping = {}
        
        # 遍历每一组匹配（例如：[idx_of_View1_ID6, idx_of_View2_ID18]）
        for match_group in matches:
            # 1. 先把这一组里所有的原始信息提取出来
            group_members = []
            original_ids = []
            
            for idx in match_group:
                if idx in idx_to_view_id:
                    view_name, local_id = idx_to_view_id[idx]
                    group_members.append((view_name, local_id))
                    original_ids.append(local_id)
            
            if not group_members:
                continue
                
            # 2. 核心策略：选出“老大”
            # 策略A：最小ID策略 (推荐，最稳健)
            # 解释：既然这一组都是同一个人，那我就用其中最小的那个ID作为大家的统称。
            # 比如 {6, 18} -> 选 6。这样 View1 的 ID 6 保持不变，View2 的 ID 18 变成了 6。
            global_id = min(original_ids)
            
            # 策略B（可选）：主视角优先策略
            # 如果你希望以 c001 为主，如果组里有 c001 的 ID，就强制用它的。
            # c001_ids = [mid for v, mid in group_members if v == 'c001']
            # if c001_ids:
            #     global_id = c001_ids[0]
            # else:
            #     global_id = min(original_ids)
            
            # 3. 建立映射
            for view_name, local_id in group_members:
                mapping[(view_name, local_id)] = global_id
        
        return mapping


def build_cross_view_associator(config: dict, neighbor_filter=None) -> CrossViewAssociator:

    return CrossViewAssociator(
        strategy=config.get("CROSS_VIEW_STRATEGY", "hierarchical_clustering"),
        distance_metric=config.get("CROSS_VIEW_DISTANCE", "euclidean"),
        threshold=config.get("CROSS_VIEW_THRESHOLD", 0.5),
        normalize_features=config.get("CROSS_VIEW_NORMALIZE", True),
        enforce_one_to_one=config.get("CROSS_VIEW_ONE_TO_ONE", True),
        neighbor_filter=neighbor_filter
    )

