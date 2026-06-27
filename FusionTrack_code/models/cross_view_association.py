

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

        # 1. Build global feature matrix and index mapping
        all_features, idx_to_view_id, view_idx_bounds = self._build_global_matrix(
            view_features, view_ids, view_names
        )
        
        if len(all_features) == 0:
            return {}
        
        # 2. Compute distance matrix
        dist_matrix = self._compute_distance_matrix(all_features)
        #print("assosication 68" + str(time.perf_counter()))        
        # 3. Associate per strategy
        if self.strategy == "pairwise_hungarian":
            matches = self._pairwise_hungarian_matching(
                dist_matrix, view_idx_bounds, view_names,
                view_features, view_ids, view_bboxes  # params for neighbor filter
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
        # 4. Build global ID mapping
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
            
            # ensure features are 2D
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
        
        # normalize
        if self.normalize_features:
            all_features = normalize(all_features, axis=-1)
        
        return all_features, idx_to_view_id, view_idx_bounds
    
    def _compute_distance_matrix(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute distance matrix.
        
        Args:
            features: (N, C) feature matrix
            
        Returns:
            dist_matrix: (N, N) numpy array
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
        
        # compute per-view start indices
        view_start_indices = [0]
        for count in view_idx_bounds[:-1]:
            view_start_indices.append(view_start_indices[-1] + count)
        
        # neighbor-filter stats (debug)
        neighbor_filter_stats = {
            "total_pairs_before_filter": 0,
            "total_pairs_after_filter": 0,
            "total_rejected": 0,
            "rejection_details": []
        }
        
        # Hungarian matching between view pairs
        n_views = len(view_idx_bounds)
        for i in range(n_views):
            for j in range(i + 1, n_views):
                if view_idx_bounds[i] == 0 or view_idx_bounds[j] == 0:
                    continue
                
                # extract sub-matrix between two views
                start_i = view_start_indices[i]
                end_i = start_i + view_idx_bounds[i]
                start_j = view_start_indices[j]
                end_j = start_j + view_idx_bounds[j]
                
                sub_matrix = dist_matrix[start_i:end_i, start_j:end_j].copy()
                
                # set distances above threshold to inf
                sub_matrix[sub_matrix > self.threshold] = 1e9
                
                # skip if all distances exceed threshold
                if np.all(sub_matrix >= 1e9):
                    continue
                
                # Hungarian algorithm for optimal matching
                #print("ph 215" + str(time.perf_counter()))

                row_ind, col_ind = linear_sum_assignment(sub_matrix)
                #print("ph 218" + str(time.perf_counter()))
                
                # ==================== Apply neighbor filter ====================
                # collect candidate pairs passing threshold and TopK
                candidate_pairs = []
                for r, c in zip(row_ind, col_ind):
                    real_r = start_i + r
                    real_c = start_j + c
                    
                    if dist_matrix[real_r, real_c] <= self.threshold:
                        # convert to view-local indices
                        candidate_pairs.append((r, c))
                
                neighbor_filter_stats["total_pairs_before_filter"] += len(candidate_pairs)
                
                # apply neighbor filter
                if self.neighbor_filter is not None and view_features is not None and view_bboxes is not None:
                    # prepare data for the two views
                    view1_name = view_names[i]
                    view2_name = view_names[j]
                    
                    if view1_name in view_features and view2_name in view_features:
                        view1_features = view_features[view1_name]
                        view2_features = view_features[view2_name]
                        view1_ids = view_ids[view1_name] if view_ids else list(range(len(view1_features)))
                        view2_ids = view_ids[view2_name] if view_ids else list(range(len(view2_features)))
                        view1_bboxes = view_bboxes.get(view1_name, None)
                        view2_bboxes = view_bboxes.get(view2_name, None)
                        
                        # invoke neighbor filter
                        filtered_pairs, rejection_reasons = self.neighbor_filter.filter_association(
                            candidate_pairs=candidate_pairs,
                            view1_features=view1_features,
                            view2_features=view2_features,
                            view1_ids=view1_ids,
                            view2_ids=view2_ids,
                            view1_bboxes=view1_bboxes,
                            view2_bboxes=view2_bboxes
                        )
                        
                        # update stats
                        neighbor_filter_stats["total_pairs_after_filter"] += len(filtered_pairs)
                        neighbor_filter_stats["total_rejected"] += len(rejection_reasons)
                        for pair, reason in rejection_reasons.items():
                            neighbor_filter_stats["rejection_details"].append({
                                "view1": view1_name,
                                "view2": view2_name,
                                "pair": pair,
                                "reason": reason
                            })
                        
                        # keep filtered candidate pairs only
                        candidate_pairs = filtered_pairs
                
                neighbor_filter_stats["total_pairs_after_filter"] += len(candidate_pairs)
                
                # ==================== One-to-one constraint and merge ====================
                # add matches to union-find
                for r, c in candidate_pairs:
                    real_r = start_i + r
                    real_c = start_j + c
                    
                    # check one-to-one constraint
                    if self.enforce_one_to_one:
                        # ensure real_r and real_c not yet assigned
                        root_r = uf.find(real_r)
                        root_c = uf.find(real_c)
                        
                        # check one-to-one after merge
                        if root_r != root_c:
                            # members of both groups
                            group_r = [k for k in range(n_total) if uf.find(k) == root_r]
                            group_c = [k for k in range(n_total) if uf.find(k) == root_c]
                            
                            # at most one target per view after merge
                            merged_group = group_r + group_c
                            view_counts = self._count_targets_per_view(
                                merged_group, view_start_indices, view_idx_bounds
                            )
                            
                            # skip merge if any view would have >1 target
                            if all(count <= 1 for count in view_counts.values()):
                                uf.union(real_r, real_c)
                    else:
                        uf.union(real_r, real_c)
                #print("ph 250" + str(time.perf_counter()))
        #print("ph 251" + str(time.perf_counter()))
        
        # print neighbor-filter stats
        if self.neighbor_filter is not None:
            print(f"\n{'='*80}")
            print(f"[NeighborFilter] Neighbor filter statistics:")
            print(f"  - Candidate pairs before filter: {neighbor_filter_stats['total_pairs_before_filter']}")
            print(f"  - Pairs kept after filter: {neighbor_filter_stats['total_pairs_after_filter']}")
            print(f"  - Rejected pairs: {neighbor_filter_stats['total_rejected']}")
            if neighbor_filter_stats['total_pairs_before_filter'] > 0:
                rejection_rate = neighbor_filter_stats['total_rejected'] / neighbor_filter_stats['total_pairs_before_filter'] * 100
                print(f"  - Rejection rate: {rejection_rate:.2f}%")
            print(f"{'='*80}\n")
        
        # extract match groups from union-find
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
            # find which view idx belongs to
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
        
        # compute per-view start indices
        view_start_indices = [0]
        for count in view_idx_bounds[:-1]:
            view_start_indices.append(view_start_indices[-1] + count)
        
        # Opt 1: precompute view index per global idx
        idx_to_view = {}
        for idx in range(n_total):
            view_idx = self._get_view_index(idx, view_start_indices, view_idx_bounds)
            idx_to_view[idx] = view_idx
        
        # Opt 2: precompute all target indices per view
        view_to_indices = {}
        for view_idx in range(len(view_idx_bounds)):
            start = view_start_indices[view_idx]
            end = start + view_idx_bounds[view_idx]
            view_to_indices[view_idx] = set(range(start, end))
        
        # init dynamic distance matrix (updated during clustering)
        dynamic_dist = dist_matrix.copy()
        
        # Rule 1: intra-view distances set to inf
        for i, (start, count) in enumerate(zip(view_start_indices, view_idx_bounds)):
            end = start + count
            dynamic_dist[start:end, start:end] = np.inf
        
        # diagonal set to inf (self-pairs)
        np.fill_diagonal(dynamic_dist, np.inf)
        
        # init: each target is its own cluster
        clusters = {i: [i] for i in range(n_total)}
        active_clusters = set(range(n_total))
        
        # map global idx -> cluster
        idx_to_cluster = {i: i for i in range(n_total)}
        
        # views occupied by each cluster (one-to-one)
        cluster_views = {idx: {idx_to_view[idx]} for idx in range(n_total)}
        
        # Opt 3: cache inter-cluster distances
        cluster_dist_cache = {}
        
        def get_cluster_distance(ci, cj):
            """Minimum distance between two clusters (single linkage)."""
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
        
        # ==================== Iterative clustering ====================
        iteration = 0
        max_iterations = n_total * 2  # guard against infinite loop
        
        while iteration < max_iterations:
            iteration += 1
            
            # find minimum distance among active clusters
            min_dist = np.inf
            best_pair = None
            
            # Opt 4: iterate active cluster pairs only
            active_list = list(active_clusters)
            for i, ci in enumerate(active_list):
                for cj in active_list[i+1:]:
                    cluster_dist = get_cluster_distance(ci, cj)
                    
                    if cluster_dist < min_dist:
                        min_dist = cluster_dist
                        best_pair = (ci, cj)
            
            # stop when no mergeable cluster pair
            if min_dist > self.threshold or np.isinf(min_dist) or best_pair is None:
                break
            
            ci, cj = best_pair
            
            # ==================== Check one-to-one constraint ====================
            merged_views = cluster_views[ci] | cluster_views[cj]
            
            # cannot merge if views overlap (one-to-one violation)
            if len(merged_views) < len(cluster_views[ci]) + len(cluster_views[cj]):
                # mask all distances between these clusters
                for idx_i in clusters[ci]:
                    for idx_j in clusters[cj]:
                        dynamic_dist[idx_i, idx_j] = np.inf
                        dynamic_dist[idx_j, idx_i] = np.inf
                
                # clear related cache entries
                key = (min(ci, cj), max(ci, cj))
                cluster_dist_cache[key] = np.inf
                continue
            
            # ==================== Perform merge ====================
            # merge cluster cj into ci
            clusters[ci].extend(clusters[cj])
            cluster_views[ci] = merged_views
            
            # update idx_to_cluster
            for idx in clusters[cj]:
                idx_to_cluster[idx] = ci
            
            # clear cache entries involving cj
            keys_to_remove = [k for k in cluster_dist_cache.keys() if ci in k or cj in k]
            for k in keys_to_remove:
                del cluster_dist_cache[k]
            
            # remove cluster cj
            del clusters[cj]
            del cluster_views[cj]
            active_clusters.remove(cj)
            
            # Rule 2: update distance matrix
            # Opt 5: use precomputed mapping
            
            # for each target in ci
            for idx_i in clusters[ci]:
                view_i = idx_to_view[idx_i]
                
                # for each associated view
                for assoc_view in cluster_views[ci]:
                    if assoc_view == view_i:
                        continue
                    
                    # use precomputed view index set
                    for idx_other in view_to_indices[assoc_view]:
                        # mask distance if idx_other not in same cluster
                        if idx_to_cluster[idx_other] != ci:
                            dynamic_dist[idx_i, idx_other] = np.inf
                            dynamic_dist[idx_other, idx_i] = np.inf
        
        # ==================== Extract final matches ====================
        matches = [sorted(cluster) for cluster in clusters.values()]
        
        return matches
    
    
    def _can_merge_clusters(self,
                           cluster1: List[int],
                           cluster2: List[int],
                           view_start_indices: List[int],
                           view_idx_bounds: List[int]) -> bool:

        merged = cluster1 + cluster2
        
        # check size
        if len(merged) > len(view_idx_bounds):
            return False
        
        # at most one target per view
        view_counts = self._count_targets_per_view(
            merged, view_start_indices, view_idx_bounds
        )
        
        return all(count <= 1 for count in view_counts.values())
    
    def _greedy_matching(self,
                        dist_matrix: np.ndarray,
                        view_idx_bounds: List[int]) -> List[List[int]]:

        n_total = dist_matrix.shape[0]
        uf = UnionFind(n_total)
        
        # compute per-view start indices
        view_start_indices = [0]
        for count in view_idx_bounds[:-1]:
            view_start_indices.append(view_start_indices[-1] + count)
        
        # all cross-view distance pairs (dist, i, j)
        pairs = []
        for i in range(n_total):
            view_i = self._get_view_index(i, view_start_indices, view_idx_bounds)
            for j in range(i + 1, n_total):
                view_j = self._get_view_index(j, view_start_indices, view_idx_bounds)
                
                # cross-view pairs only
                if view_i != view_j and dist_matrix[i, j] <= self.threshold:
                    pairs.append((dist_matrix[i, j], i, j))
        
        # sort by distance
        pairs.sort(key=lambda x: x[0])
        
        # greedy matching
        for dist, i, j in pairs:
            root_i = uf.find(i)
            root_j = uf.find(j)
            
            if root_i != root_j:
                # check merge constraints
                group_i = [k for k in range(n_total) if uf.find(k) == root_i]
                group_j = [k for k in range(n_total) if uf.find(k) == root_j]
                
                if self._can_merge_clusters(
                    group_i, group_j,
                    view_start_indices, view_idx_bounds
                ):
                    uf.union(i, j)
        
        # extract match groups
        groups = uf.get_groups()
        matches = [sorted(group) for group in groups.values()]
        
        return matches
    
    def _generate_global_mapping(self,
                                matches: List[List[int]],
                                idx_to_view_id: Dict[int, Tuple[str, int]],
                                view_names: List[str]) -> Dict[Tuple[str, int], int]:
        """
        Build global ID mapping (inherit minimum local ID).
        """
        mapping = {}
        
        # for each match group (e.g. [idx_view1_id6, idx_view2_id18])
        for match_group in matches:
            # 1. collect original info for the group
            group_members = []
            original_ids = []
            
            for idx in match_group:
                if idx in idx_to_view_id:
                    view_name, local_id = idx_to_view_id[idx]
                    group_members.append((view_name, local_id))
                    original_ids.append(local_id)
            
            if not group_members:
                continue
                
            # 2. pick leader ID
            # Strategy A: minimum ID (recommended, most stable)
            # Same person in a group -> use smallest ID as global ID.
            # e.g. {6, 18} -> 6; view1 keeps 6, view2 remaps 18 -> 6.
            global_id = min(original_ids)
            
            # Strategy B (optional): primary view priority
            # Prefer c001 ID when present in the group.
            # c001_ids = [mid for v, mid in group_members if v == 'c001']
            # if c001_ids:
            #     global_id = c001_ids[0]
            # else:
            #     global_id = min(original_ids)
            
            # 3. build mapping
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

