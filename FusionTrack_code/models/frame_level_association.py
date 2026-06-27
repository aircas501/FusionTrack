

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
        
        # history [(frame_id, associations, confidence_scores), ...]
        self.history = []
        self.prev_associations = None
        
    def associate_frame(self,
                       frame_id: int,
                       view_detections: Dict[str, List[Tuple[int, torch.Tensor]]],
                       view_names: List[str],
                       view_bboxes: Dict[str, torch.Tensor] = None) -> Dict[Tuple[str, int], int]:

        # 1. extract features and IDs
        view_features = {}
        view_ids = {}
        
        for view_name in view_names:
            if view_name not in view_detections or len(view_detections[view_name]) == 0:
                continue
            
            detections = view_detections[view_name]
            
            # split track_id and feature
            track_ids = [det[0] for det in detections]
            features = torch.stack([det[1] for det in detections])  # (N, C)
            
            view_features[view_name] = features
            view_ids[view_name] = track_ids
        
        # 2. base associator for current frame (with bboxes)
        current_associations = self.base_associator.associate(
            view_features, view_ids, view_names, view_bboxes
        )
        
        # 3. final associations per voting strategy
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
        
        # 4. update history
        self.prev_associations = final_associations
        
        return final_associations
    
    def _sliding_window_vote(self,
                             frame_id: int,
                             current_associations:  Dict,
                             view_features: Dict) -> Dict:
        """
        Sliding-window voting.
        
        Count pair occurrences in the window; keep the most frequent.
        """
        # append to history
        self.history.append((frame_id, current_associations, view_features))
        
        # enforce window size
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # insufficient history: return current associations
        if len(self.history) < self.min_votes:
            return current_associations
        
        # count occurrences per pair
        pair_votes = defaultdict(int)
        
        for _, assoc, _ in self.history:
            pairs = self._extract_pairs(assoc)
            for pair in pairs:
                pair_votes[pair] += 1
        
        # rebuild associations from votes
        final_associations = self._rebuild_from_votes(pair_votes)
        
        return final_associations
    
    def _weighted_vote(self,
                      frame_id: int,
                      current_associations: Dict,
                      view_features: Dict) -> Dict:
        """
        Weighted voting.
        
        Use feature similarity as weight.
        """
        # append to history
        self.history.append((frame_id, current_associations, view_features))
        
        # enforce window size
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        if len(self.history) < self.min_votes:
            return current_associations
        
        # compute weighted votes
        pair_scores = defaultdict(float)
        
        for fid, assoc, features in self.history:
            pairs = self._extract_pairs(assoc)
            
            for pair in pairs:
                # feature similarity as pair weight
                weight = self._compute_pair_similarity(pair, features)
                pair_scores[pair] += weight
        
        # rebuild from scores (threshold = mean score)
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
        Temporal consistency constraint.
        
        Prefer consistency with previous frame associations.
        """
        if self.prev_associations is None:
            return current_associations
        
        # keep associations consistent with previous frame
        consistent_associations = {}
        
        for key, global_id in current_associations.items():
            if key in self.prev_associations:
                prev_global_id = self.prev_associations[key]
                
                # check consistency
                if global_id == prev_global_id:
                    # consistent: keep
                    consistent_associations[key] = global_id
                else:
                    # inconsistent: decide by weight
                    # keep previous if random < consistency_weight
                    if np.random.random() < self.consistency_weight:
                        consistent_associations[key] = prev_global_id
                    else:
                        consistent_associations[key] = global_id
            else:
                # new target: use current association
                consistent_associations[key] = global_id
        
        return consistent_associations
    
    def _extract_pairs(self, associations: Dict) -> List[Tuple]:
        """
        Extract pairs from associations.
        
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
        # group by global_id
        global_groups = defaultdict(list)
        for key, global_id in associations.items():
            global_groups[global_id].append(key)
        
        # extract pairs (within each group)
        pairs = []
        for global_id, group in global_groups.items():
            if len(group) < 2:
                continue
            
            # pairwise pairs
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # sort for stable ordering
                    pair = tuple(sorted([group[i], group[j]]))
                    pairs.append(pair)
        
        return pairs
    
    def _compute_pair_similarity(self, 
                                 pair: Tuple, 
                                 view_features: Dict) -> float:
        """
        Compute feature similarity for a pair.
        
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
            return 1.0  # default weight
        
        try:
            # get features
            feats1 = view_features[view1]  # (N1, C)
            feats2 = view_features[view2]  # (N2, C)
            
            # locate matching features
            # note: tid is index in detections list
            if tid1 >= len(feats1) or tid2 >= len(feats2):
                return 1.0
            
            feat1 = feats1[tid1]  # (C,)
            feat2 = feats2[tid2]  # (C,)
            
            # cosine similarity
            feat1_np = feat1.cpu().numpy()
            feat2_np = feat2.cpu().numpy()
            
            similarity = np.dot(feat1_np, feat2_np) / (
                np.linalg.norm(feat1_np) * np.linalg.norm(feat2_np) + 1e-6
            )
            
            return max(0, similarity)
            
        except Exception as e:
            # on error, return default weight
            return 1.0
    
    def _rebuild_from_votes(self, pair_votes: Dict) -> Dict:

        # 1. filter low-vote pairs
        threshold = self.min_votes
        valid_pairs = [pair for pair, votes in pair_votes.items() 
                      if votes >= threshold]
        
        # 2. handle empty case
        if len(valid_pairs) == 0:
            # insufficient votes: return latest associations
            if len(self.history) > 0:
                return self.history[-1][1]
            else:
                return {}
        
        # 3. prepare union-find
        from models.loss.matrix_loss import UnionFind
        
        # collect all keys
        all_keys = set()
        for pair in valid_pairs:
            all_keys.add(pair[0])
            all_keys.add(pair[1])
        
        # key -> index mapping for union-find
        key_to_idx = {key: idx for idx, key in enumerate(all_keys)}
        idx_to_key = {idx: key for key, idx in key_to_idx.items()}
        
        # init union-find
        uf = UnionFind(len(all_keys))
        
        # 4. merge pairs
        for pair in valid_pairs:
            idx1 = key_to_idx[pair[0]]
            idx2 = key_to_idx[pair[1]]
            uf.union(idx1, idx2)
        
        # 5. build associations (core logic)
        groups = uf.get_groups()
        associations = {}
        
        # for each union-find group
        for group_indices in groups.values():
            # map indices back to keys
            # e.g. cluster_keys = [('c001', 10), ('c002', 17), ('c004', 52)]
            cluster_keys = [idx_to_key[idx] for idx in group_indices]
            
            # --- key change start ---
            
            # find minimum key in the group.
            # Tuple order: cam_id (str) then track_id (int).
            # Matches earliest-ID logic; c001 < c002.
            leader_key = min(cluster_keys)
            
            # use leader_key track_id as global ID
            # key is (camera_id, track_id); take index 1
            # e.g. group with ('c001', 10) gets global ID 10
            target_global_id = leader_key[1]
            
            # map all tracks in group to this ID
            for key in cluster_keys:
                associations[key] = target_global_id
                
            # --- key change end ---
        
        return associations
    
    def reset(self):
        """Reset history."""
        self.history = []
        self.prev_associations = None


def build_frame_level_associator(config: dict, neighbor_filter=None) -> FrameLevelCrossViewAssociator:

    from models.cross_view_association import build_cross_view_associator
    
    # build base associator (with neighbor filter)
    base_associator = build_cross_view_associator(config, neighbor_filter=neighbor_filter)
    
    # build frame-level associator
    return FrameLevelCrossViewAssociator(
        base_associator=base_associator,
        voting_strategy=config.get("VOTING_STRATEGY", "sliding_window"),
        window_size=config.get("WINDOW_SIZE", 10),
        min_votes=config.get("MIN_VOTES", 5),
        consistency_weight=config.get("CONSISTENCY_WEIGHT", 0.3)
    )

