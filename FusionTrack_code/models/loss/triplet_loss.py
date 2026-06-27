
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import random


class TripletLoss(nn.Module):
    """
    Triplet loss.

    Samples anchor, positive, and negative from ReIDPool and computes triplet loss.
    """
    def __init__(self, margin: float = 0.3, normalize_feature: bool = True):
        """
        Args:
            margin: Margin for triplet loss
            normalize_feature: Whether to L2-normalize features
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
    
    def forward(self, 
                anchor_features: torch.Tensor,
                positive_features: torch.Tensor,
                negative_features: torch.Tensor):
        """
        Compute triplet loss.

        Args:
            anchor_features: (N, D) - anchor features
            positive_features: (N, D) - positive features (same ID as anchor)
            negative_features: (N, D) - negative features (different ID from anchor)

        Returns:
            loss: scalar tensor
        """
        if self.normalize_feature:
            anchor_features = F.normalize(anchor_features, p=2, dim=1)
            positive_features = F.normalize(positive_features, p=2, dim=1)
            negative_features = F.normalize(negative_features, p=2, dim=1)
        
        # Compute distances
        pos_dist = F.pairwise_distance(anchor_features, positive_features, p=2)  # (N,)
        neg_dist = F.pairwise_distance(anchor_features, negative_features, p=2)  # (N,)
        
        # Triplet loss: max(0, margin + pos_dist - neg_dist)
        loss = F.relu(self.margin + pos_dist - neg_dist)
        
        return loss.mean()
    
    def forward_with_pool(self,
                          reid_pool,
                          num_samples: int = 32,
                          hard_mining: bool = True,
                          min_samples: int = 4,
                          min_unique_ids: int = 2,
                          verbose: bool = False):
        """
        Sample from ReIDPool and compute triplet loss.

        Args:
            reid_pool: ReIDPool instance
            num_samples: Number of samples to draw
            hard_mining: Whether to use hard negative mining
            min_samples: Minimum sample count; returns None if below this
            min_unique_ids: Minimum number of distinct IDs (default 2); returns None if below this
            verbose: Whether to print detailed logs

        Returns:
            loss: scalar tensor, or None if there are not enough samples
        """
        # Check: ReIDPool must not be empty
        if reid_pool is None:
            if verbose:
                print(f"[TripletLoss] ReIDPool is None")
            return None
        
        if not hasattr(reid_pool, 'view_list') or len(reid_pool.view_list) == 0:
            if verbose:
                print(f"[TripletLoss] ReIDPool has no views")
            return None
        # Collect all available IDs and features
        all_ids = []
        all_features = []
        all_views = []
        
        for view in reid_pool.view_list:
            for id in reid_pool.view_id_reid_feat_dict_list[view].keys():
                reid_feats = reid_pool.view_id_reid_feat_dict_list[view][id]
                if reid_feats is None:
                    continue
                
                try:
                    # Use the latest feature or average feature
                    if isinstance(reid_feats, torch.Tensor):
                        if reid_feats.dim() > 1:
                            if reid_feats.shape[0] > 0:
                                feat = reid_feats[-1]
                            else:
                                continue
                        else:
                            feat = reid_feats
                        
                        # Check that the feature is valid (no NaN/Inf)
                        if torch.isnan(feat).any() or torch.isinf(feat).any():
                            if verbose:
                                print(f"[TripletLoss] Invalid feature (NaN/Inf) for ID {id} in view {view}, skipping")
                            continue  # Skip only when the feature is invalid
                        
                    all_ids.append(id)
                    all_features.append(feat)
                    all_views.append(view)
                except Exception as e:
                    if verbose:
                        print(f"[TripletLoss] Error processing feature for ID {id} in view {view}: {e}")
                    continue
        
        # Check: need at least min_samples samples
        if len(all_ids) < min_samples:
            if verbose:
                print(f"[TripletLoss] Not enough samples: {len(all_ids)} < {min_samples}")
            return None
        
        # Group by ID
        id_to_indices = {}
        for idx, id in enumerate(all_ids):
            if id not in id_to_indices:
                id_to_indices[id] = []
            id_to_indices[id].append(idx)
        
        unique_ids = list(id_to_indices.keys())
        
        # Check: need at least min_unique_ids distinct IDs to compute triplet loss
        if len(unique_ids) < min_unique_ids:
            if verbose:
                print(f"[TripletLoss] Not enough unique IDs: {len(unique_ids)} < {min_unique_ids}")
            return None
        
        # Count IDs with multiple samples (for positive sampling)
        ids_with_multiple = [id for id in unique_ids if len(id_to_indices[id]) > 1]
        
        # If no ID has multiple samples, use data augmentation
        use_augmentation = len(ids_with_multiple) == 0
        if verbose and use_augmentation:
            print(f"[TripletLoss] No ID with multiple samples, using augmentation for positive pairs")
        
        # Sample triplets
        anchor_features = []
        positive_features = []
        negative_features = []
        
        sampled_count = 0
        attempts = 0
        max_attempts = num_samples * 3  # Try at most 3x the target sample count
        
        while sampled_count < num_samples and attempts < max_attempts:
            attempts += 1
            
            if len(unique_ids) < 2:
                break
            
            # Randomly choose an ID as anchor
            anchor_id = random.choice(unique_ids)
            anchor_indices = id_to_indices[anchor_id]
            
            # Choose positive from the same ID (if multiple records exist)
            if len(anchor_indices) > 1:
                anchor_idx, pos_idx = random.sample(anchor_indices, 2)
            else:
                # With only one sample, use the same instance as anchor and positive
                anchor_idx = pos_idx = anchor_indices[0]
                # Note: this makes pos_dist ≈ 0, but the model can still learn to stay away from negatives
            
            # Choose negative from a different ID
            negative_ids = [id for id in unique_ids if id != anchor_id]
            if len(negative_ids) == 0:
                continue
            negative_id = random.choice(negative_ids)
            negative_indices = id_to_indices[negative_id]
            negative_idx = random.choice(negative_indices)
            
            # Collect features
            anchor_feat = all_features[anchor_idx]
            pos_feat = all_features[pos_idx]
            neg_feat = all_features[negative_idx]
            
            # If anchor and positive are the same (single-sample ID), add slight noise for augmentation
            if anchor_idx == pos_idx and use_augmentation:
                # Add small noise to avoid pos_dist being exactly 0
                noise = torch.randn_like(pos_feat) * 0.01
                pos_feat = pos_feat + noise
            
            anchor_features.append(anchor_feat)
            positive_features.append(pos_feat)
            negative_features.append(neg_feat)
            sampled_count += 1
        
        # Check: enough triplets sampled
        if sampled_count < min_samples:
            if verbose:
                print(f"[TripletLoss] Not enough triplets sampled: {sampled_count} < {min_samples}")
            return None
        
        if sampled_count == 0:
            if verbose:
                print(f"[TripletLoss] Failed to sample any triplets")
            return None
        
        # Ensure all features are on the same device
        try:
            # Use the device of the first feature
            device = anchor_features[0].device
            
            # Move to the same device and stack
            anchor_features = torch.stack([f.to(device) for f in anchor_features], dim=0)  # (N, D)
            positive_features = torch.stack([f.to(device) for f in positive_features], dim=0)  # (N, D)
            negative_features = torch.stack([f.to(device) for f in negative_features], dim=0)  # (N, D)
        except Exception as e:
            if verbose:
                print(f"[TripletLoss] Error stacking features: {e}")
            return None
        
        # Compute loss
        loss = self.forward(anchor_features, positive_features, negative_features)
        
        if verbose:
            print(f"[TripletLoss] Sampled {sampled_count} triplets from {len(unique_ids)} unique IDs, loss={loss.item():.4f}")
        
        return loss


def build_triplet_loss(config: dict):
    """
    Build triplet loss module.

    Args:
        config: Config dict containing:
            - TRIPLET_MARGIN: margin value (optional, default 0.3)
            - NORMALIZE_FEATURE: whether to normalize features (optional, default True)

    Returns:
        TripletLoss instance
    """
    margin = config.get("TRIPLET_MARGIN", 0.3)
    normalize_feature = config.get("NORMALIZE_FEATURE", True)
    
    return TripletLoss(margin=margin, normalize_feature=normalize_feature)
