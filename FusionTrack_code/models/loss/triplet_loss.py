
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import random


class TripletLoss(nn.Module):
    """
    三元组损失
    
    从ReIDPool中采样anchor、positive和negative样本，计算三元组损失
    """
    def __init__(self, margin: float = 0.3, normalize_feature: bool = True):
        """
        Args:
            margin: 三元组损失的margin
            normalize_feature: 是否对特征进行L2归一化
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
    
    def forward(self, 
                anchor_features: torch.Tensor,
                positive_features: torch.Tensor,
                negative_features: torch.Tensor):
        """
        计算三元组损失
        
        Args:
            anchor_features: (N, D) - anchor特征
            positive_features: (N, D) - positive特征（与anchor相同ID）
            negative_features: (N, D) - negative特征（与anchor不同ID）
        
        Returns:
            loss: scalar tensor
        """
        if self.normalize_feature:
            anchor_features = F.normalize(anchor_features, p=2, dim=1)
            positive_features = F.normalize(positive_features, p=2, dim=1)
            negative_features = F.normalize(negative_features, p=2, dim=1)
        
        # 计算距离
        pos_dist = F.pairwise_distance(anchor_features, positive_features, p=2)  # (N,)
        neg_dist = F.pairwise_distance(anchor_features, negative_features, p=2)  # (N,)
        
        # 三元组损失：max(0, margin + pos_dist - neg_dist)
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
        从ReIDPool中采样并计算三元组损失
        
        Args:
            reid_pool: ReIDPool实例
            num_samples: 采样数量
            hard_mining: 是否使用hard negative mining
            min_samples: 最小采样数量，低于此数量则返回None
            min_unique_ids: 最小不同ID数量（默认2），低于此数量则返回None
            verbose: 是否输出详细日志
        
        Returns:
            loss: scalar tensor，如果没有足够的样本则返回None
        """
        # 检查：ReIDPool是否为空
        if reid_pool is None:
            if verbose:
                print(f"[TripletLoss] ReIDPool is None")
            return None
        
        if not hasattr(reid_pool, 'view_list') or len(reid_pool.view_list) == 0:
            if verbose:
                print(f"[TripletLoss] ReIDPool has no views")
            return None
        # 收集所有可用的ID和特征
        all_ids = []
        all_features = []
        all_views = []
        
        for view in reid_pool.view_list:
            for id in reid_pool.view_id_reid_feat_dict_list[view].keys():
                reid_feats = reid_pool.view_id_reid_feat_dict_list[view][id]
                if reid_feats is None:
                    continue
                
                try:
                    # 取最新的特征或平均特征
                    if isinstance(reid_feats, torch.Tensor):
                        if reid_feats.dim() > 1:
                            if reid_feats.shape[0] > 0:
                                feat = reid_feats[-1]
                            else:
                                continue
                        else:
                            feat = reid_feats
                        
                        # 检查特征是否有效（非NaN、非Inf）
                        if torch.isnan(feat).any() or torch.isinf(feat).any():
                            if verbose:
                                print(f"[TripletLoss] Invalid feature (NaN/Inf) for ID {id} in view {view}, skipping")
                            continue  # ✅ 修正缩进：只在特征无效时跳过
                        
                    all_ids.append(id)
                    all_features.append(feat)
                    all_views.append(view)
                except Exception as e:
                    if verbose:
                        print(f"[TripletLoss] Error processing feature for ID {id} in view {view}: {e}")
                    continue
        
        # 检查：至少需要min_samples个样本
        if len(all_ids) < min_samples:
            if verbose:
                print(f"[TripletLoss] Not enough samples: {len(all_ids)} < {min_samples}")
            return None
        
        # 按ID分组
        id_to_indices = {}
        for idx, id in enumerate(all_ids):
            if id not in id_to_indices:
                id_to_indices[id] = []
            id_to_indices[id].append(idx)
        
        unique_ids = list(id_to_indices.keys())
        
        # 检查：至少需要min_unique_ids个不同的ID才能计算triplet loss
        if len(unique_ids) < min_unique_ids:
            if verbose:
                print(f"[TripletLoss] Not enough unique IDs: {len(unique_ids)} < {min_unique_ids}")
            return None
        
        # 统计有多个样本的ID数量（用于positive采样）
        ids_with_multiple = [id for id in unique_ids if len(id_to_indices[id]) > 1]
        
        # 如果没有任何ID有多个样本，需要添加数据增强
        use_augmentation = len(ids_with_multiple) == 0
        if verbose and use_augmentation:
            print(f"[TripletLoss] No ID with multiple samples, using augmentation for positive pairs")
        
        # 采样三元组
        anchor_features = []
        positive_features = []
        negative_features = []
        
        sampled_count = 0
        attempts = 0
        max_attempts = num_samples * 3  # 最多尝试3倍的采样次数
        
        while sampled_count < num_samples and attempts < max_attempts:
            attempts += 1
            
            if len(unique_ids) < 2:
                break
            
            # 随机选择一个ID作为anchor
            anchor_id = random.choice(unique_ids)
            anchor_indices = id_to_indices[anchor_id]
            
            # 从同一ID中选择positive（如果有多条记录）
            if len(anchor_indices) > 1:
                anchor_idx, pos_idx = random.sample(anchor_indices, 2)
            else:
                # 只有一个样本时，使用同一个作为anchor和positive
                anchor_idx = pos_idx = anchor_indices[0]
                # 注意：这会导致pos_dist≈0，但仍然可以学习远离negative
            
            # 从不同ID中选择negative
            negative_ids = [id for id in unique_ids if id != anchor_id]
            if len(negative_ids) == 0:
                continue
            negative_id = random.choice(negative_ids)
            negative_indices = id_to_indices[negative_id]
            negative_idx = random.choice(negative_indices)
            
            # 收集特征
            anchor_feat = all_features[anchor_idx]
            pos_feat = all_features[pos_idx]
            neg_feat = all_features[negative_idx]
            
            # 如果anchor和positive是同一个（单样本ID），添加轻微噪声作为数据增强
            if anchor_idx == pos_idx and use_augmentation:
                # 添加小噪声，避免pos_dist完全为0
                noise = torch.randn_like(pos_feat) * 0.01
                pos_feat = pos_feat + noise
            
            anchor_features.append(anchor_feat)
            positive_features.append(pos_feat)
            negative_features.append(neg_feat)
            sampled_count += 1
        
        # 检查：采样数量是否足够
        if sampled_count < min_samples:
            if verbose:
                print(f"[TripletLoss] Not enough triplets sampled: {sampled_count} < {min_samples}")
            return None
        
        if sampled_count == 0:
            if verbose:
                print(f"[TripletLoss] Failed to sample any triplets")
            return None
        
        # 确保所有特征在同一设备上
        try:
            # 获取目标设备（使用第一个特征的设备）
            device = anchor_features[0].device
            
            # 移动到同一设备并stack
            anchor_features = torch.stack([f.to(device) for f in anchor_features], dim=0)  # (N, D)
            positive_features = torch.stack([f.to(device) for f in positive_features], dim=0)  # (N, D)
            negative_features = torch.stack([f.to(device) for f in negative_features], dim=0)  # (N, D)
        except Exception as e:
            if verbose:
                print(f"[TripletLoss] Error stacking features: {e}")
            return None
        
        # 计算损失
        loss = self.forward(anchor_features, positive_features, negative_features)
        
        if verbose:
            print(f"[TripletLoss] Sampled {sampled_count} triplets from {len(unique_ids)} unique IDs, loss={loss.item():.4f}")
        
        return loss


def build_triplet_loss(config: dict):
    """
    构建三元组损失
    
    Args:
        config: 配置字典，需要包含：
            - TRIPLET_MARGIN: margin值（可选，默认0.3）
            - NORMALIZE_FEATURE: 是否归一化特征（可选，默认True）
    
    Returns:
        TripletLoss实例
    """
    margin = config.get("TRIPLET_MARGIN", 0.3)
    normalize_feature = config.get("NORMALIZE_FEATURE", True)
    
    return TripletLoss(margin=margin, normalize_feature=normalize_feature)
