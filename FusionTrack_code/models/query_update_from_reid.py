
import torch
from typing import Dict, Any, Optional, List, Tuple


def update_query_with_reid_features(tracks: Any,
                                      reid_features: Dict[int, torch.Tensor],
                                      reid_update_weight: float = 0.1,
                                      use_dab: bool = True) -> Any:

    # 支持列表格式
    is_list = isinstance(tracks, list)
    if is_list:
        if len(tracks) == 0:
            return tracks
        tracks = tracks[0]
    
    ids = getattr(tracks, "ids", None)
    query_embed = getattr(tracks, "query_embed", None)
    
    if ids is None or query_embed is None or len(ids) == 0:
        return [tracks] if is_list else tracks
    
    device = query_embed.device
    hidden_dim = query_embed.shape[-1] if use_dab else query_embed.shape[-1] // 2
    
    # 提取当前query特征
    if use_dab:
        current_feats = query_embed  # (N, H)
    else:
        current_feats = query_embed[..., hidden_dim:]  # (N, H)
    
    updated_feats = []
    update_count = 0
    
    for i in range(len(ids)):
        tid = int(ids[i].item())
        current_feat = current_feats[i]  # (H,)
        
        # 无效ID或reid_features中没有该ID的特征，保持不变
        if tid < 0 or tid not in reid_features:
            updated_feats.append(current_feat)
            continue
        
        # 从reid_features获取该ID的ReID特征
        reid_feat = reid_features[tid]
        
        if reid_feat is None:
            updated_feats.append(current_feat)
            continue
        
        # 转换到正确的设备和dtype
        reid_feat_tensor = reid_feat.to(device)
        if reid_feat_tensor.dim() > 1:
            reid_feat_tensor = reid_feat_tensor.squeeze(0)
        
        # 如果维度不匹配，需要处理
        if reid_feat_tensor.shape[-1] != hidden_dim:
            # 简单的截断或padding
            if reid_feat_tensor.shape[-1] > hidden_dim:
                reid_feat_tensor = reid_feat_tensor[:hidden_dim]
            else:
                # padding zeros
                padding = torch.zeros(hidden_dim - reid_feat_tensor.shape[-1], 
                                    device=device, dtype=reid_feat_tensor.dtype)
                reid_feat_tensor = torch.cat([reid_feat_tensor, padding], dim=0)
        
        # 加权融合：new_feat = (1 - w) * current_feat + w * reid_feat
        updated_feat = (1 - reid_update_weight) * current_feat + reid_update_weight * reid_feat_tensor
        updated_feats.append(updated_feat)
        update_count += 1
    
    # 重新组装query_embed（原地修改）
    updated_feats_tensor = torch.stack(updated_feats, dim=0).to(device)  # (N, H)
    
    if use_dab:
        tracks.query_embed = updated_feats_tensor
    else:
        # 保留pos部分，只更新feat部分
        pos_part = query_embed[:, :hidden_dim]
        tracks.query_embed = torch.cat([pos_part, updated_feats_tensor], dim=-1)
    
    return [tracks] if is_list else tracks


def update_query_from_reid_pool(tracks: Any,
                                  view: str,
                                  reid_pool: Any,
                                  reid_update_weight: float = 0.3,
                                  use_dab: bool = True) -> Any:

    # 支持列表格式
    is_list = isinstance(tracks, list)
    if is_list:
        if len(tracks) == 0:
            return tracks
        tracks = tracks[0]
    
    ids = getattr(tracks, "ids", None)
    query_embed = getattr(tracks, "query_embed", None)
    
    if ids is None or query_embed is None or len(ids) == 0:
        return [tracks] if is_list else tracks
    
    device = query_embed.device
    hidden_dim = query_embed.shape[-1] if use_dab else query_embed.shape[-1] // 2
    
    # 获取ReIDPool中的特征字典
    if not hasattr(reid_pool, 'view_id_reid_feat_dict_list'):
        return [tracks] if is_list else tracks
    
    view_pool = reid_pool.view_id_reid_feat_dict_list.get(view, {})
    if len(view_pool) == 0:
        return [tracks] if is_list else tracks
    
    # 提取当前query特征
    if use_dab:
        current_feats = query_embed  # (N, H)
    else:
        current_feats = query_embed[..., hidden_dim:]  # (N, H)
    
    updated_feats = []
    
    for i in range(len(ids)):
        tid = int(ids[i].item())
        current_feat = current_feats[i]  # (H,)
        
        # 无效ID或ReIDPool中没有该ID的特征，保持不变
        if tid < 0 or tid not in view_pool:
            updated_feats.append(current_feat)
            continue
        
        # 从ReIDPool获取该ID的ReID特征（可能是多帧的均值）
        reid_feat = view_pool[tid]
        
        if reid_feat is None:
            updated_feats.append(current_feat)
            continue
        
        # 转换到正确的设备和dtype
        reid_feat_tensor = reid_feat.to(device)
        if reid_feat_tensor.dim() > 1:
            reid_feat_tensor = reid_feat_tensor.squeeze(0)
        
        # 如果维度不匹配，需要投影（通常ReID特征维度和query维度相同）
        if reid_feat_tensor.shape[-1] != hidden_dim:
            # 简单的线性插值或截断（更好的方式是用投影层，但这里简化处理）
            if reid_feat_tensor.shape[-1] > hidden_dim:
                reid_feat_tensor = reid_feat_tensor[:hidden_dim]
            else:
                # padding zeros
                padding = torch.zeros(hidden_dim - reid_feat_tensor.shape[-1], device=device, dtype=reid_feat_tensor.dtype)
                reid_feat_tensor = torch.cat([reid_feat_tensor, padding], dim=0)
        
        # 加权融合：new_feat = (1 - w) * current_feat + w * reid_feat
        updated_feat = (1 - reid_update_weight) * current_feat + reid_update_weight * reid_feat_tensor
        updated_feats.append(updated_feat)
    
    # 重新组装query_embed
    updated_feats_tensor = torch.stack(updated_feats, dim=0).to(device)  # (N, H)
    
    if use_dab:
        tracks.query_embed = updated_feats_tensor
    else:
        # 保留pos部分，只更新feat部分
        pos_part = query_embed[:, :hidden_dim]
        tracks.query_embed = torch.cat([pos_part, updated_feats_tensor], dim=-1)
    
    return [tracks] if is_list else tracks


def update_query_from_memory_bank(tracks: Any,
                                    view: str,
                                    memory_bank: Any,
                                    current_t: int,
                                    reid_update_weight: float = 0.3,
                                    use_dab: bool = True,
                                    max_history_frames: int = 5) -> Any:

    # 支持列表格式
    is_list = isinstance(tracks, list)
    if is_list:
        if len(tracks) == 0:
            return tracks
        tracks = tracks[0]
    
    ids = getattr(tracks, "ids", None)
    query_embed = getattr(tracks, "query_embed", None)
    
    if ids is None or query_embed is None or len(ids) == 0:
        return [tracks] if is_list else tracks
    
    device = query_embed.device
    hidden_dim = query_embed.shape[-1] if use_dab else query_embed.shape[-1] // 2
    
    # 提取当前query特征
    if use_dab:
        current_feats = query_embed  # (N, H)
    else:
        current_feats = query_embed[..., hidden_dim:]  # (N, H)
    
    updated_feats = []
    
    for i in range(len(ids)):
        tid = int(ids[i].item())
        current_feat = current_feats[i]  # (H,)
        
        if tid < 0:
            updated_feats.append(current_feat)
            continue
        
        # 从MemoryBank获取该ID的历史序列
        history_seq = memory_bank.gather_seq(view, tid, k=max_history_frames)
        
        if len(history_seq) == 0:
            updated_feats.append(current_feat)
            continue
        
        # 收集历史特征（简单平均）
        history_feats = []
        for rec in history_seq:
            if rec.get("feat") is not None:
                feat = rec["feat"].to(device)
                if feat.dtype == torch.float16:
                    feat = feat.to(torch.float32)
                history_feats.append(feat)
        
        if len(history_feats) == 0:
            updated_feats.append(current_feat)
            continue
        
        # 计算历史特征的均值
        history_feat_mean = torch.stack(history_feats, dim=0).mean(dim=0)  # (H,)
        
        # 加权融合
        updated_feat = (1 - reid_update_weight) * current_feat + reid_update_weight * history_feat_mean
        updated_feats.append(updated_feat)
    
    # 重新组装query_embed
    updated_feats_tensor = torch.stack(updated_feats, dim=0).to(device)  # (N, H)
    
    if use_dab:
        tracks.query_embed = updated_feats_tensor
    else:
        pos_part = query_embed[:, :hidden_dim]
        tracks.query_embed = torch.cat([pos_part, updated_feats_tensor], dim=-1)
    
    return [tracks] if is_list else tracks
