

from collections import deque
from typing import Dict, Optional, Any, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- utils ---------------------------

def _to_cpu(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x

def _ensure_tensor(x, device, dtype=None):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.to(device=device)
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x

def _safe_getattr(obj, name, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else default


# --------------------------- MemoryBank core ---------------------------

class MemoryBank(nn.Module):
    """
    Memory bank that stores recent T frames of tracking outputs for multi-view and multi-target.

    Data layout (deque node):
      {
        "t": <int or float_timestamp>,
        "views": {
          cam_id(str): {
            track_id(int): {
              "cam": str,                   # 视角名称
              "t": int|float,               # 时间戳
              "id": int,                    # 目标ID
              "feat": Tensor(H)             # query特征
            }, ...
          }, ...
        }
      }

    """

    def __init__(self,
                 bank_len: int = 30,
                 hidden_dim: int = 256,
                 use_dab: bool = True,
                 temporal_k: int = 8,
                 decay_alpha: float = 0.25,
                 num_heads: int = 8,
                 attn_dropout: float = 0.0,
                 ff_dropout: float = 0.0,
                 device: str = "cpu",
                 store_fp16: bool = False,
                 training: bool = True,
                 reid_update_weight: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.bank_len = int(bank_len)
        self.hidden_dim = int(hidden_dim)
        self.use_dab = bool(use_dab)
        self.temporal_k = int(temporal_k)#用几个帧计算
        self.decay_alpha = float(decay_alpha)
        self.num_heads = int(num_heads)
        self.store_fp16 = bool(store_fp16)
        self.training = bool(training)
        self.reid_update_weight = float(reid_update_weight)  # ReID特征更新的权重

        # storage on CPU, computations on device
        self.runtime_device = torch.device(device)
        # 修复：训练时也限制长度为bank_len（与clip长度一致），避免内存无限增长
        # 训练和推理都使用相同的maxlen，因为会在clip间清空
        self.deque = deque(maxlen=self.bank_len)

        # projection layers for attention updates (shared)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
        # ReID特征投影层（延迟初始化，因为不知道reid特征维度）
        self.reid_proj = None

    # --------------------------- public APIs ---------------------------

    def clear(self):
        self.deque.clear()
    
    # --------------------------- Query Update Methods ---------------------------
    
    def weighted_cross_frame_attention(self, 
                                       current_query: torch.Tensor,
                                       historical_queries: List[torch.Tensor],
                                       time_diffs: List[float],
                                       alpha: float = 0.1) -> torch.Tensor:

        if len(historical_queries) == 0:
            return current_query  # 没有历史帧，返回原query
        
        device = current_query.device
        C = current_query.shape[-1]
        
        # Stack所有历史query: (N, C)
        # 注意：N可以是1-tau之间的任意值（历史帧不足tau时会自动使用现有的帧）
        hist_feats = torch.stack(historical_queries, dim=0).to(device)  # (N, C), N <= tau
        
        # 计算时间衰减权重: W = exp(-alpha * dt)
        time_diffs_tensor = torch.tensor(time_diffs, device=device, dtype=torch.float32)  # (N,)
        time_weights = torch.exp(-alpha * time_diffs_tensor)  # (N,) - 越近的帧权重越大
        
        # 使用当前query作为Q，历史queries作为K和V
        # 简化版本：使用加权平均 + 注意力
        # Q: (1, C), K: (N, C), V: (N, C)
        query = current_query.unsqueeze(0)  # (1, C)
        key = hist_feats  # (N, C)
        value = hist_feats  # (N, C)
        
        # 计算注意力分数 (1, N)
        attn_scores = torch.matmul(query, key.T) / math.sqrt(C)  # (1, N)
        
        # 结合时间权重
        attn_scores = attn_scores.squeeze(0)  # (N,)
        attn_scores = attn_scores * time_weights  # 时间加权
        attn_weights = F.softmax(attn_scores, dim=0)  # (N,)
        
        # 加权求和
        updated_feat = torch.matmul(attn_weights.unsqueeze(0), value).squeeze(0)  # (C,)
        
        # 残差连接：融合当前query和更新后的特征
        updated_query = 0.5 * current_query + 0.5 * updated_feat
        
        return updated_query
    
    
    def update_tracks_with_memory(self,
                                  tracks: Any,
                                  view: str,
                                  current_t: int,
                                  tau: int = 3,
                                  alpha: float = 0.1,
                                  cross_view: bool = True,
                                  all_view_tracks: Optional[Dict[str, Any]] = None) -> Any:

        # 支持列表格式：TrackInstances.init_tracks() 返回的是 list
        is_list = isinstance(tracks, list)
        if is_list:
            if len(tracks) == 0:
                return tracks
            # 如果是列表，通常取第一个元素（batch_size=1的情况）
            tracks = tracks[0]
        
        ids = _safe_getattr(tracks, "ids")
        if ids is None or len(ids) == 0:
            return [tracks] if is_list else tracks
        
        q_embed = _safe_getattr(tracks, "query_embed")
        if q_embed is None:
            return [tracks] if is_list else tracks
        
        device = q_embed.device
        
        # 提取当前的feat（根据use_dab判断如何提取）
        current_feats = self._extract_feat_from_query_embed(q_embed)
        if current_feats is None:
            return [tracks] if is_list else tracks
        
        updated_feats = []
        
        for i in range(len(ids)):
            tid = int(ids[i].item())
            if tid < 0:
                updated_feats.append(current_feats[i])
                continue
            
            current_feat = current_feats[i]  # (C,)
            
            # 1. 跨帧更新
            historical_seq = self.gather_seq(view, tid, k=tau)
            if len(historical_seq) > 0:
                historical_queries = []
                time_diffs = []
                for rec in historical_seq:
                    if rec.get("feat") is not None:
                        hist_feat = rec["feat"].to(device)
                        if hist_feat.dtype == torch.float16:
                            hist_feat = hist_feat.to(torch.float32)
                        historical_queries.append(hist_feat)
                        time_diffs.append(float(current_t - rec["t"]))
                
                if len(historical_queries) > 0:
                    current_feat = self.weighted_cross_frame_attention(
                        current_feat, historical_queries, time_diffs, alpha
                    )
            
            
            updated_feats.append(current_feat)
        
        # 重新组装query_embed
        updated_feats_tensor = torch.stack(updated_feats, dim=0).to(device)  # (N, C)
        
        # 根据use_dab重新构造query_embed
        if self.use_dab:
            # use_dab=True: query_embed就是feat
            tracks.query_embed = updated_feats_tensor
        else:
            # use_dab=False: query_embed = [pos | feat]
            # 保留原来的pos部分，只更新feat部分
            original_q_embed = q_embed.to(device)
            pos_part = original_q_embed[:, :self.hidden_dim]  # (N, H)
            tracks.query_embed = torch.cat([pos_part, updated_feats_tensor], dim=-1)  # (N, 2H)
        
        # 如果输入是列表，输出也应该是列表
        return [tracks] if is_list else tracks

    @torch.no_grad()
    def push_from_views(self,
                        per_view_tracks: Dict[str, Any],  # {cam_id: TrackInstances}
                        t: int | float,
                        reid_features: Dict[Tuple[str, int], torch.Tensor] = None):
        """
        MemoryBank更新流程（训练和推理统一）：
        1. 插入原始记录（所有视角）
        2. 跨帧更新（temporal update）- 使用历史帧增强当前query
        """
        # 1) Insert raw records for all cams (no update yet)
        frame_node = self._ensure_frame_node(t)
        for cam_id, tracks in per_view_tracks.items():
            self._insert_tracks_for_view(frame_node, cam_id, tracks)

        # 2) Temporal update for each (cam_id, track_id)
        # 跨帧更新：使用同一视角的历史帧来增强当前query
        self._temporal_update_all(t)

    # Query helpers

    def recent_frames(self, k: Optional[int] = None) -> List[dict]:
        arr = list(self.deque)
        if k is None or k >= len(arr):
            return arr
        return arr[-k:]

    def gather_seq(self, cam_id: str, track_id: int, k: Optional[int] = None) -> List[dict]:
        """Return ascending-time sequence of records for (cam_id, track_id) over last k frames."""
        seq = []
        frames = self.recent_frames(k)
        for fr in frames:
            store = fr["views"].get(str(cam_id), {})
            if track_id in store:
                seq.append(store[track_id])
        return seq

    def gather_crossview_at_t(self, t: int | float, track_id: int) -> Dict[str, dict]:
        """Return {cam_id: record} at timestamp t for same track_id (if exists)."""
        fr = self._find_frame_node(t)
        if fr is None: return {}
        out = {}
        for cam, d in fr["views"].items():
            if track_id in d:
                out[cam] = d[track_id]
        return out

    # --------------------------- internal: insert ---------------------------

    def _ensure_frame_node(self, t):
        if len(self.deque) > 0 and self.deque[-1]["t"] == t:
            return self.deque[-1]
        node = {"t": t, "views": {}}
        self.deque.append(node)
        return node

    def _find_frame_node(self, t) -> Optional[dict]:
        # linear scan is fine for small bank_len
        for fr in reversed(self.deque):
            if fr["t"] == t:
                return fr
        return None

    def _insert_tracks_for_view(self, frame_node: dict, cam_id: str, tracks: Any):

        view_store = frame_node["views"].setdefault(str(cam_id), {})

        # 支持列表格式：TrackInstances.init_tracks() 返回的是 list
        if isinstance(tracks, list):
            if len(tracks) == 0:
                return
            # 如果是列表，通常取第一个元素（batch_size=1的情况）
            tracks = tracks[0]
        
        ids = _safe_getattr(tracks, "ids")
        if ids is None:
            return

        q_embed = _safe_getattr(tracks, "query_embed")

        # 从query_embed提取特征
        feats = self._extract_feat_from_query_embed(q_embed)

        # 移到CPU存储（detach）
        ids = _to_cpu(ids)
        feats = _to_cpu(feats)

        # 写入存储（精简版）
        for i in range(len(ids)):
            tid = int(ids[i].item())
            if tid < 0:
                continue
            rec = {
                "cam": str(cam_id),
                "t": frame_node["t"],
                "id": tid,
                "feat": feats[i] if feats is not None else None,
            }
            # 可选：fp16存储以节省内存
            if self.store_fp16 and isinstance(rec.get("feat"), torch.Tensor):
                rec["feat"] = rec["feat"].to(dtype=torch.float16)
            view_store[tid] = rec

    def _extract_feat_from_query_embed(self, q_embed) -> Optional[torch.Tensor]:
        """Return the feature (H) from query_embed which may be (H) or (2H) depending on DAB usage."""
        if q_embed is None or not isinstance(q_embed, torch.Tensor):
            return None
        if self.use_dab:
            # query_embed itself is the feature (H)
            assert q_embed.shape[-1] == self.hidden_dim, \
                f"use_dab=True expects query_embed dim=H({self.hidden_dim}), got {q_embed.shape}"
            return q_embed
        else:
            # non-DAB: query_embed = [pos(H) | feat(H)] of length 2H
            assert q_embed.shape[-1] == 2 * self.hidden_dim, \
                f"use_dab=False expects query_embed dim=2H({2*self.hidden_dim}), got {q_embed.shape}"
            return q_embed[..., self.hidden_dim:]  # take the feature half

    # --------------------------- internal: attention updates ---------------------------

    @torch.no_grad()
    def _temporal_update_all(self, t_now: int | float):
        """
        For each (cam, id) that exists at t_now, update its feat with temporal attention
        over up to 'temporal_k' previous frames in the SAME camera.
        """
        cur_frame = self._find_frame_node(t_now)
        if cur_frame is None:
            return

        for cam, store in cur_frame["views"].items():
            for tid, rec_now in store.items():
                # gather up to K previous feats for (cam, tid)
                prev_feats, deltas = self._gather_prev_feats(cam, tid, t_now, self.temporal_k)#变成列表
                if len(prev_feats) == 0 or rec_now.get("feat") is None:
                    continue

                # run attention with decay
                feat_now = rec_now["feat"]  # CPU fp32/fp16
                updated = self._attend_with_decay(
                    q=feat_now, keys=prev_feats, deltas=deltas
                )
                # write back updated feat (+ residual)
                rec_now["feat"] = updated

                # query_embed已删除，不需要更新

    def _gather_prev_feats(self, cam: str, tid: int, t_now, K: int) -> Tuple[List[torch.Tensor], List[float]]:

        prev_feats, deltas = [], []
        count = 0
        for fr in reversed(self.deque):
            # 不再跳过当前帧，而是将其包含在内（delta=0）
            view = fr["views"].get(cam, {})
            if tid in view and view[tid].get("feat") is not None:
                prev_feats.append(view[tid]["feat"])
                deltas.append(abs(float(t_now) - float(fr["t"])))  # 当前帧: delta=0
                count += 1
                if count >= K:
                    break
        return prev_feats, deltas



    # --------------------------- attention kernels ---------------------------

    def _attend_with_decay(self, q: torch.Tensor, keys: List[torch.Tensor], deltas: List[float]) -> torch.Tensor:
        """
        q:    CPU tensor (H) or device tensor; will be moved to runtime_device
        keys: list of CPU tensors (H)
        deltas: list of time gaps (>=0), same length as keys
        Return updated feature (H) tensor on CPU.
        """
        assert len(keys) == len(deltas) and len(keys) > 0
        q = _ensure_tensor(q, self.runtime_device, dtype=torch.float32).unsqueeze(0)  # (1,H)
        K = torch.stack([_ensure_tensor(k, self.runtime_device, dtype=torch.float32) for k in keys], dim=0)  # (N,H)

        # projections
        qh = self.q_proj(q)                     # (1,H)
        kh = self.k_proj(K)                     # (N,H)
        vh = self.v_proj(K)                     # (N,H)

        # multi-head split
        H = self.hidden_dim
        h = self.num_heads
        dh = H // h
        qh = qh.view(1, h, dh)                  # (1,h,dh)
        kh = kh.view(K.size(0), h, dh)          # (N,h,dh)
        vh = vh.view(K.size(0), h, dh)          # (N,h,dh)

        # scaled dot-product logits: (h,N)
        logits = (qh * kh).sum(dim=-1) / math.sqrt(dh)   # broadcast (1,h,dh)*(N,h,dh)->(N,h,dh)->(h,N)
        logits = logits.transpose(0,1)                   # (h,N)

        # add log-decay prior: log(w) where w = exp(-alpha * dt)
        # so adding (-alpha*dt) directly to logits biases attention to recent frames
        with torch.no_grad():
            dt = torch.tensor(deltas, device=self.runtime_device, dtype=torch.float32)  # (N,)
            log_w = - self.decay_alpha * dt                                            # (N,)
        logits = logits + log_w.unsqueeze(0)  # (h,N)

        attn = self.softmax(logits)           # (h,N)
        attn = self.attn_drop(attn)

        # weighted sum
        out = torch.einsum("hN,NhD->hD", attn, vh)    # (h,dh)
        out = out.reshape(1, H)                       # (1,H)
        out = self.out_proj(out)                      # (1,H)

        # residual + ffn
        base = q.view(1, H)
        out = self.norm1(base + out)
        out2 = self.ffn(out)
        out = self.norm2(out + out2)                  # (1,H)
        return out.squeeze(0).detach().cpu()

    def _attend_simple(self, q: torch.Tensor, keys: List[torch.Tensor]) -> torch.Tensor:
        """
        Cross-view attention without decay (same timestamp).
        q: (H) on device
        keys: list of (H) on device
        """
        K = torch.stack(keys, dim=0)                   # (N,H)
        qh = self.q_proj(q.unsqueeze(0))               # (1,H)
        kh = self.k_proj(K)                            # (N,H)
        vh = self.v_proj(K)                            # (N,H)

        H = self.hidden_dim
        h = self.num_heads
        dh = H // h
        qh = qh.view(1, h, dh)                         # (1,h,dh)
        kh = kh.view(K.size(0), h, dh)                 # (N,h,dh)
        vh = vh.view(K.size(0), h, dh)                 # (N,h,dh)

        logits = (qh * kh).sum(dim=-1) / math.sqrt(dh) # (N,h) -> transpose
        logits = logits.transpose(0,1)                 # (h,N)
        attn = self.softmax(logits)
        attn = self.attn_drop(attn)

        out = torch.einsum("hN,NhD->hD", attn, vh)     # (h,dh)
        out = out.reshape(1, H)                        # (1,H)
        out = self.out_proj(out)

        base = q.unsqueeze(0)                          # (1,H)
        out = self.norm1(base + out)
        out2 = self.ffn(out)
        out = self.norm2(out + out2)
        return out.squeeze(0)

    
    @torch.no_grad()
    def _reid_update_all_at_t(self, t_now: int | float, reid_features: Dict[Tuple[str, int], torch.Tensor]):
        """
        使用ReID特征更新query特征
        
        Args:
            t_now: 当前时间戳
            reid_features: {(view, id): reid_feat} - ReID特征字典
        """
        cur_frame = self._find_frame_node(t_now)
        if cur_frame is None:
            return
        
        for cam, store in cur_frame["views"].items():
            for tid, rec_now in store.items():
                # 查找对应的ReID特征
                key = (cam, tid)
                if key not in reid_features or rec_now.get("feat") is None:
                    continue
                
                reid_feat = reid_features[key]  # ReID特征
                query_feat = rec_now["feat"]  # 当前query特征
                
                # 将ReID特征投影到query特征空间并更新
                # 简单方式：加权融合
                reid_feat_cpu = _to_cpu(reid_feat)
                query_feat_tensor = _ensure_tensor(query_feat, self.runtime_device, dtype=torch.float32)
                reid_feat_tensor = _ensure_tensor(reid_feat_cpu, self.runtime_device, dtype=torch.float32)
                
                # 如果维度不匹配，需要投影
                if reid_feat_tensor.shape[-1] != self.hidden_dim:
                    # 延迟初始化reid_proj
                    if self.reid_proj is None:
                        self.reid_proj = nn.Linear(reid_feat_tensor.shape[-1], self.hidden_dim).to(self.runtime_device)
                    reid_feat_tensor = self.reid_proj(reid_feat_tensor)
                
                # 加权融合：query_feat = (1 - w) * query_feat + w * reid_feat
                updated_feat = (1 - self.reid_update_weight) * query_feat_tensor + \
                              self.reid_update_weight * reid_feat_tensor
                
                # 写回更新后的特征
                rec_now["feat"] = updated_feat.detach().cpu().to(dtype=rec_now["feat"].dtype)


