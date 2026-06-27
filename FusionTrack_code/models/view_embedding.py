
import torch
import torch.nn as nn
import math


class ViewEmbedding(nn.Module):

    def __init__(self, num_views: int, embed_dim: int):

        super(ViewEmbedding, self).__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        
        # 可学习的视角编码
        self.view_embedding = nn.Embedding(num_views, embed_dim)
        
        # 初始化
        nn.init.normal_(self.view_embedding.weight, std=0.02)
    
    def forward(self, view_ids: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:

        B = view_ids.shape[0]
        H, W = spatial_shape
        
        # 获取视角编码 (B, embed_dim)
        view_emb = self.view_embedding(view_ids)  # (B, embed_dim)
        
        # 广播到空间维度 (B, embed_dim, H, W)
        view_embed = view_emb.unsqueeze(-1).unsqueeze(-1)  # (B, embed_dim, 1, 1)
        view_embed = view_embed.expand(B, self.embed_dim, H, W)  # (B, embed_dim, H, W)
        
        return view_embed


def build_view_embedding(config: dict):

    num_views = config.get("VIEW_POINT", 4)
    # position embedding的维度是 HIDDEN_DIM（2*num_pos_feats）
    embed_dim = config.get("HIDDEN_DIM", 256)
    
    return ViewEmbedding(num_views=num_views, embed_dim=embed_dim)

