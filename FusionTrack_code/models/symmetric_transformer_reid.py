
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricTransformerReID(nn.Module):
    
    def __init__(self,
                 query_dim=256,
                 embed_dim=768,
                 num_classes=1677,
                 encoder_depth=4,
                 decoder_depth=2,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 consistency_loss_type="cosine",
                 consistency_weight=0.1):
        super().__init__()
        
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.consistency_loss_type = consistency_loss_type
        self.consistency_weight = consistency_weight
        
        print("="*80)
        print("对称Transformer ReID模型配置:")
        print(f"  输入维度: {query_dim}")
        print(f"  嵌入维度: {embed_dim}")
        print(f"  编码器深度: {encoder_depth}")
        print(f"  解码器深度: {decoder_depth}")
        print(f"  注意力头数: {num_heads}")
        print(f"  一致性损失: {consistency_loss_type}, 权重: {consistency_weight}")
        
        # 1. 输入投影：Query -> Embedding
        self.input_proj = nn.Sequential(
            nn.Linear(query_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 2. 位置编码（可选，对单向量不太重要）
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # 3. Encoder: 提取ReID特征
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # 4. ReID特征后处理
        self.reid_norm = nn.LayerNorm(embed_dim)
        self.reid_bottleneck = nn.BatchNorm1d(embed_dim)
        self.reid_bottleneck.bias.requires_grad_(False)
        
        # 5. ID分类器
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        
        # 6. Decoder: 重构Query特征（对称结构）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
        
        # 7. 输出投影：Embedding -> Query
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, query_dim)
        )
        
        # 8. 残差连接权重（可学习）
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        print(f"  参数量: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
        print("="*80)
    
    def forward_encoder(self, query):

        # 投影到嵌入空间
        x = self.input_proj(query)  # (N, embed_dim)
        x = x.unsqueeze(1)  # (N, 1, embed_dim) - Transformer需要序列维度
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 通过Encoder
        x = self.encoder(x)  # (N, 1, embed_dim)
        
        # 提取特征
        reid_feat = x.squeeze(1)  # (N, embed_dim)
        reid_feat = self.reid_norm(reid_feat)
        
        return reid_feat
    
    def forward_decoder(self, reid_feat, memory=None):
        # 将reid特征作为decoder的输入
        tgt = reid_feat.unsqueeze(1)  # (N, 1, embed_dim)
        
        # 如果有memory，用于cross-attention
        if memory is None:
            memory = tgt  # 自解码
        
        # 通过Decoder
        x = self.decoder(tgt, memory)  # (N, 1, embed_dim)
        x = x.squeeze(1)  # (N, embed_dim)
        
        # 投影回query空间
        reconstructed_query = self.output_proj(x)  # (N, query_dim)
        
        return reconstructed_query
    
    def compute_consistency_loss(self, original_query, reconstructed_query):
        """计算一致性损失"""
        if self.consistency_loss_type == "l1":
            loss = F.l1_loss(reconstructed_query, original_query)
        elif self.consistency_loss_type == "l2":
            loss = F.mse_loss(reconstructed_query, original_query)
        elif self.consistency_loss_type == "cosine":
            # 余弦相似度损失（推荐）
            cos_sim = F.cosine_similarity(original_query, reconstructed_query, dim=-1)
            loss = 1.0 - cos_sim.mean()
        elif self.consistency_loss_type == "combined":
            # L2 + 余弦
            l2 = F.mse_loss(reconstructed_query, original_query)
            cos_sim = F.cosine_similarity(original_query, reconstructed_query, dim=-1)
            cos_loss = 1.0 - cos_sim.mean()
            loss = l2 + cos_loss
        else:
            raise ValueError(f"Unknown loss type: {self.consistency_loss_type}")
        
        return loss * self.consistency_weight
    
    def forward(self, query_list, labels=None):
        # 1. 收集query
        query_feats = []
        for q in query_list:
            if isinstance(q, torch.Tensor):
                if q.dim() == 1:
                    query_feats.append(q)
                elif q.dim() == 2 and q.shape[0] > 0:
                    query_feats.append(q.mean(0))  # 序列取平均
        
        if len(query_feats) == 0:
            device = next(self.parameters()).device
            if self.training:
                return (torch.empty(0, self.num_classes, device=device),
                        torch.empty(0, self.embed_dim, device=device),
                        torch.tensor(0.0, device=device))
            else:
                return torch.empty(0, self.embed_dim, device=device)
        
        queries = torch.stack(query_feats)  # (N, query_dim)
        
        # 2. Encoder: Query -> ReID特征
        reid_features = self.forward_encoder(queries)  # (N, embed_dim)
        
        # 3. 计算一致性损失（如果训练）
        consistency_loss = torch.tensor(0.0, device=queries.device)
        if self.training:
            # Decoder: ReID特征 -> 重构Query
            reconstructed_queries = self.forward_decoder(reid_features)  # (N, query_dim)
            
            # 残差连接（可选）
            reconstructed_queries = reconstructed_queries + self.residual_weight * queries
            
            # 一致性损失
            consistency_loss = self.compute_consistency_loss(queries, reconstructed_queries)
        
        # 4. 训练模式：返回分类和损失
        if self.training:
            # BatchNorm（多样本）
            if len(query_feats) > 1:
                feat = self.reid_bottleneck(reid_features)
            else:
                feat = reid_features
            
            # ID分类
            cls_scores = self.classifier(feat)
            
            return cls_scores, reid_features, consistency_loss
        
        # 5. 推理模式：只返回特征
        else:
            if len(query_feats) > 1:
                return self.reid_bottleneck(reid_features)
            else:
                return reid_features


def build_symmetric_transformer_reid(config):
    """构建对称Transformer ReID模型"""
    model = SymmetricTransformerReID(
        query_dim=config.get("QUERY_DIM", config.get("HIDDEN_DIM", 256)),
        embed_dim=config.get("EMBED_DIM", 768),
        num_classes=config.get("NUM_CLASS", 1677),
        encoder_depth=config.get("REID_ENCODER_DEPTH", 4),
        decoder_depth=config.get("REID_DECODER_DEPTH", 2),
        num_heads=config.get("REID_NUM_HEADS", 8),
        mlp_ratio=config.get("REID_MLP_RATIO", 4.0),
        dropout=config.get("REID_DROPOUT", 0.1),
        consistency_loss_type=config.get("REID_CONSISTENCY_LOSS", "cosine"),
        consistency_weight=config.get("REID_CONSISTENCY_WEIGHT", 0.1)
    )
    return model

if __name__ == "__main__":
    # 配置
    config = {
        "QUERY_DIM": 256,
        "EMBED_DIM": 768,
        "NUM_CLASS": 1677,
        "REID_ENCODER_DEPTH": 4,
        "REID_DECODER_DEPTH": 2,
        "REID_NUM_HEADS": 8,
        "REID_CONSISTENCY_LOSS": "cosine",
        "REID_CONSISTENCY_WEIGHT": 0.1
    }
    
    # 创建模型
    model = build_symmetric_transformer_reid(config)
    model.train()
    
    # 模拟输入
    query_list = [torch.randn(256) for _ in range(32)]
    labels = torch.randint(0, 1677, (32,))
    
    # 前向传播
    cls_scores, reid_features, consistency_loss = model(query_list, labels)
    
    print(f"\n输出形状:")
    print(f"  cls_scores: {cls_scores.shape}")
    print(f"  reid_features: {reid_features.shape}")
    print(f"  consistency_loss: {consistency_loss.item():.4f}")
    
    # 训练损失
    id_loss = F.cross_entropy(cls_scores, labels)
    total_loss = id_loss + consistency_loss
    print(f"\n损失:")
    print(f"  ID loss: {id_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
