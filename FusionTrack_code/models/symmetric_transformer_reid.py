
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
        print("Symmetric Transformer ReID model config:")
        print(f"  Input dim: {query_dim}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Encoder depth: {encoder_depth}")
        print(f"  Decoder depth: {decoder_depth}")
        print(f"  Num attention heads: {num_heads}")
        print(f"  Consistency loss: {consistency_loss_type}, weight: {consistency_weight}")
        
        # 1. Input projection: Query -> Embedding
        self.input_proj = nn.Sequential(
            nn.Linear(query_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 2. Positional encoding (optional; less important for single vectors)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # 3. Encoder: extract ReID features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN is more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # 4. ReID feature post-processing
        self.reid_norm = nn.LayerNorm(embed_dim)
        self.reid_bottleneck = nn.BatchNorm1d(embed_dim)
        self.reid_bottleneck.bias.requires_grad_(False)
        
        # 5. ID classifier
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        
        # 6. Decoder: reconstruct Query features (symmetric)
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
        
        # 7. Output projection: Embedding -> Query
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, query_dim)
        )
        
        # 8. Learnable residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        print(f"  Params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
        print("="*80)
    
    def forward_encoder(self, query):

        # Project to embedding space
        x = self.input_proj(query)  # (N, embed_dim)
        x = x.unsqueeze(1)  # (N, 1, embed_dim) - Transformer expects sequence dim
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Pass through encoder
        x = self.encoder(x)  # (N, 1, embed_dim)
        
        # Extract features
        reid_feat = x.squeeze(1)  # (N, embed_dim)
        reid_feat = self.reid_norm(reid_feat)
        
        return reid_feat
    
    def forward_decoder(self, reid_feat, memory=None):
        # Use ReID features as decoder input
        tgt = reid_feat.unsqueeze(1)  # (N, 1, embed_dim)
        
        # If memory is provided, use for cross-attention
        if memory is None:
            memory = tgt  # self-decoding
        
        # Pass through decoder
        x = self.decoder(tgt, memory)  # (N, 1, embed_dim)
        x = x.squeeze(1)  # (N, embed_dim)
        
        # Project back to query space
        reconstructed_query = self.output_proj(x)  # (N, query_dim)
        
        return reconstructed_query
    
    def compute_consistency_loss(self, original_query, reconstructed_query):
        """Compute consistency loss."""
        if self.consistency_loss_type == "l1":
            loss = F.l1_loss(reconstructed_query, original_query)
        elif self.consistency_loss_type == "l2":
            loss = F.mse_loss(reconstructed_query, original_query)
        elif self.consistency_loss_type == "cosine":
            # cosine similarity loss (recommended)
            cos_sim = F.cosine_similarity(original_query, reconstructed_query, dim=-1)
            loss = 1.0 - cos_sim.mean()
        elif self.consistency_loss_type == "combined":
            # L2 + cosine
            l2 = F.mse_loss(reconstructed_query, original_query)
            cos_sim = F.cosine_similarity(original_query, reconstructed_query, dim=-1)
            cos_loss = 1.0 - cos_sim.mean()
            loss = l2 + cos_loss
        else:
            raise ValueError(f"Unknown loss type: {self.consistency_loss_type}")
        
        return loss * self.consistency_weight
    
    def forward(self, query_list, labels=None):
        # 1. Collect queries
        query_feats = []
        for q in query_list:
            if isinstance(q, torch.Tensor):
                if q.dim() == 1:
                    query_feats.append(q)
                elif q.dim() == 2 and q.shape[0] > 0:
                    query_feats.append(q.mean(0))  # average over sequence
        
        if len(query_feats) == 0:
            device = next(self.parameters()).device
            if self.training:
                return (torch.empty(0, self.num_classes, device=device),
                        torch.empty(0, self.embed_dim, device=device),
                        torch.tensor(0.0, device=device))
            else:
                return torch.empty(0, self.embed_dim, device=device)
        
        queries = torch.stack(query_feats)  # (N, query_dim)
        
        # 2. Encoder: Query -> ReID features
        reid_features = self.forward_encoder(queries)  # (N, embed_dim)
        
        # 3. Consistency loss (when training)
        consistency_loss = torch.tensor(0.0, device=queries.device)
        if self.training:
            # Decoder: ReID features -> reconstructed Query
            reconstructed_queries = self.forward_decoder(reid_features)  # (N, query_dim)
            
            # optional residual connection
            reconstructed_queries = reconstructed_queries + self.residual_weight * queries
            
            # consistency loss
            consistency_loss = self.compute_consistency_loss(queries, reconstructed_queries)
        
        # 4. Training: return classification and losses
        if self.training:
            # BatchNorm (multiple samples)
            if len(query_feats) > 1:
                feat = self.reid_bottleneck(reid_features)
            else:
                feat = reid_features
            
            # ID classification
            cls_scores = self.classifier(feat)
            
            return cls_scores, reid_features, consistency_loss
        
        # 5. Inference: return features only
        else:
            if len(query_feats) > 1:
                return self.reid_bottleneck(reid_features)
            else:
                return reid_features


def build_symmetric_transformer_reid(config):
    """Build symmetric Transformer ReID model."""
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
    # Configuration
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
    
    # Create model
    model = build_symmetric_transformer_reid(config)
    model.train()
    
    # Mock input
    query_list = [torch.randn(256) for _ in range(32)]
    labels = torch.randint(0, 1677, (32,))
    
    # Forward pass
    cls_scores, reid_features, consistency_loss = model(query_list, labels)
    
    print(f"\nOutput shapes:")
    print(f"  cls_scores: {cls_scores.shape}")
    print(f"  reid_features: {reid_features.shape}")
    print(f"  consistency_loss: {consistency_loss.item():.4f}")
    
    # Training loss
    id_loss = F.cross_entropy(cls_scores, labels)
    total_loss = id_loss + consistency_loss
    print(f"\nLosses:")
    print(f"  ID loss: {id_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
