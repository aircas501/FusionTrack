
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbones.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.vit_pytorch import trunc_normal_
from utils.utils import is_distributed, distributed_rank


# ==================== Mask-aware Transformer Block ====================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP module."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MaskedAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # Key: apply mask
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) for broadcasting
            # attention mask: set attention scores to -inf at padding positions
            # so softmax weights at those positions become 0
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            
            # create attention mask: valid tokens keep scores, padding tokens get -inf
            # valid token scores unchanged; padding token scores set to -inf
            attn_mask = torch.zeros_like(attn)
            attn_mask.masked_fill_(~mask, float('-inf'))  # fill padding positions with -inf
            
            attn = attn + attn_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):

        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ==================== Main Model ====================

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ReversibleDualLayerReID(nn.Module):

    
    def __init__(self, num_classes, config):
        super(ReversibleDualLayerReID, self).__init__()
        
        # ==================== Enforce consistent dims for true bidirectional symmetry ====================
        # Query and Embedding dims must match for weight-shared bidirectional mapping
        self.dim = 256  # unified dimension
        self.num_classes = num_classes
        
        print("="*80)
        print("Two-level weight-shared reversible TransReID model config:")
        print(f"  Unified dim: {self.dim} (Query, Embedding, ReID features)")
        print(f"  Num ID classes: {num_classes}")
        
        # ==================== Level 1: spatial mapping (linear or MLP) ====================
        self.use_mlp_mapping = config.get("USE_MLP_MAPPING", False)
        
        if self.use_mlp_mapping:
            # Option B: bidirectional mapping with MLP
            mlp_hidden_dim = config.get("MLP_HIDDEN_DIM", self.dim * 2)  # default 512
            mlp_layers = config.get("MLP_LAYERS", 2)  # default 2 layers
            dropout = config.get("REID_DROPOUT", 0.1)
            
            print(f"  [Mapping] MLP bidirectional mapping")
            print(f"    Forward MLP: {self.dim} -> {mlp_hidden_dim} -> {self.dim}")
            print(f"    Inverse MLP: {self.dim} -> {mlp_hidden_dim} -> {self.dim}")
            print(f"    Layers: {mlp_layers}, Dropout: {dropout}")
        
            # Build forward MLP
            self.forward_mlp = self._build_mlp(
                    input_dim=self.dim,
                    output_dim=self.dim,
                    hidden_dim=mlp_hidden_dim,
                    num_layers=mlp_layers,
                    dropout=dropout)
        
            # Build inverse MLP
            self.inverse_mlp = self._build_mlp(
                input_dim=self.dim,
                output_dim=self.dim,
                hidden_dim=mlp_hidden_dim,
                num_layers=mlp_layers,
                dropout=dropout
            )
            
            # Count parameters
            forward_params = sum(p.numel() for p in self.forward_mlp.parameters())
            inverse_params = sum(p.numel() for p in self.inverse_mlp.parameters())
            total_mlp_params = forward_params + inverse_params
            print(f"  MLP params: {total_mlp_params} (forward: {forward_params}, inverse: {inverse_params})")
            
        else:
            # Option A: single linear layer with shared weights
            print(f"  [Mapping] Weight-shared single linear layer")
            print(f"    Forward: y = xW^T + b_fwd")
            print(f"    Inverse: x' = yW + b_inv")
        
            self.shared_weight = nn.Parameter(torch.Tensor(self.dim, self.dim))
            self.shared_bias_fwd = nn.Parameter(torch.Tensor(self.dim))  # forward bias
            self.shared_bias_inv = nn.Parameter(torch.Tensor(self.dim))  # inverse bias
            
            # Initialize weights with Kaiming init
            nn.init.kaiming_uniform_(self.shared_weight, a=math.sqrt(5))
            nn.init.constant_(self.shared_bias_fwd, 0.0)
            nn.init.constant_(self.shared_bias_inv, 0.0)
            
            print(f"  Shared weight matrix shape: {self.dim}x{self.dim}")
            print(f"  Params: {self.dim * self.dim + 2 * self.dim} (~50% fewer than MLP)")
        
        # ==================== Level 2: TransReID feature extraction ====================
        # TransReID config
        depth = config.get("REID_TRANSFORMER_DEPTH", 2)
        num_heads = config.get("REID_NUM_HEADS", 8)
        mlp_ratio = config.get("REID_MLP_RATIO", 4.0)
        qkv_bias = config.get("REID_QKV_BIAS", True)
        drop_rate = config.get("REID_DROPOUT", 0.1)
        attn_drop_rate = config.get("REID_ATTN_DROPOUT", 0.1)
        drop_path_rate = config.get("REID_DROP_PATH", 0.1)
        
        print(f"  TransReID depth: {depth}")
        print(f"  Num attention heads: {num_heads}")
        
        # Positional encoding and cls_token (unified dim)
        # Note: pos_embed length must match sequence length
        max_seq_len = config.get("MAX_QUERY_SEQ_LEN", 10)  # max sequence length
        self.max_seq_len = max_seq_len
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, self.dim))  # cls + seq_len
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        
        print(f"  Max sequence length: {max_seq_len}")
        print(f"  Positional encoding shape: (1, {max_seq_len + 1}, {self.dim})")
        
        # Transformer encoder blocks (mask-aware MaskedBlock)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MaskedBlock(
                dim=self.dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(self.dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # ==================== ID classifier ====================
        self.bottleneck = nn.BatchNorm1d(self.dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.ID_LOSS_TYPE = config.get("ID_LOSS_TYPE", "softmax")
        if self.ID_LOSS_TYPE == 'arcface':
            print(f"  Using ArcFace classifier")
            self.classifier = Arcface(self.dim, self.num_classes,
                                      s=config.get("COSINE_SCALE", 64), 
                                      m=config.get("COSINE_MARGIN", 0.3))
        elif self.ID_LOSS_TYPE == 'cosface':
            print(f"  Using CosFace classifier")
            self.classifier = Cosface(self.dim, self.num_classes,
                                      s=config.get("COSINE_SCALE", 64), 
                                      m=config.get("COSINE_MARGIN", 0.35))
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print(f"  Using AMSoftmax classifier")
            self.classifier = AMSoftmax(self.dim, self.num_classes,
                                        s=config.get("COSINE_SCALE", 64), 
                                        m=config.get("COSINE_MARGIN", 0.3))
        elif self.ID_LOSS_TYPE == 'circle':
            print(f"  Using CircleLoss classifier")
            self.classifier = CircleLoss(self.dim, self.num_classes,
                                        s=config.get("COSINE_SCALE", 64), 
                                        m=config.get("COSINE_MARGIN", 0.25))
        else:
            print(f"  Using Softmax classifier")
            self.classifier = nn.Linear(self.dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        
        # ==================== Reversibility loss config ====================
        # Use MSE to constrain numerical reconstruction (recommended for weight-shared reversible mapping)
        self.use_reversibility_loss = config.get("USE_REVERSIBILITY_LOSS", True)  # main switch
        self.reversibility_weight1 = config.get("REVERSIBILITY_WEIGHT1", 0.1)  # Query↔Embedding
        self.reversibility_weight2 = config.get("REVERSIBILITY_WEIGHT2", 0.1)  # ReID feature stability
        
        if self.use_reversibility_loss:
            print(f"  Reversibility loss: ENABLED")
            print(f"     - Loss type: MSE (L2)")
            print(f"     - Level 1 weight (Query<->Embedding): {self.reversibility_weight1}")
            print(f"     - Level 2 weight (ReID feature stability): {self.reversibility_weight2}")
        else:
            print(f"  Reversibility loss: DISABLED")
            print(f"     - No reconstruction loss; mapping unconstrained")
        print("="*80)
    
    def _build_mlp(self, input_dim, output_dim, hidden_dim, num_layers, dropout=0.1):
        if num_layers == 1:
            # Single layer: direct mapping
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        layers = []
            
            # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
            
            # Middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            
            # Output layer
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        ])
            
        mlp = nn.Sequential(*layers)
        mlp.apply(weights_init_kaiming)
        return mlp
    
    def space_mapping(self, x, inverse=False):

        if self.use_mlp_mapping:
            # Use MLP
            if not inverse:
                return self.forward_mlp(x)
            else:
                return self.inverse_mlp(x)
        else:
            # Use weight-shared linear transform
            if not inverse:
                # Forward: Query -> Embedding
                # F.linear(x, W) is equivalent to xW^T
                return F.linear(x, self.shared_weight, self.shared_bias_fwd)
            else:
                # Inverse: Embedding -> Query (transposed weights)
                # F.linear(x, W^T) is equivalent to xW
                return F.linear(x, self.shared_weight.t(), self.shared_bias_inv)
    
    def _detect_valid_queries(self, queries):

        # Compute L1 norm per query (sum of absolute values across dims)
        # queries: (N, T, dim) -> l1_norms: (N, T)
        l1_norms = queries.abs().sum(dim=-1)
        
        # Treat as zero vector (missing frame) if L1 norm is below threshold
        # mask: True=valid, False=zero vector
        threshold = 1e-6
        mask = l1_norms > threshold  # (N, T)
        
        return mask
    
    def forward_transform(self, query_features, query_mask=None):
        # Normalize input shape to (1, T, dim)
        if query_features.dim() == 2:
            # (T, dim) -> (1, T, dim): single-track sequence
            query_features = query_features.unsqueeze(0)
            if query_mask is not None:
                query_mask = query_mask.unsqueeze(0)  # (T,) -> (1, T)
        
        N, T, _ = query_features.shape  # N should be 1 (single track)
        
        # Check sequence length
        if T > self.max_seq_len:
            print(f"Warning: sequence length {T} exceeds max {self.max_seq_len}, truncating")
            query_features = query_features[:, :self.max_seq_len, :]
            if query_mask is not None:
                query_mask = query_mask[:, :self.max_seq_len]
            T = self.max_seq_len
        
        # Step 1: Query seq -> Embedding seq (shared-weight forward mapping)
        # Reshape (N, T, dim) to (N*T, dim) for mapping
        queries_flat = query_features.reshape(N * T, self.dim)
        embeddings_flat = self.space_mapping(queries_flat, inverse=False)  # (N*T, dim)
        embeddings = embeddings_flat.reshape(N, T, self.dim)  # (N, T, dim)
        
        # Step 2: Embedding seq -> ReID features (via TransReID)
        # Add cls_token
        cls_tokens = self.cls_token.expand(N, -1, -1)  # (N, 1, dim)
        x = torch.cat((cls_tokens, embeddings), dim=1)  # (N, 1+T, dim)
        
        # Build attention mask: cls_token always visible; queries follow query_mask
        if query_mask is not None:
            # cls_token is always True (visible)
            cls_mask = torch.ones(N, 1, dtype=torch.bool, device=query_mask.device)
            # Concatenate: [cls_token(True), query_mask]
            attention_mask = torch.cat([cls_mask, query_mask], dim=1)  # (N, 1+T)
        else:
            # No mask: all tokens visible
            attention_mask = None
        
        # Add positional encoding (truncate to actual length)
        pos_embed_used = self.pos_embed[:, :1+T, :]  # (1, 1+T, dim)
        x = x + pos_embed_used
        x = self.pos_drop(x)
        
        # Pass through Transformer blocks with mask to ignore empty features
        for blk in self.blocks:
            x = blk(x, mask=attention_mask)
        
        x = self.norm(x)
        
        reid_features = x[:, 0]  # Shape: (N, dim) -> (1, 256)
        
        return embeddings, reid_features
    
    def get_cls_token_feature(self, query_features, query_mask=None):

        # Normalize input shape
        if query_features.dim() == 2:
            query_features = query_features.unsqueeze(0)
            if query_mask is not None:
                query_mask = query_mask.unsqueeze(0)
        
        N, T, _ = query_features.shape
        
        # Forward pass to cls_token
        queries_flat = query_features.reshape(N * T, self.dim)
        embeddings_flat = self.space_mapping(queries_flat, inverse=False)
        embeddings = embeddings_flat.reshape(N, T, self.dim)
        
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, embeddings), dim=1)
        
        if query_mask is not None:
            cls_mask = torch.ones(N, 1, dtype=torch.bool, device=query_mask.device)
            attention_mask = torch.cat([cls_mask, query_mask], dim=1)
        else:
            attention_mask = None
        
        pos_embed_used = self.pos_embed[:, :1+T, :]
        x = x + pos_embed_used
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, mask=attention_mask)
        
        x = self.norm(x)
        
        # Extract cls_token
        cls_feat = x[:, 0]  # (N, dim)
        
        return cls_feat
    
    def forward(self, query_input, labels=None):
        # 1. Normalize input to (N, T, dim)
        if isinstance(query_input, list):
            # Legacy interface: list of tensors
            # Use model device so all tensors are co-located
            model_device = next(self.parameters()).device
            
            query_seqs = []
            for q in query_input:
                if isinstance(q, torch.Tensor):
                        # Move tensor to model device
                        q = q.to(model_device)
                    
                if q.dim() == 1:
                        query_seqs.append(q.unsqueeze(0))  # (dim,) -> (1, dim)
                elif q.dim() == 2:
                        query_seqs.append(q)  # keep (T, dim) as-is
        
            if len(query_seqs) == 0:
                device = next(self.parameters()).device
                if self.training:
                    return (torch.empty(0, self.dim, device=device),
                            torch.tensor(0.0, device=device),
                            torch.tensor(0.0, device=device))
                else:
                    return torch.empty(0, self.dim, device=device)
                
                # Pad to same length and stack
            max_len = max(q.shape[0] for q in query_seqs)
            padded_seqs = []
            for q in query_seqs:
                if q.shape[0] < max_len:
                    # Pad with zeros
                    pad = torch.zeros(max_len - q.shape[0], q.shape[1], device=q.device)
                    q = torch.cat([q, pad], dim=0)
                padded_seqs.append(q)
        
            queries = torch.stack(padded_seqs,dim=1)  # (N, T, dim)
        
        elif isinstance(query_input, torch.Tensor):
            # Input is a tensor
            # Move input tensor to model device
            model_device = next(self.parameters()).device
            query_input = query_input.to(model_device)
            
            if query_input.dim() == 2:
                queries = query_input.unsqueeze(1)  # (N, dim) -> (N, 1, dim)
            elif query_input.dim() == 3:
                queries = query_input  # (N, T, dim)
            else:
                raise ValueError(f"Unsupported query_input shape: {query_input.shape}")
        else:
            raise TypeError(f"Unsupported query_input type: {type(query_input)}")
        
        N, T, _ = queries.shape
        
        # 1.5. Auto-detect empty features and build mask
        # Empty feature: all-zero vector (missing frame from MemoryBank)
        # mask: True=valid query, False=empty/padding
        query_mask = self._detect_valid_queries(queries)  # (N, T) validity mask
        
        # 2. Forward: Query seq -> Embedding seq -> ReID features
        embeddings, reid_features = self.forward_transform(queries, query_mask=query_mask)  # embeddings: (N, T, dim), reid_features: (N, dim)
        
        # 3. Compute reversibility loss (training)
        reversibility_loss1 = torch.tensor(0.0, device=queries.device)
        reversibility_loss2 = torch.tensor(0.0, device=queries.device)
        
        # Compute reversibility loss only in training when enabled
        if self.training and self.use_reversibility_loss:
            # ==================== Level 1: Query seq <-> Embedding seq ====================
            # Map Embedding seq back to Query space via inverse mapping (W)
            # embeddings: (N, T, dim) -> flatten -> (N*T, dim) -> mapping -> (N*T, dim) -> reshape -> (N, T, dim)
            embeddings_flat = embeddings.reshape(N * T, self.dim)
            reconstructed_queries_flat = self.space_mapping(embeddings_flat, inverse=True)
            reconstructed_queries = reconstructed_queries_flat.reshape(N, T, self.dim)
            
            # Only compute reconstruction loss on valid queries (mask=True)
            # Reconstructing zero vectors is meaningless; focus on valid frames
            if query_mask is not None:
                # query_mask: (N, T), True=valid query
                # Expand to (N, T, dim) for element-wise masking
                mask_expanded = query_mask.unsqueeze(-1).float()  # (N, T, 1)
                
                # Per-position MSE, averaged over valid queries only
                diff_squared = (reconstructed_queries - queries) ** 2  # (N, T, dim)
                masked_diff = diff_squared * mask_expanded  # keep errors for valid queries only
                
                # Average: sum(masked_diff) / number of valid elements
                num_valid_elements = mask_expanded.sum() * self.dim
                if num_valid_elements > 0:
                    reversibility_loss1 = (masked_diff.sum() / num_valid_elements) * self.reversibility_weight1
                else:
                    # Edge case: all-zero vectors -> zero loss
                    reversibility_loss1 = torch.tensor(0.0, device=queries.device)
            else:
                # No mask: use all queries (legacy behavior)
                reversibility_loss1 = F.mse_loss(reconstructed_queries, queries) * self.reversibility_weight1
            
            # ==================== Level 2: ReID feature stability check ====================
            # Forward (W^T) then inverse (W) on ReID features; check reconstruction
            # Validates ReID feature stability in the shared weight space
            mapped_reid = self.space_mapping(reid_features, inverse=False)  # (N, dim)
            reconstructed_reid = self.space_mapping(mapped_reid, inverse=True)  # (N, dim)
            reversibility_loss2 = F.mse_loss(reconstructed_reid, reid_features) * self.reversibility_weight2
        
        # 4. Training: return features and reversibility losses
        # ID and triplet losses computed externally (calculate_reid_loss_with_current_and_memory)
        if self.training:
            return reid_features, reversibility_loss1, reversibility_loss2
        
        # 5. Inference: return features only
        else:
            return self.bottleneck(reid_features)
    
    def load_param(self, trained_path):
        """Load pretrained weights."""
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print(f'Loading pretrained model from {trained_path}')
    
    def load_param_finetune(self, model_path):
        """Load pretrained model for fine-tuning."""
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print(f'Loading pretrained model for finetuning from {model_path}')


def build_reversible_reid(config):
    """Build weight-shared bidirectional reversible TransReID model."""
    model = ReversibleDualLayerReID(config["NUM_CLASS"], config)
    
    print('===========Building Weight-Shared Reversible TransReID Model===========')
    if config.get("AVAILABLE_GPUS") is not None and config.get("DEVICE") == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config.get("DEVICE", "cpu")))
    
    return model
