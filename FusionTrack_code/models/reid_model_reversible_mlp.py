
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbones.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.vit_pytorch import trunc_normal_
from utils.utils import is_distributed, distributed_rank

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
        print("Two-level weight-shared reversible ReID model config (simplified - MLP on Query directly):")
        print(f"  Unified dim: {self.dim} (Query, ReID features)")
        print(f"  Num ID classes: {num_classes}")
        print(f"  Pipeline: Query seq -> ReID MLP (per-frame) -> mean -> final ReID feature")
        
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
        self.reid_mlp = self._build_mlp(
                input_dim=self.dim,
                output_dim=self.dim,
                hidden_dim=self.dim * 2,
                num_layers=2,
                dropout=0.1
            )
        # ==================== Level 2: ReID feature extraction ====================
        # Extract per-frame ReID features with MLP
        reid_mlp_hidden_dim = config.get("REID_MLP_HIDDEN_DIM", self.dim * 2)  # default 512
        reid_mlp_layers = config.get("REID_MLP_LAYERS", 2)  # default 2 layers
        dropout = config.get("REID_DROPOUT", 0.1)
        
        print(f"  [ReID extraction] Per-frame MLP extraction")
        print(f"    MLP: {self.dim} -> {reid_mlp_hidden_dim} -> {self.dim}")
        print(f"    Layers: {reid_mlp_layers}, Dropout: {dropout}")
        
        self.reid_mlp = self._build_mlp(
            input_dim=self.dim,
            output_dim=self.dim,
            hidden_dim=reid_mlp_hidden_dim,
            num_layers=reid_mlp_layers,
            dropout=dropout
        )
        
        reid_mlp_params = sum(p.numel() for p in self.reid_mlp.parameters())
        print(f"  ReID MLP params: {reid_mlp_params}")
        
        max_seq_len = config.get("MAX_QUERY_SEQ_LEN", 10)  # max sequence length
        self.max_seq_len = max_seq_len
        print(f"  Max sequence length: {max_seq_len}")
        
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

        # Normalize input shape to (N, T, dim)
        if query_features.dim() == 2:
            query_features = query_features.unsqueeze(0)  # (T, dim) -> (1, T, dim)
            if query_mask is not None:
                query_mask = query_mask.unsqueeze(0)  # (T,) -> (1, T)
        
        N, T, _ = query_features.shape
        
        # Check sequence length
        if T > self.max_seq_len:
            print(f"Warning: sequence length {T} exceeds max {self.max_seq_len}, truncating")
            query_features = query_features[:, :self.max_seq_len, :]
            if query_mask is not None:
                query_mask = query_mask[:, :self.max_seq_len]
            T = self.max_seq_len
        
        # Step 1: per-query per-frame ReID features via ReID MLP
        # query_features: (N, T, dim) -> flatten -> (N*T, dim) -> reid_mlp -> (N*T, dim)
        queries_flat = query_features.reshape(N * T, self.dim)  # (N*T, dim)
        per_frame_reid_features_flat = self.reid_mlp(queries_flat)  # (N*T, dim)
        per_frame_reid_features = per_frame_reid_features_flat.reshape(N, T, self.dim)  # (N, T, dim)
        
        # Step 2: average pool ReID features over valid frames
        if query_mask is not None:
            # query_mask: (N, T), True=valid query
            mask_float = query_mask.unsqueeze(-1).float()  # (N, T, 1)
            
            # Weighted sum over valid frames only
            masked_reid_features = per_frame_reid_features * mask_float  # (N, T, dim)
            sum_reid_features = masked_reid_features.sum(dim=1)  # (N, dim)
            
            # Divide by number of valid frames
            num_valid = mask_float.sum(dim=1).clamp(min=1)  # (N, 1), clamp to at least 1 to avoid div-by-zero
            reid_features = sum_reid_features / num_valid  # (N, dim)
        else:
            # No mask: average over all frames
            reid_features = per_frame_reid_features.mean(dim=1)  # (N, dim)
        
        return per_frame_reid_features, reid_features
    
    def get_cls_token_feature(self, query_features, query_mask=None):

        # Get ReID features via forward_transform
        _, reid_features = self.forward_transform(query_features, query_mask)  # (N, dim)
        
        # Classification features via bottleneck
        cls_feat = self.bottleneck(reid_features)  # (N, dim)
        
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
        
        # 2. Forward: Query seq -> per-frame ReID -> averaged final ReID
        per_frame_reid_features, reid_features = self.forward_transform(queries, query_mask=query_mask)  
        # per_frame_reid_features: (N, T, dim) per-frame ReID
        # reid_features: (N, dim) averaged final ReID
        
        # 3. Compute reversibility loss (training)
        reversibility_loss1 = torch.tensor(0.0, device=queries.device)
        reversibility_loss2 = torch.tensor(0.0, device=queries.device)
        
        # Compute reversibility loss only in training when enabled
        if self.training and self.use_reversibility_loss:
            # ==================== Level 1: Query seq <-> per_frame_reid_features ====================
            # Check whether per-frame ReID features reconstruct original Query
            # per_frame_reid_features: (N, T, dim) -> flatten -> (N*T, dim)
            per_frame_reid_flat = per_frame_reid_features.reshape(N * T, self.dim)
            
            # Reconstruct Query via inverse mapping
            reconstructed_queries_flat = self.space_mapping(per_frame_reid_flat, inverse=True)
            reconstructed_queries = reconstructed_queries_flat.reshape(N, T, self.dim)
            
            # Reconstruction loss on valid queries (mask=True) only
            if query_mask is not None:
                mask_expanded = query_mask.unsqueeze(-1).float()  # (N, T, 1)
                diff_squared = (reconstructed_queries - queries) ** 2  # (N, T, dim)
                masked_diff = diff_squared * mask_expanded
                
                num_valid_elements = mask_expanded.sum() * self.dim
                if num_valid_elements > 0:
                    reversibility_loss1 = (masked_diff.sum() / num_valid_elements) * self.reversibility_weight1
                else:
                    reversibility_loss1 = torch.tensor(0.0, device=queries.device)
            else:
                reversibility_loss1 = F.mse_loss(reconstructed_queries, queries) * self.reversibility_weight1
            
            # ==================== Level 2: final ReID feature stability check ====================
            # Forward then inverse on averaged ReID features; check stability
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

