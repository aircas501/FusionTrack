
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbones.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.vit_pytorch import trunc_normal_
from utils.utils import is_distributed, distributed_rank


# ==================== 支持Mask的Transformer Block ====================

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
    """MLP模块"""
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

        # 计算attention score
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # ⭐ 关键：应用mask
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) 用于广播
            # attention mask: 对于padding位置，将attention score设为-inf
            # 这样softmax后对应位置的权重就是0
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            
            # 创建attention mask: True->0, False->-inf
            # 即：有效token的attention score保持不变，padding token的score设为-inf
            attn_mask = torch.zeros_like(attn)
            attn_mask.masked_fill_(~mask, float('-inf'))  # padding位置填充-inf
            
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


# ==================== 主模型 ====================

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
        
        # ==================== 强制维度一致以实现真正的双向对称 ====================
        # 为了实现权重共享的双向映射，Query和Embedding维度必须相同
        self.dim = 256  # 统一维度
        self.num_classes = num_classes
        
        print("="*80)
        print("双层权重共享可逆TransReID模型配置:")
        print(f"  统一维度: {self.dim} (Query、Embedding、ReID特征)")
        print(f"  ID类别数: {num_classes}")
        
        # ==================== Level 1: 空间映射层（可选：线性 or MLP）====================
        self.use_mlp_mapping = config.get("USE_MLP_MAPPING", False)
        
        if self.use_mlp_mapping:
            # 方案B：使用MLP进行双向映射
            mlp_hidden_dim = config.get("MLP_HIDDEN_DIM", self.dim * 2)  # 默认512
            mlp_layers = config.get("MLP_LAYERS", 2)  # 默认2层
            dropout = config.get("REID_DROPOUT", 0.1)
            
            print(f"  [映射方式] 使用MLP双向映射")
            print(f"    正向MLP: {self.dim} → {mlp_hidden_dim} → {self.dim}")
            print(f"    反向MLP: {self.dim} → {mlp_hidden_dim} → {self.dim}")
            print(f"    层数: {mlp_layers}, Dropout: {dropout}")
        
            # 构建正向MLP
            self.forward_mlp = self._build_mlp(
                    input_dim=self.dim,
                    output_dim=self.dim,
                    hidden_dim=mlp_hidden_dim,
                    num_layers=mlp_layers,
                    dropout=dropout)
        
            # 构建反向MLP
            self.inverse_mlp = self._build_mlp(
                input_dim=self.dim,
                output_dim=self.dim,
                hidden_dim=mlp_hidden_dim,
                num_layers=mlp_layers,
                dropout=dropout
            )
            
            # 计算参数量
            forward_params = sum(p.numel() for p in self.forward_mlp.parameters())
            inverse_params = sum(p.numel() for p in self.inverse_mlp.parameters())
            total_mlp_params = forward_params + inverse_params
            print(f"  MLP参数量: {total_mlp_params} (正向: {forward_params}, 反向: {inverse_params})")
            
        else:
            # 方案A：使用单层线性变换（权重共享）
            print(f"  [映射方式] 使用权重共享的单层线性变换")
            print(f"    正向: y = xW^T + b_fwd")
            print(f"    反向: x' = yW + b_inv")
        
            self.shared_weight = nn.Parameter(torch.Tensor(self.dim, self.dim))
            self.shared_bias_fwd = nn.Parameter(torch.Tensor(self.dim))  # 正向偏置
            self.shared_bias_inv = nn.Parameter(torch.Tensor(self.dim))  # 反向偏置
            
            # 初始化权重（使用Kaiming初始化）
            nn.init.kaiming_uniform_(self.shared_weight, a=math.sqrt(5))
            nn.init.constant_(self.shared_bias_fwd, 0.0)
            nn.init.constant_(self.shared_bias_inv, 0.0)
            
            print(f"  共享权重矩阵形状: {self.dim}×{self.dim}")
            print(f"  参数量: {self.dim * self.dim + 2 * self.dim} (相比MLP减少约50%)")
        
        # ==================== Level 2: TransReID特征提取 ====================
        # TransReID配置
        depth = config.get("REID_TRANSFORMER_DEPTH", 2)
        num_heads = config.get("REID_NUM_HEADS", 8)
        mlp_ratio = config.get("REID_MLP_RATIO", 4.0)
        qkv_bias = config.get("REID_QKV_BIAS", True)
        drop_rate = config.get("REID_DROPOUT", 0.1)
        attn_drop_rate = config.get("REID_ATTN_DROPOUT", 0.1)
        drop_path_rate = config.get("REID_DROP_PATH", 0.1)
        
        print(f"  TransReID深度: {depth}")
        print(f"  注意力头数: {num_heads}")
        
        # 位置编码和cls_token（维度改为统一的dim）
        # 注意：pos_embed 的长度需要适应序列长度
        max_seq_len = config.get("MAX_QUERY_SEQ_LEN", 10)  # 最大序列长度
        self.max_seq_len = max_seq_len
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, self.dim))  # cls + seq_len
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        
        print(f"  最大序列长度: {max_seq_len}")
        print(f"  位置编码形状: (1, {max_seq_len + 1}, {self.dim})")
        
        # Transformer Encoder Blocks（使用支持mask的MaskedBlock）
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
        
        # ==================== ID分类器 ====================
        self.bottleneck = nn.BatchNorm1d(self.dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.ID_LOSS_TYPE = config.get("ID_LOSS_TYPE", "softmax")
        if self.ID_LOSS_TYPE == 'arcface':
            print(f"  使用ArcFace分类器")
            self.classifier = Arcface(self.dim, self.num_classes,
                                      s=config.get("COSINE_SCALE", 64), 
                                      m=config.get("COSINE_MARGIN", 0.3))
        elif self.ID_LOSS_TYPE == 'cosface':
            print(f"  使用CosFace分类器")
            self.classifier = Cosface(self.dim, self.num_classes,
                                      s=config.get("COSINE_SCALE", 64), 
                                      m=config.get("COSINE_MARGIN", 0.35))
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print(f"  使用AMSoftmax分类器")
            self.classifier = AMSoftmax(self.dim, self.num_classes,
                                        s=config.get("COSINE_SCALE", 64), 
                                        m=config.get("COSINE_MARGIN", 0.3))
        elif self.ID_LOSS_TYPE == 'circle':
            print(f"  使用CircleLoss分类器")
            self.classifier = CircleLoss(self.dim, self.num_classes,
                                        s=config.get("COSINE_SCALE", 64), 
                                        m=config.get("COSINE_MARGIN", 0.25))
        else:
            print(f"  使用Softmax分类器")
            self.classifier = nn.Linear(self.dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        
        # ==================== 可逆性损失配置 ====================
        # 使用MSE损失以精确约束数值重构（推荐用于权重共享的可逆映射）
        self.use_reversibility_loss = config.get("USE_REVERSIBILITY_LOSS", True)  # 主开关
        self.reversibility_weight1 = config.get("REVERSIBILITY_WEIGHT1", 0.1)  # Query↔Embedding
        self.reversibility_weight2 = config.get("REVERSIBILITY_WEIGHT2", 0.1)  # ReID特征稳定性
        
        if self.use_reversibility_loss:
            print(f"  ✅ 可逆性损失: ENABLED")
            print(f"     - 损失类型: MSE (L2)")
            print(f"     - Level 1权重 (Query↔Embedding): {self.reversibility_weight1}")
            print(f"     - Level 2权重 (ReID特征稳定性): {self.reversibility_weight2}")
        else:
            print(f"  ⚠️  可逆性损失: DISABLED")
            print(f"     - 不计算重建损失，映射不受约束")
        print("="*80)
    
    def _build_mlp(self, input_dim, output_dim, hidden_dim, num_layers, dropout=0.1):
        if num_layers == 1:
            # 单层：直接映射
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        layers = []
            
            # 第一层
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
            
            # 中间层
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            
            # 输出层
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        ])
            
        mlp = nn.Sequential(*layers)
        mlp.apply(weights_init_kaiming)
        return mlp
    
    def space_mapping(self, x, inverse=False):

        if self.use_mlp_mapping:
            # 使用MLP
            if not inverse:
                return self.forward_mlp(x)
            else:
                return self.inverse_mlp(x)
        else:
            # 使用权重共享的线性变换
            if not inverse:
                # 正向变换：Query -> Embedding
                # F.linear(x, W) 等价于 xW^T
                return F.linear(x, self.shared_weight, self.shared_bias_fwd)
            else:
                # 反向变换：Embedding -> Query (使用转置权重)
                # F.linear(x, W^T) 等价于 xW
                return F.linear(x, self.shared_weight.t(), self.shared_bias_inv)
    
    def _detect_valid_queries(self, queries):

        # 计算每个query的L1范数（所有维度绝对值之和）
        # queries: (N, T, dim) -> l1_norms: (N, T)
        l1_norms = queries.abs().sum(dim=-1)
        
        # 判断：如果L1范数<阈值，则认为是0向量（缺失帧）
        # mask: True表示有效，False表示0向量
        threshold = 1e-6
        mask = l1_norms > threshold  # (N, T)
        
        return mask
    
    def forward_transform(self, query_features, query_mask=None):
        # 处理输入维度：统一为 (1, T, dim)
        if query_features.dim() == 2:
            # (T, dim) -> (1, T, dim)，理解为单个track的序列
            query_features = query_features.unsqueeze(0)
            if query_mask is not None:
                query_mask = query_mask.unsqueeze(0)  # (T,) -> (1, T)
        
        N, T, _ = query_features.shape  # N应该=1（单个track）
        
        # 检查序列长度
        if T > self.max_seq_len:
            print(f"警告：序列长度 {T} 超过最大长度 {self.max_seq_len}，将截断")
            query_features = query_features[:, :self.max_seq_len, :]
            if query_mask is not None:
                query_mask = query_mask[:, :self.max_seq_len]
            T = self.max_seq_len
        
        # Step 1: Query序列 → Embedding序列 (通过共享权重正向映射)
        # 将 (N, T, dim) reshape 为 (N*T, dim) 进行映射
        queries_flat = query_features.reshape(N * T, self.dim)
        embeddings_flat = self.space_mapping(queries_flat, inverse=False)  # (N*T, dim)
        embeddings = embeddings_flat.reshape(N, T, self.dim)  # (N, T, dim)
        
        # Step 2: Embedding序列 → ReID特征 (通过TransReID)
        # 添加cls_token
        cls_tokens = self.cls_token.expand(N, -1, -1)  # (N, 1, dim)
        x = torch.cat((cls_tokens, embeddings), dim=1)  # (N, 1+T, dim)
        
        # 构建attention mask：cls_token始终可见，query根据query_mask决定
        if query_mask is not None:
            # cls_token始终为True（可见）
            cls_mask = torch.ones(N, 1, dtype=torch.bool, device=query_mask.device)
            # 拼接：[cls_token(True), query_mask]
            attention_mask = torch.cat([cls_mask, query_mask], dim=1)  # (N, 1+T)
        else:
            # 没有mask，所有token都可见
            attention_mask = None
        
        # 添加位置编码（截取到实际长度）
        pos_embed_used = self.pos_embed[:, :1+T, :]  # (1, 1+T, dim)
        x = x + pos_embed_used
        x = self.pos_drop(x)
        
        # 通过Transformer blocks（传递mask，屏蔽空特征）
        for blk in self.blocks:
            x = blk(x, mask=attention_mask)
        
        x = self.norm(x)
        
        reid_features = x[:, 0]  # Shape: (N, dim) -> (1, 256)
        
        return embeddings, reid_features
    
    def get_cls_token_feature(self, query_features, query_mask=None):

        # 处理输入维度
        if query_features.dim() == 2:
            query_features = query_features.unsqueeze(0)
            if query_mask is not None:
                query_mask = query_mask.unsqueeze(0)
        
        N, T, _ = query_features.shape
        
        # 前向传播到cls_token
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
        
        # 提取cls_token
        cls_feat = x[:, 0]  # (N, dim)
        
        return cls_feat
    
    def forward(self, query_input, labels=None):
        # 1. 处理输入，统一为 (N, T, dim) 格式
        if isinstance(query_input, list):
            # 兼容旧接口：List of Tensors
            # 获取模型所在设备，确保所有tensor在同一设备
            model_device = next(self.parameters()).device
            
            query_seqs = []
            for q in query_input:
                if isinstance(q, torch.Tensor):
                        # 确保tensor在正确的设备上
                        q = q.to(model_device)
                    
                if q.dim() == 1:
                        query_seqs.append(q.unsqueeze(0))  # (dim,) -> (1, dim)
                elif q.dim() == 2:
                        query_seqs.append(q)  # (T, dim) 保持不变
        
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
            # 直接是Tensor
            # 确保输入tensor在模型所在设备
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
        
        # 1.5. 自动检测空特征并生成mask
        # 空特征定义：全0向量（从MemoryBank中缺失的帧）
        # mask: True表示有效query，False表示空特征/padding
        query_mask = self._detect_valid_queries(queries)  # (N, T)判断是不是有效的
        
        # 2. 正向变换：Query序列 → Embedding序列 → ReID特征
        embeddings, reid_features = self.forward_transform(queries, query_mask=query_mask)  # embeddings: (N, T, dim), reid_features: (N, dim)
        
        # 3. 计算可逆性损失（训练模式）
        reversibility_loss1 = torch.tensor(0.0, device=queries.device)
        reversibility_loss2 = torch.tensor(0.0, device=queries.device)
        
        # ⭐ 只在训练模式 且 开关打开时 计算可逆性损失
        if self.training and self.use_reversibility_loss:
            # ==================== Level 1: Query序列 ↔ Embedding序列 ====================
            # 将 Embedding序列 通过反向映射（W）回到 Query 空间
            # embeddings: (N, T, dim) -> flatten -> (N*T, dim) -> mapping -> (N*T, dim) -> reshape -> (N, T, dim)
            embeddings_flat = embeddings.reshape(N * T, self.dim)
            reconstructed_queries_flat = self.space_mapping(embeddings_flat, inverse=True)
            reconstructed_queries = reconstructed_queries_flat.reshape(N, T, self.dim)
            
            # ⭐ 改进：只计算有效query（mask=True）的重构损失
            # 原因：0向量重构成0向量没有意义，应该聚焦于有效帧的可逆性
            if query_mask is not None:
                # query_mask: (N, T), True表示有效query
                # 扩展到 (N, T, dim) 用于逐元素mask
                mask_expanded = query_mask.unsqueeze(-1).float()  # (N, T, 1)
                
                # 计算每个位置的MSE，然后只对有效query求平均
                diff_squared = (reconstructed_queries - queries) ** 2  # (N, T, dim)
                masked_diff = diff_squared * mask_expanded  # 只保留有效query的误差
                
                # 计算平均：sum(masked_diff) / 有效元素数量
                num_valid_elements = mask_expanded.sum() * self.dim
                if num_valid_elements > 0:
                    reversibility_loss1 = (masked_diff.sum() / num_valid_elements) * self.reversibility_weight1
                else:
                    # 边界情况：如果全是0向量，损失为0
                    reversibility_loss1 = torch.tensor(0.0, device=queries.device)
            else:
                # 没有mask，使用全部query计算损失（兼容旧逻辑）
                reversibility_loss1 = F.mse_loss(reconstructed_queries, queries) * self.reversibility_weight1
            
            # ==================== Level 2: ReID特征的稳定性验证 ====================
            # 思路：将 ReID特征 先正向映射（W^T），再反向映射（W），看是否能重构回来
            # 这验证了 ReID特征在共享权重空间中的稳定性
            mapped_reid = self.space_mapping(reid_features, inverse=False)  # (N, dim)
            reconstructed_reid = self.space_mapping(mapped_reid, inverse=True)  # (N, dim)
            reversibility_loss2 = F.mse_loss(reconstructed_reid, reid_features) * self.reversibility_weight2
        
        # 4. 训练模式：返回特征和可逆性损失
        # 注意：ID分类损失和Triplet损失在外部计算（calculate_reid_loss_with_current_and_memory）
        if self.training:
            return reid_features, reversibility_loss1, reversibility_loss2
        
        # 5. 推理模式：只返回特征
        else:
            return self.bottleneck(reid_features)
    
    def load_param(self, trained_path):
        """加载预训练权重"""
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print(f'Loading pretrained model from {trained_path}')
    
    def load_param_finetune(self, model_path):
        """加载预训练模型用于微调"""
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print(f'Loading pretrained model for finetuning from {model_path}')


def build_reversible_reid(config):
    """构建权重共享双向可逆TransReID模型"""
    model = ReversibleDualLayerReID(config["NUM_CLASS"], config)
    
    print('===========Building Weight-Shared Reversible TransReID Model===========')
    if config.get("AVAILABLE_GPUS") is not None and config.get("DEVICE") == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config.get("DEVICE", "cpu")))
    
    return model
