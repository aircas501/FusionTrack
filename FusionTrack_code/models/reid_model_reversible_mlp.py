
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbones.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.vit_pytorch import trunc_normal_
from utils.utils import is_distributed, distributed_rank

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
        print("双层权重共享可逆ReID模型配置（简化版 - 直接对Query做MLP）:")
        print(f"  统一维度: {self.dim} (Query、ReID特征)")
        print(f"  ID类别数: {num_classes}")
        print(f"  核心流程: Query序列 → ReID MLP(逐帧) → 平均 → 最终ReID特征")
        
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
        self.reid_mlp = self._build_mlp(
                input_dim=self.dim,
                output_dim=self.dim,
                hidden_dim=self.dim * 2,
                num_layers=2,
                dropout=0.1
            )
        # ==================== Level 2: ReID特征提取 ====================
        # 使用MLP对单帧embedding提取ReID特征
        reid_mlp_hidden_dim = config.get("REID_MLP_HIDDEN_DIM", self.dim * 2)  # 默认512
        reid_mlp_layers = config.get("REID_MLP_LAYERS", 2)  # 默认2层
        dropout = config.get("REID_DROPOUT", 0.1)
        
        print(f"  [ReID特征提取] 使用MLP逐帧提取")
        print(f"    MLP结构: {self.dim} → {reid_mlp_hidden_dim} → {self.dim}")
        print(f"    层数: {reid_mlp_layers}, Dropout: {dropout}")
        
        self.reid_mlp = self._build_mlp(
            input_dim=self.dim,
            output_dim=self.dim,
            hidden_dim=reid_mlp_hidden_dim,
            num_layers=reid_mlp_layers,
            dropout=dropout
        )
        
        reid_mlp_params = sum(p.numel() for p in self.reid_mlp.parameters())
        print(f"  ReID MLP参数量: {reid_mlp_params}")
        
        max_seq_len = config.get("MAX_QUERY_SEQ_LEN", 10)  # 最大序列长度
        self.max_seq_len = max_seq_len
        print(f"  最大序列长度: {max_seq_len}")
        
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

        # 处理输入维度：统一为 (N, T, dim)
        if query_features.dim() == 2:
            query_features = query_features.unsqueeze(0)  # (T, dim) -> (1, T, dim)
            if query_mask is not None:
                query_mask = query_mask.unsqueeze(0)  # (T,) -> (1, T)
        
        N, T, _ = query_features.shape
        
        # 检查序列长度
        if T > self.max_seq_len:
            print(f"警告：序列长度 {T} 超过最大长度 {self.max_seq_len}，将截断")
            query_features = query_features[:, :self.max_seq_len, :]
            if query_mask is not None:
                query_mask = query_mask[:, :self.max_seq_len]
            T = self.max_seq_len
        
        # Step 1: 对每个Query，直接使用ReID MLP提取单帧ReID特征
        # query_features: (N, T, dim) -> flatten -> (N*T, dim) -> reid_mlp -> (N*T, dim)
        queries_flat = query_features.reshape(N * T, self.dim)  # (N*T, dim)
        per_frame_reid_features_flat = self.reid_mlp(queries_flat)  # (N*T, dim)
        per_frame_reid_features = per_frame_reid_features_flat.reshape(N, T, self.dim)  # (N, T, dim)
        
        # Step 2: 对有效帧的ReID特征进行平均池化
        if query_mask is not None:
            # query_mask: (N, T), True表示有效query
            mask_float = query_mask.unsqueeze(-1).float()  # (N, T, 1)
            
            # 加权求和：只累加有效帧的ReID特征
            masked_reid_features = per_frame_reid_features * mask_float  # (N, T, dim)
            sum_reid_features = masked_reid_features.sum(dim=1)  # (N, dim)
            
            # 除以有效帧数量
            num_valid = mask_float.sum(dim=1).clamp(min=1)  # (N, 1)，至少为1避免除0
            reid_features = sum_reid_features / num_valid  # (N, dim)
        else:
            # 没有mask，对所有帧平均池化
            reid_features = per_frame_reid_features.mean(dim=1)  # (N, dim)
        
        return per_frame_reid_features, reid_features
    
    def get_cls_token_feature(self, query_features, query_mask=None):

        # 直接使用forward_transform获取ReID特征
        _, reid_features = self.forward_transform(query_features, query_mask)  # (N, dim)
        
        # 经过bottleneck得到分类特征
        cls_feat = self.bottleneck(reid_features)  # (N, dim)
        
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
        
        # 2. 正向变换：Query序列 → 单帧ReID特征 → 平均得到最终ReID特征
        per_frame_reid_features, reid_features = self.forward_transform(queries, query_mask=query_mask)  
        # per_frame_reid_features: (N, T, dim) - 每帧的ReID特征
        # reid_features: (N, dim) - 平均后的最终ReID特征
        
        # 3. 计算可逆性损失（训练模式）
        reversibility_loss1 = torch.tensor(0.0, device=queries.device)
        reversibility_loss2 = torch.tensor(0.0, device=queries.device)
        
        # ⭐ 只在训练模式 且 开关打开时 计算可逆性损失
        if self.training and self.use_reversibility_loss:
            # ==================== Level 1: Query序列 ↔ per_frame_reid_features ====================
            # 思路：验证单帧ReID特征能否重构回原始Query
            # per_frame_reid_features: (N, T, dim) -> flatten -> (N*T, dim)
            per_frame_reid_flat = per_frame_reid_features.reshape(N * T, self.dim)
            
            # 使用反向映射尝试重构Query
            reconstructed_queries_flat = self.space_mapping(per_frame_reid_flat, inverse=True)
            reconstructed_queries = reconstructed_queries_flat.reshape(N, T, self.dim)
            
            # ⭐ 只计算有效query（mask=True）的重构损失
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
            
            # ==================== Level 2: 最终ReID特征的稳定性验证 ====================
            # 思路：将平均后的ReID特征先正向映射，再反向映射，验证稳定性
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

