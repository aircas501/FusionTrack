import os
import time
import torch
import torch.nn as nn
import torch.distributed

from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from models import build_model
from data import build_dataset, build_sampler, build_dataloader
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank, set_seed, is_main_process, \
    distributed_world_size
from models.simple_reid_model import build_simple_reid_model
from models.reid_model_reversible_mlp import build_reversible_reid
from models.reid_pool import build as build_reid_pool, ReIDPool
from models.memory_bank import MemoryBank
from models.loss.triplet_loss import build_triplet_loss
from models.loss.uncertainty_loss import build_uncertainty_loss
from models.loss.reversibility_weight import build_reversibility_weight_learner
from utils.nested_tensor import tensor_list_to_nested_tensor
from models.memotr import FusionTrack
from structures.track_instances import TrackInstances
from models.criterion import build as build_criterion, ClipCriterion
from models.utils import get_model, save_checkpoint, load_checkpoint
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from log.logger import Logger, ProgressLogger
from log.log import MetricLog
from models.utils import load_pretrained_model


def extract_multiframe_queries_from_memorybank(memory_bank, track_id, view_name, window_size=5, feat_dim=256):

    if memory_bank is None or not hasattr(memory_bank, 'deque'):
        # 如果没有MemoryBank，返回window_size个0向量
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return [torch.zeros(feat_dim, device=device) for _ in range(window_size)]
    
    # 从MemoryBank的deque中提取最近window_size帧
    # deque的最后一个元素是最新的帧
    recent_frames = list(memory_bank.deque)[-window_size:] if len(memory_bank.deque) >= window_size else list(memory_bank.deque)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化固定长度的列表，全部填充0向量
    history_queries = [torch.zeros(feat_dim, device=device) for _ in range(window_size)]
    
    # 计算起始位置：如果MemoryBank不足window_size帧，从后面开始填充
    # 例如：window_size=5，但只有3帧历史，则填充在index [2,3,4]位置
    start_idx = window_size - len(recent_frames)
    
    # 遍历recent_frames，填充到对应位置
    for i, frame_data in enumerate(recent_frames):
        views = frame_data.get("views", {})
        if view_name in views:
            track_data = views[view_name].get(track_id, None)
            if track_data is not None and "feat" in track_data:
                # feat是query特征
                query_feat = track_data["feat"]
                if isinstance(query_feat, torch.Tensor):
                    # 填充到对应位置（保持时序关系）
                    history_queries[start_idx + i] = query_feat.clone().to(device)
    
    # ⭐ 关键：返回固定长度的列表，缺失的帧保持为0向量
    # ReID模型会自动检测0向量并生成mask=False
    return history_queries


def calculate_reid_loss_with_current_and_memory(current_features, reID_pool, reid_model, triplet_loss_fn, config, device):

    import torch.nn.functional as F
    import random
    from models.utils import get_model
    
    if len(current_features) == 0:
        return None, None
    
    reid_model_unwrapped = get_model(reid_model)
    
    # ========================================
    # Step 1: 按ID分组（用于跨视角正样本构建）
    # ========================================
    features_by_id = {}  # {track_id: [(view, feat), ...]}
    
    for (view, track_id), reid_feat in current_features.items():
        if track_id not in features_by_id:
            features_by_id[track_id] = []
        features_by_id[track_id].append((view, reid_feat))
    
    # ========================================
    # Part 1: ID 分类损失（所有样本都参与）
    # ========================================
    current_id_loss = None
    cls_score_list = []
    gt_labels_list = []
    
    # 获取classifier的类别数
    num_classes = None
    if hasattr(reid_model_unwrapped, 'classifier'):
        if hasattr(reid_model_unwrapped.classifier, 'out_features'):
            num_classes = reid_model_unwrapped.classifier.out_features
        elif hasattr(reid_model_unwrapped.classifier, 'num_classes'):
            num_classes = reid_model_unwrapped.classifier.num_classes
    
    # 遍历所有ID的所有视角
    for track_id, view_feat_list in features_by_id.items():
        # ⚠️ 检查track_id是否超出范围
        if num_classes is not None and track_id >= num_classes:
            continue
        
        # ✅ 每个视角的特征都计算一次分类损失
        for view, reid_feat in view_feat_list:
            reid_feat_input = reid_feat.unsqueeze(0) if reid_feat.dim() == 1 else reid_feat
            
            if hasattr(reid_model_unwrapped, 'classifier'):
                is_metric_learning = hasattr(reid_model_unwrapped, 'ID_LOSS_TYPE') and \
                                   reid_model_unwrapped.ID_LOSS_TYPE in ['arcface', 'cosface', 'amsoftmax', 'circle']
                
                target_label = torch.tensor([track_id], device=device)
                
                if is_metric_learning:
                    cls_score = reid_model_unwrapped.classifier(reid_feat_input, target_label)
                else:
                    cls_score = reid_model_unwrapped.classifier(reid_feat_input)
                
                cls_score_list.append(cls_score)
                gt_labels_list.append(target_label)
    
    if len(cls_score_list) > 0:
        current_id_loss = F.cross_entropy(
            torch.cat(cls_score_list, dim=0),
            torch.cat(gt_labels_list, dim=0)
        )
    
    # ========================================
    # Part 2: Triplet Loss（优先使用跨视角正样本）
    # ========================================
    current_triplet_loss = None
    min_unique_ids_for_triplet = config.get("TRIPLET_MIN_UNIQUE_IDS", 2)
    num_cross_view_pairs = 0
    num_noise_pairs = 0
    
    if len(features_by_id) >= min_unique_ids_for_triplet and triplet_loss_fn is not None:
        triplet_losses = []
        margin = triplet_loss_fn.margin if hasattr(triplet_loss_fn, 'margin') else 0.3
        normalize_feature = triplet_loss_fn.normalize_feature if hasattr(triplet_loss_fn, 'normalize_feature') else True
        
        for anchor_id, anchor_view_feat_list in features_by_id.items():
            # 为每个anchor选择正样本和负样本
            for anchor_idx, (anchor_view, anchor_feat) in enumerate(anchor_view_feat_list):
                
                # ✅ 正样本：优先使用同一ID的其他视角
                positive_candidates = [
                    (v, f) for v, f in anchor_view_feat_list 
                    if v != anchor_view  # 不同视角
                ]
                
                if len(positive_candidates) > 0:
                    # ✅ 使用真实的跨视角正样本（质量高）
                    positive_view, positive_feat = random.choice(positive_candidates)
                    num_cross_view_pairs += 1
                else:
                    # 退化方案：使用噪声增强（质量较低）
                    positive_feat = anchor_feat + torch.randn_like(anchor_feat) * 0.1
                    num_noise_pairs += 1
                
                # 负样本：不同ID的任意视角
                negative_ids = [id for id in features_by_id.keys() if id != anchor_id]
                if len(negative_ids) == 0:
                    continue
                
                negative_id = random.choice(negative_ids)
                negative_view, negative_feat = random.choice(features_by_id[negative_id])
                
                # Normalize
                if normalize_feature:
                    anchor_feat_norm = F.normalize(anchor_feat, p=2, dim=-1)
                    positive_feat_norm = F.normalize(positive_feat, p=2, dim=-1)
                    negative_feat_norm = F.normalize(negative_feat, p=2, dim=-1)
                else:
                    anchor_feat_norm = anchor_feat
                    positive_feat_norm = positive_feat
                    negative_feat_norm = negative_feat
                
                # 计算距离
                pos_dist = F.pairwise_distance(
                    anchor_feat_norm.unsqueeze(0) if anchor_feat_norm.dim() == 1 else anchor_feat_norm,
                    positive_feat_norm.unsqueeze(0) if positive_feat_norm.dim() == 1 else positive_feat_norm,
                    p=2
                )
                neg_dist = F.pairwise_distance(
                    anchor_feat_norm.unsqueeze(0) if anchor_feat_norm.dim() == 1 else anchor_feat_norm,
                    negative_feat_norm.unsqueeze(0) if negative_feat_norm.dim() == 1 else negative_feat_norm,
                    p=2
                )
                
                # Triplet loss
                loss = F.relu(margin + pos_dist - neg_dist)
                triplet_losses.append(loss)
        
        if len(triplet_losses) > 0:
            current_triplet_loss = torch.stack(triplet_losses).mean()
    
    # ========================================
    # Part 3: 跨帧对比损失（与历史特征对比）
    # ========================================
    cross_clip_loss = None
    num_memory_samples = 0
    
    if reID_pool is not None and len(current_features) > 0:
        # 从 ReIDPool 收集历史特征
        mem_feats_list = []
        mem_ids_list = []
        
        for view in reID_pool.view_list:
            for track_id, reid_feat in reID_pool.view_id_reid_feat_dict_list[view].items():
                if reid_feat.dim() > 1:
                    reid_feat = reid_feat.squeeze(0)
                mem_feats_list.append(reid_feat.to(device))
                mem_ids_list.append(track_id)
        
        num_memory_samples = len(mem_feats_list)
        
        if num_memory_samples > 0:
            # 准备当前帧特征矩阵 (N, C)
            curr_ids_list = []
            curr_feats_list = []
            
            for (view, track_id), reid_feat in current_features.items():
                curr_ids_list.append(track_id)
                curr_feats_list.append(reid_feat)
            
            curr_feats_tensor = torch.stack(curr_feats_list)  # (N, C)
            curr_ids_tensor = torch.tensor(curr_ids_list, device=device)
            
            # 准备历史特征矩阵 (M, C)
            mem_feats_tensor = torch.stack(mem_feats_list)  # (M, C)
            mem_ids_tensor = torch.tensor(mem_ids_list, device=device)
            
            # 归一化
            curr_feats_norm = F.normalize(curr_feats_tensor, p=2, dim=1)  # (N, C)
            mem_feats_norm = F.normalize(mem_feats_tensor, p=2, dim=1)    # (M, C)
            
            # 计算相似度矩阵 (N, M)
            sim_matrix = torch.mm(curr_feats_norm, mem_feats_norm.t())  # (N, M)
            
            # 构建正负样本掩码
            pos_mask = curr_ids_tensor.unsqueeze(1) == mem_ids_tensor.unsqueeze(0)  # (N, M)
            neg_mask = ~pos_mask  # (N, M)
            
            # 计算 Hardest Positive 和 Hardest Negative
            margin = config.get("CROSS_CLIP_MARGIN", 0.5)
            
            pos_sim_masked = sim_matrix.clone()
            pos_sim_masked[~pos_mask] = 1e9
            min_pos_sim, _ = pos_sim_masked.min(dim=1)  # (N,)
            
            valid_row_mask = (pos_mask.sum(dim=1) > 0)  # (N,)
            
            neg_sim_masked = sim_matrix.clone()
            neg_sim_masked[~neg_mask] = -1e9
            max_neg_sim, _ = neg_sim_masked.max(dim=1)  # (N,)
            
            # 计算 Triplet Loss
            losses = F.relu(margin - min_pos_sim + max_neg_sim)  # (N,)
            
            if valid_row_mask.sum() > 0:
                cross_clip_loss = losses[valid_row_mask].mean()
    
    # ========================================
    # 组合损失
    # ========================================
    id_loss_weight = config.get("ID_LOSS_WEIGHT", 1.0)
    triplet_loss_weight = config.get("TRIPLET_LOSS_WEIGHT", 0.1)
    cross_clip_loss_weight = config.get("CROSS_CLIP_LOSS_WEIGHT", 0.1)
    
    total_loss = 0
    loss_components = {}
    
    if current_id_loss is not None:
        total_loss = total_loss + id_loss_weight * current_id_loss
        loss_components['id_loss'] = current_id_loss.item()
    
    if current_triplet_loss is not None:
        total_loss = total_loss + triplet_loss_weight * current_triplet_loss
        loss_components['triplet_loss'] = current_triplet_loss.item()
    
    if cross_clip_loss is not None:
        total_loss = total_loss + cross_clip_loss_weight * cross_clip_loss
        loss_components['cross_clip_loss'] = cross_clip_loss.item()
    
    if isinstance(total_loss, int) and total_loss == 0:
        return None, None
    
    # 统计信息
    num_total_samples = sum(len(view_feat_list) for view_feat_list in features_by_id.values())
    num_unique_ids = len(features_by_id)
    
    # 统计多视角ID的数量
    num_multi_view_ids = sum(1 for view_feat_list in features_by_id.values() if len(view_feat_list) > 1)
    
    stats_dict = {
        'num_current_samples': num_total_samples,  # 兼容旧的key名
        'num_total_samples': num_total_samples,
        'num_unique_ids': num_unique_ids,
        'num_multi_view_ids': num_multi_view_ids,
        'num_memory_samples': num_memory_samples,
        **loss_components
    }
    
    # 如果计算了triplet loss，添加正样本来源统计
    if current_triplet_loss is not None:
        stats_dict['num_cross_view_pairs'] = num_cross_view_pairs
        stats_dict['num_noise_pairs'] = num_noise_pairs
    
    return total_loss, stats_dict


def compute_simple_triplet_loss(features_dict, margin, normalize_feature, device):

    import random
    import torch.nn.functional as F
    
    unique_ids = list(features_dict.keys())
    if len(unique_ids) < 2:
        return None
    
    triplet_losses = []
    
    for anchor_id in unique_ids:
        anchor_feat = features_dict[anchor_id]
        
        # 选择 negative（不同 ID）
        negative_ids = [id for id in unique_ids if id != anchor_id]
        if len(negative_ids) == 0:
            continue
        
        negative_id = random.choice(negative_ids)
        negative_feat = features_dict[negative_id]
        
        # Positive: 使用相同 ID 的特征 + 小噪声作为增强
        positive_feat = anchor_feat + torch.randn_like(anchor_feat) * 0.1
        
        # Normalize
        if normalize_feature:
            anchor_feat = F.normalize(anchor_feat, p=2, dim=-1)
            positive_feat = F.normalize(positive_feat, p=2, dim=-1)
            negative_feat = F.normalize(negative_feat, p=2, dim=-1)
        
        # 计算距离
        pos_dist = F.pairwise_distance(
            anchor_feat.unsqueeze(0) if anchor_feat.dim() == 1 else anchor_feat,
            positive_feat.unsqueeze(0) if positive_feat.dim() == 1 else positive_feat,
            p=2
        )
        neg_dist = F.pairwise_distance(
            anchor_feat.unsqueeze(0) if anchor_feat.dim() == 1 else anchor_feat,
            negative_feat.unsqueeze(0) if negative_feat.dim() == 1 else negative_feat,
            p=2
        )
        
        # Triplet loss
        loss = F.relu(margin + pos_dist - neg_dist)
        triplet_losses.append(loss)
    
    if len(triplet_losses) == 0:
        return None
    
    return torch.stack(triplet_losses).mean()


def train(config: dict):
    train_logger = Logger(logdir=os.path.join(config["OUTPUTS_DIR"], "train"), only_main=True)
    train_logger.show(head="Configs:", log=config)
    train_logger.write(log=config, filename="config.yaml", mode="w")
    train_logger.tb_add_git_version(git_version=config["GIT_VERSION"])

    set_seed(config["SEED"])

    model = build_model(config=config)#修改后的MeMOTR模型，其实还是memotr，只不过引入了reid

    # Load Pretrained Model (仅在非恢复训练时加载)
    # 注意：PRETRAINED_MODEL 和 RESUME1 应该互斥使用
    # - PRETRAINED_MODEL: 首次训练时加载预训练权重（如COCO预训练的backbone）
    # - RESUME1: 恢复训练时加载之前保存的checkpoint
    if config["PRETRAINED_MODEL"] is not None and (config.get("RESUME1") is None or config["RESUME1"] in [None, 'None', '', 'null']):
        print(f"[INFO] Loading pretrained model from: {config['PRETRAINED_MODEL']}")
        show_pretrain_details = config.get("SHOW_PRETRAIN_DETAILS", True)
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=show_pretrain_details)
    elif config["PRETRAINED_MODEL"] is not None and config.get("RESUME1") is not None:
        print("[WARNING] Both PRETRAINED_MODEL and RESUME1 are set. RESUME1 will take precedence.")

    # Data process
    dataset_train = build_dataset(config=config, split="train")
    sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
    dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
                                        batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"],
                                        config=config)# 加载数据集花费时间很长

    # 多视角数据集判断
    multiview_datasets = ["UAV_V", "CAMPUS", "WILDTRACK", "MvMHAT", "DIVOTrack"]
    if config["DATASET"] in multiview_datasets:
        #viewpoints = dataset_train.viewpoints
        #我把其他地方的视角数都设置为按场景处理的了，但是reid这里我看是写死的，我先写成最大的5，不知道对4的情况是否有影响

        viewpoints = ["c00"+str(i+1) for i in range(config["VIEW_POINT"])]
        criterion = {}
        
        # ==================== ReID分阶段训练配置 ====================
        # 前N个epoch只训练跟踪，之后再训练ReID
        reid_start_epoch = config.get("REID_START_EPOCH", 0)  # 默认0表示从第0个epoch开始训练ReID
        print("="*80)
        if reid_start_epoch > 0:
            print(f"[分阶段训练] 前 {reid_start_epoch} 个epoch只训练跟踪")
            print(f"[分阶段训练] 从第 {reid_start_epoch} 个epoch开始训练ReID")
        else:
            print(f"[分阶段训练] 从第0个epoch开始同时训练跟踪和ReID")
        print("="*80)
        
        # ==================== ReID模型选择：只有两种 ====================
        # 1. SimpleReIDModel：轻量级MLP模型
        # 2. ReversibleReIDModel：权重共享可逆模型（推荐）
        use_simple_reid = config.get("USE_SIMPLE_REID", False)
        
        if use_simple_reid:
            print(f"[ReID模型] 使用 SimpleReIDModel")
            reid_model = build_simple_reid_model(config=config)
            # 将SimpleReIDModel移动到GPU
            if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
                reid_model = reid_model.to(device=torch.device(config["DEVICE"], distributed_rank()))
            else:
                reid_model = reid_model.to(device=torch.device(config["DEVICE"]))
        else:
            print(f"[ReID模型] 使用 ReversibleReIDModel（权重共享可逆版本）")
            reid_model = build_reversible_reid(config=config)  # 内部已处理设备移动
        
        # ==================== 初始化时冻结ReID模型（如果需要） ====================
        if reid_start_epoch > 0:
            print(f"[参数冻结] 初始化时冻结ReID模型参数（前{reid_start_epoch}个epoch不更新）")
            for param in reid_model.parameters():
                param.requires_grad = False
            reid_model.eval()  # 设置为eval模式

        # 构建ReIDPool，保留整个batch的reid特征
        keep_all_batch_reid = config.get("KEEP_ALL_BATCH_REID", True)# 这个参数目前没用上
        reID_pool = build_reid_pool(views=viewpoints, max_forget_length = config["MAX_FORGET_LENGTH"], 
                                    training = True, keep_all_batch_reid=keep_all_batch_reid,reid_update_weight=config.get("REID_UPDATE_WEIGHT", 0.8))
        
        # 初始化MemoryBank
        memory_bank = MemoryBank(
            bank_len=config.get("MEMORY_BANK_LEN", 30),
            hidden_dim=config.get("HIDDEN_DIM", 256),
            use_dab=config.get("USE_DAB", True),
            temporal_k=config.get("TEMPORAL_K", 8),
            decay_alpha=config.get("DECAY_ALPHA", 0.25),
            num_heads=config.get("NUM_HEADS", 8),
            device="cuda" if torch.cuda.is_available() else "cpu",
            training=True,
            reid_update_weight=config.get("REID_UPDATE_WEIGHT", 0.1)
        )
        memory_bank = memory_bank.to(device=torch.device("cuda", distributed_rank()) if torch.cuda.is_available() else torch.device("cpu"))
        
        # 初始化ReID损失函数
        # 1. 三元组损失（用于ReID特征学习）
        triplet_loss_fn = build_triplet_loss(config=config) if config.get("USE_TRIPLET_LOSS", False) else None #构建三元组损失
        
        # 2. 不确定性加权损失（用于平衡Tracking和ReID两个任务的损失）
        #    注意：这是元学习层的损失，用于动态调整任务权重
        use_uncertainty_loss = config.get("USE_UNCERTAINTY_LOSS", True)
        uncertainty_loss_fn = None
        if use_uncertainty_loss:
            uncertainty_loss_fn = build_uncertainty_loss(config=config)#只是构建了个空壳，本质在于前向传播的调用
            uncertainty_loss_fn = uncertainty_loss_fn.to(device=torch.device("cuda", distributed_rank()) if torch.cuda.is_available() else torch.device("cpu"))
        
        # 3. 可逆性损失的可学习权重（用于平衡两个可逆性损失）
        #    仅在 USE_REVERSIBILITY_LOSS=True 且使用ReversibleReID模型时需要
        use_reversibility_loss = config.get("USE_REVERSIBILITY_LOSS", True)
        use_learnable_rev_weight = (
            use_reversibility_loss and 
            config.get("USE_LEARNABLE_REV_WEIGHT", False) and 
            not config.get("USE_SIMPLE_REID", False)
        )
        reversibility_weight_learner = None
        
        if use_reversibility_loss:
            print(f"\n{'='*80}")
            print(f"[Train] ✅ Reversibility Loss ENABLED")
            print(f"  - Level 1 Weight: {config.get('REVERSIBILITY_WEIGHT1', 0.1)}")
            print(f"  - Level 2 Weight: {config.get('REVERSIBILITY_WEIGHT2', 0.1)}")
            print(f"  - Learnable Weight: {use_learnable_rev_weight}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"[Train] ⚠️  Reversibility Loss DISABLED")
            print(f"  - Reconstruction loss will not be calculated")
            print(f"{'='*80}\n")
        
        if use_learnable_rev_weight:
            reversibility_weight_learner = build_reversibility_weight_learner(config=config)
            reversibility_weight_learner = reversibility_weight_learner.to(device=torch.device("cuda", distributed_rank()) if torch.cuda.is_available() else torch.device("cpu"))
        
        for view in viewpoints:#每个视角一个损失
            # Criterion
            criterion_one = build_criterion(config=config)
            criterion_one.set_device(torch.device("cuda", distributed_rank()))
            criterion[view] = criterion_one
    else:
        criterion = build_criterion(config=config)
    
    # Optimizer
    param_groups, lr_names = get_param_groups(config=config, model=model)#为不同部分设置不同的学习率


    # 多视角数据集判断
    multiview_datasets = ["UAV_V", "CAMPUS", "WILDTRACK", "MvMHAT", "DIVOTrack"]
    if config["DATASET"] in multiview_datasets: 
        param_groups_reid, lr_names_reid = get_param_groups_reid(config=config, model=reid_model)
        param_groups += param_groups_reid
        lr_names += lr_names_reid
        
        # 添加不确定性损失参数到优化器，主要是更新权重参数
        if uncertainty_loss_fn is not None:
            param_groups.append({
                'params': uncertainty_loss_fn.parameters(),
                'lr': config.get("LR", 2.0e-4),  # 使用主学习率
                'name': 'uncertainty_loss'
            })
            lr_names.append('uncertainty_loss')
        
        # 添加可逆性权重学习器参数到优化器
        if reversibility_weight_learner is not None:
            param_groups.append({
                'params': reversibility_weight_learner.parameters(),
                'lr': config.get("LR_REVERSIBILITY_WEIGHT", config.get("LR", 2.0e-4)),  # 使用主学习率或单独学习率
                'name': 'reversibility_weight'
            })
            lr_names.append('reversibility_weight')

    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    
    # ==================== 验证优化器参数组（调试用） ====================
    if config["DATASET"] in multiview_datasets:
        print("="*80)
        print("[优化器初始化验证]")
        print(f"总参数组数量: {len(optimizer.param_groups)}")
        for i, (group, name) in enumerate(zip(optimizer.param_groups, lr_names)):
            num_params = len(group['params'])
            lr = group['lr']
            print(f"  Group {i} ({name:25s}): {num_params:4d} params, lr={lr:.2e}")
            
            # 额外检查 ReID 参数组是否为空
            if 'reid' in name.lower() or 'base' in name.lower() or 'bottleneck' in name.lower() or 'classifier' in name.lower():
                if num_params == 0:
                    print(f"    ⚠️  WARNING: ReID 参数组 '{name}' 为空！")
                else:
                    # 检查这些参数的 requires_grad 状态
                    num_trainable = sum(1 for p in group['params'] if p.requires_grad)
                    num_frozen = num_params - num_trainable
                    print(f"    ✅ 可训练: {num_trainable}, 冻结: {num_frozen}")
        print("="*80)
    
    # Scheduler
    if config["LR_SCHEDULER"] == "MultiStep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config["LR_DROP_MILESTONES"],
            gamma=config["LR_DROP_RATE"]
        )
    elif config["LR_SCHEDULER"] == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config["EPOCHS"]
        )
    else:
        raise ValueError(f"Do not support lr scheduler '{config['LR_SCHEDULER']}'")

    # Training states
    train_states = {
        "start_epoch": 0,
        "global_iters": 0
    }

    # Resume
    if config["RESUME1"] is not None and config["RESUME1"] not in [None, 'None', '', 'null']:
        print(f"[INFO] Loading tracking model from checkpoint: {config['RESUME1']}")
        if config["RESUME_SCHEDULER"]:
            load_checkpoint(model=model, path=config["RESUME1"], states=train_states,
                            optimizer=optimizer, scheduler=scheduler)
        else:
            load_checkpoint(model=model, path=config["RESUME1"], states=train_states)
            for _ in range(train_states["start_epoch"]):
                scheduler.step()
    
    # 加载ReID模型（如果指定了RESUME2）
    if config["DATASET"] in multiview_datasets and config.get("RESUME2") is not None and config["RESUME2"] not in [None, 'None', '', 'null']:
        print(f"[INFO] Loading ReID model from checkpoint: {config['RESUME2']}")
        load_checkpoint(model=reid_model, path=config["RESUME2"], states=train_states)

    # Set start epoch
    start_epoch = train_states["start_epoch"]

    # ==================== DDP包装 ====================
    reid_start_epoch = config.get("REID_START_EPOCH", 0)
    
    if is_distributed():
        # 主模型：始终有可训练参数，直接包装DDP
        model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=True)
        
        # ReID模型：根据分阶段训练策略决定是否包装DDP
        if config["DATASET"] in multiview_datasets:
            has_trainable_params = any(p.requires_grad for p in reid_model.parameters())
            
            if has_trainable_params:
                # 有可训练参数：直接包装DDP（REID_START_EPOCH=0的情况）
                reid_model = DDP(module=reid_model, device_ids=[distributed_rank()], find_unused_parameters=True)
                print(f"[DDP初始化] ReID模型已包装DDP（rank={distributed_rank()}）")
            else:
                # 参数被冻结：仅移动到GPU，稍后在epoch循环中动态包装DDP
                reid_model = reid_model.to(torch.device(f"cuda:{distributed_rank()}"))
                print(f"[DDP初始化] ReID模型参数被冻结，仅移动到GPU（rank={distributed_rank()}）")
                print(f"[DDP初始化] 将在epoch={reid_start_epoch}时动态包装DDP")

    multi_checkpoint = "MULTI_CHECKPOINT" in config and config["MULTI_CHECKPOINT"]

    # Training:
    
    for epoch in range(start_epoch, config["EPOCHS"]):
        # 为分布式训练和数据增强设置epoch（不需要重新构建sampler/dataloader）
        if is_distributed():
            sampler_train.set_epoch(epoch)
        dataset_train.set_epoch(epoch)

        # ⚠️ 仅训练query_updater模式：冻结backbone、points和其他非query_updater参数
        if epoch >= config["ONLY_TRAIN_QUERY_UPDATER_AFTER"]:
            # 🔧 修复：使用lr_names动态匹配，避免硬编码索引（更健壮）
            for idx, name in enumerate(lr_names):
                if name in ["lr_backbone", "lr_points", "lr"] and name != "lr_query_updater":
                    optimizer.param_groups[idx]["lr"] = 0.0
            # 注意：ReID相关的参数组不受影响，继续训练
        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        assert len(lrs) == len(lr_names)
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        train_logger.show(head=f"[Epoch {epoch}] lr={lr_info}")
        train_logger.write(head=f"[Epoch {epoch}] lr={lr_info}")
        default_lr_idx = -1
        for _ in range(len(lr_names)):
            if lr_names[_] == "lr":
                default_lr_idx = _
        train_logger.tb_add_scalar(tag="lr", scalar_value=lrs[default_lr_idx], global_step=epoch, mode="epochs")

        no_grad_frames = None
        if "NO_GRAD_FRAMES" in config:
            for i in range(len(config["NO_GRAD_STEPS"])):
                if epoch >= config["NO_GRAD_STEPS"][i]:
                    no_grad_frames = config["NO_GRAD_FRAMES"][i]
                    break
        
        # ==================== ReID分阶段训练：动态启用/禁用 ====================
        reid_start_epoch = config.get("REID_START_EPOCH", 0)
        enable_reid_training = (epoch >= reid_start_epoch)
        
        # ⭐ ReID学习率预热（从reid_start_epoch开始，线性增长）
        reid_warmup_epochs = config.get("REID_WARMUP_EPOCHS", 0)
        if enable_reid_training and reid_warmup_epochs > 0:
            epochs_since_reid_start = epoch - reid_start_epoch
            if epochs_since_reid_start < reid_warmup_epochs:
                # 预热阶段：学习率从0线性增长到目标值
                warmup_factor = (epochs_since_reid_start + 1) / reid_warmup_epochs
                
                # 调整ReID相关的学习率
                for idx, name in enumerate(lr_names):
                    if name in ["lr_reid", "lr_bottleneck", "lr_classifier"]:
                        # 获取目标学习率（配置文件中的值）
                        if name == "lr_reid":
                            target_lr = config.get("LR_REID", config["LR"])
                        elif name == "lr_bottleneck":
                            target_lr = config.get("LR_BOTTLENECK", config.get("LR_REID", config["LR"]))
                        elif name == "lr_classifier":
                            target_lr = config.get("LR_CLASSIFIER", config.get("LR_REID", config["LR"]))
                        
                        # 应用预热因子
                        optimizer.param_groups[idx]["lr"] = target_lr * warmup_factor
                
                train_logger.show(f"[Epoch {epoch}] 🔥 ReID学习率预热中：{warmup_factor:.2%} "
                                f"(第{epochs_since_reid_start+1}/{reid_warmup_epochs}个预热epoch)")
        
        # 在达到指定epoch时，解冻ReID模型参数并动态包装DDP
        if reid_start_epoch > 0 and epoch == reid_start_epoch:
            train_logger.show(f"="*80)
            train_logger.show(f"[Epoch {epoch}] 🎯 开始训练ReID模型！")
            train_logger.write(f"[Epoch {epoch}] 开始训练ReID模型！")
            
            # Step 1: 解冻ReID模型参数
            train_logger.show(f"[Epoch {epoch}]   Step 1/3: 解冻ReID参数...")
            for param in reid_model.parameters():
                param.requires_grad = True
            
            # Step 2: 关键！如果是分布式环境，动态包装为DDP
            if is_distributed():
                train_logger.show(f"[Epoch {epoch}]   Step 2/3: 包装DDP（rank={distributed_rank()}）...")
                # 注意：这里重新赋值reid_model变量
                reid_model = DDP(module=reid_model, device_ids=[distributed_rank()], find_unused_parameters=True)
                train_logger.show(f"[Epoch {epoch}]   ✅ ReID模型已成功包装为DDP")
            else:
                train_logger.show(f"[Epoch {epoch}]   Step 2/3: 非分布式模式，跳过DDP包装")
            
            # Step 3: 设置为训练模式
            train_logger.show(f"[Epoch {epoch}]   Step 3/3: 设置为训练模式...")
            reid_model.train()
            
            # Step 4: 验证优化器是否包含ReID参数
            train_logger.show(f"[Epoch {epoch}]   🔄 检查优化器参数...")
            
            # 🔧 修复：DDP包装后需要使用get_model获取底层模型
            reid_model_for_check = get_model(reid_model)
            reid_params_set = set(reid_model_for_check.parameters())
            
            reid_params_in_optimizer = sum(1 for group in optimizer.param_groups 
                                           for p in group['params'] 
                                           if p in reid_params_set)
            train_logger.show(f"[Epoch {epoch}]   优化器中ReID参数数量: {reid_params_in_optimizer}")
            
            # 检查requires_grad状态
            num_trainable = sum(1 for p in reid_params_set if p.requires_grad)
            train_logger.show(f"[Epoch {epoch}]   ReID可训练参数数量: {num_trainable}")
            
            train_logger.show(f"[Epoch {epoch}] ✅ ReID模型已完全启动！")
            train_logger.show(f"="*80)
        
        # 显示当前epoch的训练状态
        if epoch < reid_start_epoch:
            train_logger.show(f"[Epoch {epoch}] 📍 当前模式：只训练跟踪（ReID模型冻结）")
        else:
            train_logger.show(f"[Epoch {epoch}] 📍 当前模式：同时训练跟踪和ReID")

        train_one_epoch(
            model=model,
            train_states=train_states,
            max_norm=config["CLIP_MAX_NORM"],
            dataloader=dataloader_train,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            logger=train_logger,
            accumulation_steps=config["ACCUMULATION_STEPS"],
            use_dab=config["USE_DAB"],
            multi_checkpoint=multi_checkpoint,
            no_grad_frames=no_grad_frames,
            reid_model=reid_model,
            reID_pool=reID_pool,
            enable_reid_training=enable_reid_training,  # 新增：控制是否训练ReID
            memory_bank=memory_bank,
            triplet_loss_fn=triplet_loss_fn,
            uncertainty_loss_fn=uncertainty_loss_fn,
            reversibility_weight_learner=reversibility_weight_learner,  # 新增：可逆性权重学习器
            config=config,
            lr_names=lr_names  # ⭐ 传递学习率名称列表
        )
        scheduler.step()
        train_states["start_epoch"] += 1
        if multi_checkpoint is True:
            pass
        else:
            if config["DATASET"] == "DanceTrack" or config["EPOCHS"] < 100 or (epoch + 1) % 2 == 0:
                # 保存主模型（tracking model）
                save_checkpoint(
                    model=model,
                    path=os.path.join(config["OUTPUTS_DIR"], f"model_checkpoint_{epoch}.pth"),
                    states=train_states,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                
                # 保存ReID模型（仅在多视角数据集时）
                if config["DATASET"] in multiview_datasets and enable_reid_training:
                    # 注意：推理时只需要模型权重，不需要optimizer和scheduler
                    save_checkpoint(
                        model=reid_model,
                        path=os.path.join(config["OUTPUTS_DIR"], f"reid_checkpoint_{epoch}.pth"),
                        states=train_states
                        # 不保存optimizer和scheduler，因为推理时不需要
                    )
                # # 加载主模型
                # load_checkpoint(model=model, path="model_checkpoint_X.pth")

                # # 加载ReID模型（如果需要）
                # load_checkpoint(model=reid_model, path="reid_checkpoint_X.pth")
    return


def train_one_epoch(model: FusionTrack, train_states: dict, max_norm: float,
                    dataloader: DataLoader, criterion: ClipCriterion | dict, optimizer: torch.optim,
                    epoch: int, logger: Logger,
                    accumulation_steps: int = 1, use_dab: bool = False,
                    multi_checkpoint: bool = False,
                    no_grad_frames: int | None = None, reid_model: nn.Module = None , reID_pool: ReIDPool = None,
                    enable_reid_training: bool = True,  # 新增：控制是否启用ReID训练 
                    memory_bank: MemoryBank = None, triplet_loss_fn = None, uncertainty_loss_fn = None, 
                    reversibility_weight_learner = None,  # 新增：可逆性权重学习器
                    config = None, lr_names: list = None):
    """
    Args:
        model: Model.
        train_states:
        max_norm: clip max norm.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Training optimizer.
        epoch: Current epoch.
        # metric_log: Metric Log.
        logger: unified logger.
        accumulation_steps:
        use_dab:
        multi_checkpoint:
        no_grad_frames:

    Returns:
        Logs
    """

    model.train()
    
    # ⭐ 关键：根据enable_reid_training设置ReID模型的train/eval模式
    if reid_model is not None:
        if enable_reid_training:
            reid_model.train()
            # 确认参数可训练
            trainable_params = sum(p.requires_grad for p in reid_model.parameters())
            if trainable_params == 0:
                logger.show(f"[WARNING] ReID模型没有可训练参数！所有参数的requires_grad=False")
        else:
            # 🔧 修复：ReID训练未启用时，应该设置为eval模式
            reid_model.eval()
            logger.show(f"[INFO] ReID模型设置为eval模式（当前epoch不训练ReID）")
    
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device

    dataloader_len = len(dataloader)
    metric_log = MetricLog()
    epoch_start_timestamp = time.time()
    
    cur_frame_view = [0 for _ in criterion.keys()]

    for i, batch in enumerate(dataloader):
        # ✅ 记录batch开始时间（用于计算iter耗时）
        
        iter_start_timestamp = time.time()
        
        # 每个batch开始时清空MemoryBank和ReIDPool
        if memory_bank is not None:
            memory_bank.clear()
        if reID_pool is not None:
            reID_pool.clear_all()  # 清空所有query、reid特征、life、frame记录
        
        if config["DATASET"] == "UAV_V":
            loss_view_dict = {}
            cls_score_list = []
            global_feat_list = []
            gt_labels_list = []
            
            # 检查batch结构：是否是多clip结构
            is_multiclip = 'clips' in batch
            if is_multiclip:
                clips = batch['clips']
            else:
                # 向后兼容：单clip结构，转换为多clip格式
                clips = [batch]
            
            # [优化] 准备手动加权的权重（仅在不使用不确定性损失时使用）
            w_track_val = 1.0
            w_reid_val = config.get("ID_LOSS_WEIGHT", config.get("REID_LOSS_WEIGHT", 0.5))
            
            # 存储 detach 后的 loss 用于日志
            tracking_loss_vals = []
            reid_loss_vals = []

            # 遍历每个clip
            for clip_idx, clip_batch in enumerate(clips):
                #每个clip开始时清空MemoryBank（可选，如果希望clip间独立）
                if memory_bank is not None and clip_idx > 0:
                    memory_bank.clear()
                
                # ✅ 每个clip重置帧计数（clips之间不连续）
                cur_frame_view = [0 for _ in criterion.keys()]
                
                # ✅ 累积整个clip的损失（所有帧、所有视角）
                clip_total_loss = None
                
                # 为所有视角初始化tracks和criterion
                all_view_tracks = {}
                all_view_batches = {}
                for view_idx, view in enumerate(criterion.keys()):#遍历视角，为每个视角初始化track
                    view_criterion = criterion[view]
                    view_batch = clip_batch[view]
                    all_view_batches[view] = view_batch
                    
                    # 初始化tracks
                    tracks = TrackInstances.init_tracks(batch=view_batch,
                                                        hidden_dim=get_model(model).hidden_dim,
                                                        num_classes=get_model(model).num_classes,
                                                        device=device, use_dab=use_dab)
                    all_view_tracks[view] = tracks
                    
                    # 初始化criterion
                    view_criterion.init_a_clip(batch=view_batch,
                                        hidden_dim=get_model(model).hidden_dim,
                                        num_classes=get_model(model).num_classes,
                                        device=device)

                # 获取帧数（所有视角的帧数相同，获取第一个视角的就可以）
                first_view = list(criterion.keys())[0]
                num_frames = len(all_view_batches[first_view]["imgs"][0])
                
                # 按帧处理
                for frame_idx in range(num_frames):
                    # 用于累积当前帧所有视角的损失
                    frame_losses = []
                    frame_log_dict = {}  # 用于记录当前帧的详细损失
                    
                    # 收集当前帧的 ReID 特征（保持梯度）
                    # 使用 (view, track_id) 作为key，避免不同视角的相同ID被覆盖
                    current_frame_reid_features = {}  # {(view, track_id): reid_feat} 
                    
                    # 收集当前帧所有视角的tracks和reid特征（用于批量更新MemoryBank）
                    frame_all_view_tracks = {}  # {view: tracks}
                    frame_all_reid_features = {}  # {(view, id): reid_feat}
                    frame_reversibility_losses = []  # 收集可逆性损失
                    
                    # 1. 按视角顺序进行模型forward和处理
                    for view_idx, view in enumerate(criterion.keys()):
                        view_criterion = criterion[view]
                        view_batch = all_view_batches[view]
                        tracks = all_view_tracks[view]
                        
                        if no_grad_frames is None or frame_idx >= no_grad_frames:
                            frame = [fs[frame_idx] for fs in view_batch["imgs"]]
                            for f in frame:
                                f.requires_grad_(False)
                            frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
                            
                            # 准备视角ID（从infos中获取，batch_size=1时）
                            view_id_tensor = None
                            if len(view_batch["infos"]) > 0 and frame_idx < len(view_batch["infos"][0]):
                                # 从第一个样本的第一帧获取view_id
                                view_id = view_batch["infos"][0][frame_idx].get('view_id', view_idx)
                                view_id_tensor = torch.tensor([view_id], dtype=torch.long, device=device)
                            
                            res = model(frame=frame, tracks=tracks, view_ids=view_id_tensor)

                            if config["REID_LOSS"] and enable_reid_training:
                                previous_tracks, new_tracks, unmatched_dets, true_id = view_criterion.process_single_frame(
                                    model_outputs=res,
                                    tracked_instances=tracks,
                                    frame_idx=frame_idx,
                                    get_true_id=True
                                )
                            else:
                                previous_tracks, new_tracks, unmatched_dets = view_criterion.process_single_frame(
                                    model_outputs=res,
                                    tracked_instances=tracks,
                                    frame_idx=frame_idx
                                )
                            
                            # 利用 process_single_frame 已经填充好的 output_embed（带梯度）
                            id_to_feature_map = {}
                            if config["REID_LOSS"] and enable_reid_training:
                                # 1. 收集老轨迹的特征 (previous_tracks 在 update_tracked_instances 中已被更新)
                                for b_tracks in previous_tracks:
                                    for track_idx, track_id in enumerate(b_tracks.ids):
                                        if track_id.item() >= 0:
                                            # 直接拿 output_embed，它是 transformer 输出的对应切片，带梯度
                                            id_to_feature_map[track_id.item()] = b_tracks.output_embed[track_idx]
                                
                                # 2. 收集新轨迹的特征 (new_tracks 在 process_single_frame 中初始化时已赋值 output_embed)
                                for b_tracks in new_tracks:
                                    for track_idx, track_id in enumerate(b_tracks.ids):
                                        if track_id.item() >= 0:
                                            id_to_feature_map[track_id.item()] = b_tracks.output_embed[track_idx]
                            
                            if frame_idx < len(view_batch["imgs"][0]) - 1:
                                tracks_updated = get_model(model).postprocess_single_frame(
                                    previous_tracks, new_tracks, unmatched_dets)
                                # 更新all_view_tracks，供下一帧的跨视角更新使用
                                all_view_tracks[view] = tracks_updated
                            else:
                                # 最后一帧也需要合并以提取特征
                                tracks_updated = get_model(model).postprocess_single_frame(
                                    previous_tracks, new_tracks, unmatched_dets)
                            
                            # 提取 ReID 特征（现在只需要查表，安全且快）
                            reid_features_dict = {}
                            if config["REID_LOSS"] and enable_reid_training:
                                for t_idx, track in enumerate(tracks_updated):
                                    for local_idx, track_id in enumerate(track.ids):
                                        tid = track_id.item()
                                        if tid >= 0:
                                            if tid in id_to_feature_map:
                                                # 查表直接获取对应的 Transformer 输出特征
                                                query_feat = id_to_feature_map[tid]
                                                
                                                # 准备ReID输入：支持单帧或多帧query
                                                use_multiframe_reid = config.get("USE_MULTIFRAME_REID", False)
                                                
                                                if use_multiframe_reid and memory_bank is not None:
                                                    # 使用多帧query（列表格式）
                                                    window_size = config.get("REID_QUERY_WINDOW_SIZE", 5)
                                                    history_queries = extract_multiframe_queries_from_memorybank(
                                                        memory_bank, tid, view, window_size
                                                    )
                                                    
                                                    # 拼接历史query列表 + 当前query
                                                    # history_queries是List[Tensor(dim,)]，可能为空列表、1帧、2帧...最多window_size帧
                                                    query_list = history_queries + [query_feat]  # 添加当前帧到列表末尾
                                                    query_input = query_list  # 直接作为列表传入ReID模型
                                                else:
                                                    # 使用单帧query（旧方式）
                                                    query_input = [query_feat.unsqueeze(0)]
                                                
                                                # 经过reid模型得到重识别特征
                                                with torch.set_grad_enabled(True):
                                                    use_simple_reid = config.get("USE_SIMPLE_REID", False)
                                                    
                                                    if use_simple_reid:
                                                        # SimpleReID模型：训练模式返回 (cls_score, reid_feat)
                                                        _, reid_feat = get_model(reid_model)(query_input, None)
                                                    else:
                                                        # ReversibleReID模型：训练模式返回 (reid_feat, rev_loss1, rev_loss2
                                                        reid_output = reid_model(query_input, None)
                                                        if isinstance(reid_output, tuple) and len(reid_output) == 3:
                                                            # 训练模式：解包3个值
                                                            reid_feat, rev_loss1, rev_loss2 = reid_output
                                                            # 收集可逆性损失
                                                            frame_reversibility_losses.append((rev_loss1, rev_loss2))
                                                        else:
                                                            # 推理模式：直接返回特征
                                                            reid_feat = reid_output
                                                    
                                                    # 处理返回值（统一格式）
                                                    if isinstance(reid_feat, tuple):
                                                        reid_feat = reid_feat[1]
                                                    if reid_feat.dim() > 1:
                                                        reid_feat = reid_feat[0]
                                                    
                                                    # L2归一化：与TripletLoss保持一致
                                                    reid_feat = torch.nn.functional.normalize(reid_feat, p=2, dim=-1)
                                                    
                                                    reid_features_dict[tid] = reid_feat  # 视角内部的dict
                                                    current_frame_reid_features[(view, tid)] = reid_feat  # ✅ 使用组合key避免覆盖
                            
                            # 更新tracks为postprocess后的结果
                            if frame_idx < len(view_batch["imgs"][0]) - 1:
                                tracks = tracks_updated
                        else:
                            with torch.no_grad():
                                frame = [fs[frame_idx] for fs in view_batch["imgs"]]
                                for f in frame:
                                    f.requires_grad_(False)
                                frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
                                
                                # 准备视角ID
                                view_id_tensor = None
                                if len(view_batch["infos"]) > 0 and frame_idx < len(view_batch["infos"][0]):
                                    view_id = view_batch["infos"][0][frame_idx].get('view_id', view_idx)
                                    view_id_tensor = torch.tensor([view_id], dtype=torch.long, device=device)
                                
                                res = model(frame=frame, tracks=tracks, view_ids=view_id_tensor)
                                if config["REID_LOSS"] and enable_reid_training:
                                    previous_tracks, new_tracks, unmatched_dets, true_id = view_criterion.process_single_frame(
                                        model_outputs=res,
                                        tracked_instances=tracks,
                                        frame_idx=frame_idx,
                                        get_true_id=True
                                    )
                                else:
                                    previous_tracks, new_tracks, unmatched_dets = view_criterion.process_single_frame(
                                        model_outputs=res,
                                        tracked_instances=tracks,
                                        frame_idx=frame_idx
                                    )
                                
                                # 建立 {Track_ID: Feature} 映射表（no_grad模式，使用 output_embed）
                                id_to_feature_map = {}
                                if config["REID_LOSS"] and enable_reid_training:
                                    # 1. 收集老轨迹的特征
                                    for b_tracks in previous_tracks:
                                        for track_idx, track_id in enumerate(b_tracks.ids):
                                            if track_id.item() >= 0:
                                                id_to_feature_map[track_id.item()] = b_tracks.output_embed[track_idx]
                                    
                                    # 2. 收集新轨迹的特征
                                    for b_tracks in new_tracks:
                                        for track_idx, track_id in enumerate(b_tracks.ids):
                                            if track_id.item() >= 0:
                                                id_to_feature_map[track_id.item()] = b_tracks.output_embed[track_idx]
                                
                                # 执行postprocess
                                if frame_idx < len(view_batch["imgs"][0]) - 1:
                                    tracks_updated = get_model(model).postprocess_single_frame(
                                        previous_tracks, new_tracks, unmatched_dets, no_augment=frame_idx < no_grad_frames-1)
                                    all_view_tracks[view] = tracks_updated
                                else:
                                    # 最后一帧也需要合并以提取特征
                                    tracks_updated = get_model(model).postprocess_single_frame(
                                        previous_tracks, new_tracks, unmatched_dets, no_augment=False)
                                
                                # 提取 ReID 特征（只提取有效目标的reid特征，-1的就是皮配不上的
                                reid_features_dict = {}
                                if config["REID_LOSS"] and enable_reid_training:
                                    for t_idx, track in enumerate(tracks_updated):
                                        for local_idx, track_id in enumerate(track.ids):
                                            tid = track_id.item()
                                            if tid >= 0:
                                                if tid in id_to_feature_map:
                                                    query_feat = id_to_feature_map[tid]
                                                    
                                                    # 准备ReID输入：支持单帧或多帧query
                                                    use_multiframe_reid = config.get("USE_MULTIFRAME_REID", False)
                                                    
                                                    if use_multiframe_reid and memory_bank is not None:
                                                        # 使用多帧query（列表格式）
                                                        window_size = config.get("REID_QUERY_WINDOW_SIZE", 5)
                                                        history_queries = extract_multiframe_queries_from_memorybank(
                                                            memory_bank, tid, view, window_size
                                                        )
                                                        
                                                        # 拼接历史query列表 + 当前query
                                                        query_list = history_queries + [query_feat]
                                                        query_input = query_list  # 直接作为列表传入ReID模型
                                                    else:
                                                        # 使用单帧query（旧方式）
                                                        query_input = [query_feat.unsqueeze(0)]
                                                    
                                                    use_simple_reid = config.get("USE_SIMPLE_REID", False)
                                                    
                                                    # 两种模型在no_grad模式下都只返回特征
                                                    reid_feat = get_model(reid_model)(query_input, None)
                                                    
                                                    # 处理返回值
                                                    if isinstance(reid_feat, tuple):
                                                        reid_feat = reid_feat[1]
                                                    if reid_feat.dim() > 1:
                                                        reid_feat = reid_feat[0]
                                                    reid_features_dict[tid] = reid_feat
                                
                                # 更新tracks为postprocess后的结果
                                if frame_idx < len(view_batch["imgs"][0]) - 1:
                                    tracks = tracks_updated
                        
                        if config["REID_LOSS"] and enable_reid_training:
                            # ReIDPool存储的是历史特征，不需要梯度，必须detach防止显存爆炸
                            reid_features_detached = {k: v.detach() for k, v in reid_features_dict.items()} if len(reid_features_dict) > 0 else None
                            
                            # 更新pool，传入detached特征
                            reID_pool.update_pool(view, tracks, cur_frame_view[view_idx] + frame_idx, 
                                                reid_features=reid_features_detached)
                            
                            # 收集当前视角的tracks和reid特征（稍后批量更新MemoryBank）
                            # MemoryBank同样需要detached特征，避免计算图累积
                            if memory_bank is not None:
                                frame_all_view_tracks[view] = tracks
                                if len(reid_features_dict) > 0:
                                    for track_id, reid_feat in reid_features_dict.items():
                                        # ✅ 存入MemoryBank的特征也必须detach
                                        frame_all_reid_features[(view, track_id)] = reid_feat.detach()
                        
                    # 计算当前视角的损失
                    loss_dict, log_dict = view_criterion.get_mean_by_n_gts()
                    view_loss = view_criterion.get_sum_loss_dict(loss_dict=loss_dict)
                    
                    # 1. 记录 Loss 数值用于日志
                    if view not in loss_view_dict:
                        loss_view_dict[view] = view_loss.item()
                    else:
                        loss_view_dict[view] += view_loss.item()
                    
                    # 2. 记录 Detach Loss 用于权重更新
                    tracking_loss_vals.append(view_loss.detach())
                    
                    # 3. 累积当前帧的损失（原始损失，不加权）
                    frame_losses.append(view_loss)
                    
                    # 4. 累积当前帧的详细损失信息（用于日志）
                    # 确保只存储数值，不存储tensor，避免计算图残留
                    for log_k, log_v in log_dict.items():
                        if log_k not in frame_log_dict:
                            frame_log_dict[log_k] = []
                        # 如果是tensor，提取数值；否则直接存储
                        val = log_v[0].item() if isinstance(log_v[0], torch.Tensor) else log_v[0]
                        frame_log_dict[log_k].append(val)
                    
                    # 5. 释放不再需要的变量
                    del res
                    
                    # 当前帧的所有视角处理完后，一次性更新MemoryBank
                    # MemoryBank只做跨帧更新（temporal update）
                    if memory_bank is not None and config["REID_LOSS"] and enable_reid_training and len(frame_all_view_tracks) > 0:
                        # 所有视角使用相同的时间戳（取第一个视角的时间戳）
                        t = cur_frame_view[0] + frame_idx
                        memory_bank.push_from_views(
                            frame_all_view_tracks,
                            t
                        )
                        
                        # 训练时：使用GT ID选择ReID特征来更新query
                        # 这一步在跨帧更新之后，使用ReID特征来增强query
                        if len(frame_all_reid_features) > 0:
                            from models.query_update_from_reid import update_query_with_reid_features
                            
                            for view in frame_all_view_tracks.keys():
                                tracks = frame_all_view_tracks[view]
                                # 提取该视角的ReID特征
                                view_reid_features = {}
                                for (v, track_id), reid_feat in frame_all_reid_features.items():
                                    if v == view:
                                        view_reid_features[track_id] = reid_feat
                                
                                if len(view_reid_features) > 0:
                                    # 根据GT ID更新query（原地修改）
                                    update_query_with_reid_features(
                                        tracks=tracks,
                                        reid_features=view_reid_features,
                                        reid_update_weight=config.get("REID_UPDATE_WEIGHT", 0.1),
                                        use_dab=config["USE_DAB"]
                                    )
                    
                    #  当前帧的所有视角处理完后，计算总损失（Tracking + ReID）
                    if len(frame_losses) > 0:
                        # frame_tracking_loss 是原始的跟踪损失（未加权）
                        frame_tracking_loss = sum(frame_losses)
                        
                        #  计算 ReID loss：当前帧内部 + 跨帧对比
                        frame_reid_loss = None
                        reid_stats = None
                        # 🔥 关键：只有在启用ReID训练时才计算ReID损失
                        if config["REID_LOSS"] and enable_reid_training:
                            # 使用当前帧特征 + ReIDPool 历史特征计算损失
                            frame_reid_loss, reid_stats = calculate_reid_loss_with_current_and_memory(
                                current_features=current_frame_reid_features,  # 当前帧特征（有梯度）
                                reID_pool=reID_pool,  # 历史特征池（detached）
                                reid_model=reid_model,
                                triplet_loss_fn=triplet_loss_fn,
                                config=config,
                                device=device
                            )
                        
                        # 计算可逆性损失（
                        frame_reversibility_loss = None
                        if not config.get("USE_SIMPLE_REID", False) and len(frame_reversibility_losses) > 0:
                            # 使用ReversibleReID模型时，累积所有可逆性损失
                            total_rev_loss1 = sum([loss[0] for loss in frame_reversibility_losses])
                            total_rev_loss2 = sum([loss[1] for loss in frame_reversibility_losses])
                            
                            # 应用可学习权重（如果启用）
                            if reversibility_weight_learner is not None:
                                frame_reversibility_loss, rev_loss_dict = reversibility_weight_learner(
                                    total_rev_loss1, total_rev_loss2
                                )
                                # 记录详细信息
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss1_raw", value=total_rev_loss1.item())
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss2_raw", value=total_rev_loss2.item())
                                metric_log.update(name=f"frame{frame_idx}_reversibility_weight1", value=rev_loss_dict['weight1'])
                                metric_log.update(name=f"frame{frame_idx}_reversibility_weight2", value=rev_loss_dict['weight2'])
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss_weighted", value=frame_reversibility_loss.item())
                            else:
                                # 不使用可学习权重，直接求和
                                frame_reversibility_loss = total_rev_loss1 + total_rev_loss2
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss1", value=total_rev_loss1.item())
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss2", value=total_rev_loss2.item())
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss_total", value=frame_reversibility_loss.item())
                        
                        # 计算当前帧的总损失（加权）
                        if frame_reid_loss is not None or frame_reversibility_loss is not None:
                            # 使用不确定性加权或手动加权
                            if uncertainty_loss_fn is not None:
                                # uncertainty_loss_fn 会自动加权（只处理tracking和reid）
                                frame_total_loss, loss_dict = uncertainty_loss_fn(
                                    frame_tracking_loss, frame_reid_loss if frame_reid_loss is not None else torch.tensor(0.0, device=device)
                                )
                                # 可逆性损失单独加入（不使用uncertainty加权）
                                if frame_reversibility_loss is not None:
                                    frame_total_loss = frame_total_loss + frame_reversibility_loss
                                
                                # 记录权重信息（用于日志）
                                if (i + 1) % 50 == 0 and frame_idx == 0:
                                    weights = uncertainty_loss_fn.get_weights()
                                    logger.show(f"[Iter {i+1}] Uncertainty weights: ω1={weights['omega1']:.3f}, ω2={weights['omega2']:.3f}, "
                                               f"w1={weights['weight1']:.3f}, w2={weights['weight2']:.3f}")
                                    
                                    # 同时输出可逆性权重（如果启用）
                                    if reversibility_weight_learner is not None:
                                        rev_weights = reversibility_weight_learner.get_weights()
                                        logger.show(f"[Iter {i+1}] Reversibility weights: w1={rev_weights['weight1']:.4f}, w2={rev_weights['weight2']:.4f}")
                            else:
                                # 手动加权
                                frame_total_loss = w_track_val * frame_tracking_loss
                                if frame_reid_loss is not None:
                                    frame_total_loss = frame_total_loss + w_reid_val * frame_reid_loss
                                if frame_reversibility_loss is not None:
                                    frame_total_loss = frame_total_loss + frame_reversibility_loss
                        else:
                            # 边界情况：ReID loss 无法计算（样本不够）
                            if uncertainty_loss_fn is not None:
                                # 使用不确定性加权，但 reid_loss = 0
                                frame_total_loss, loss_dict = uncertainty_loss_fn(
                                    frame_tracking_loss, torch.tensor(0.0, device=device)
                                )
                            else:
                                frame_total_loss = w_track_val * frame_tracking_loss
                        
                        # 累积到clip总损失，而不是立即backward
                        if clip_total_loss is None:
                            clip_total_loss = frame_total_loss
                        else:
                            clip_total_loss = clip_total_loss + frame_total_loss
                        
                        # 记录用于日志
                        tracking_loss_vals.append(frame_tracking_loss.detach())
                        if frame_reid_loss is not None:
                            reid_loss_vals.append(frame_reid_loss.detach())
                        
                        # 记录每帧的损失到 metric_log
                        # 1. 记录每帧的详细跟踪损失（box_l1, box_giou, label_focal）
                        for log_k, log_vals in frame_log_dict.items():
                            avg_val = sum(log_vals) / len(log_vals) if log_vals else 0.0
                            metric_log.update(name=log_k, value=avg_val)
                        
                        # 2. 记录每帧的总跟踪损失
                        metric_log.update(name=f"frame{frame_idx}_tracking_loss", value=frame_tracking_loss.item())
                        
                        # 3. 记录每帧的 ReID 损失（仅当ReID训练启用且损失存在时）
                        total_loss_value = frame_tracking_loss.item() * w_track_val
                        
                        if enable_reid_training and frame_reid_loss is not None:
                            metric_log.update(name=f"frame{frame_idx}_reid_loss", value=frame_reid_loss.item())
                            total_loss_value += frame_reid_loss.item() * w_reid_val
                            
                            # 记录 ReID 损失的详细统计信息
                            if reid_stats is not None:
                                metric_log.update(name=f"frame{frame_idx}_reid_total_samples", value=float(reid_stats['num_total_samples']))
                                metric_log.update(name=f"frame{frame_idx}_reid_unique_ids", value=float(reid_stats['num_unique_ids']))
                                metric_log.update(name=f"frame{frame_idx}_reid_multi_view_ids", value=float(reid_stats.get('num_multi_view_ids', 0)))
                                metric_log.update(name=f"frame{frame_idx}_reid_memory_samples", value=float(reid_stats['num_memory_samples']))
                                
                                if 'id_loss' in reid_stats:
                                    metric_log.update(name=f"frame{frame_idx}_reid_id_loss", value=reid_stats['id_loss'])
                                if 'triplet_loss' in reid_stats:
                                    metric_log.update(name=f"frame{frame_idx}_reid_triplet_loss", value=reid_stats['triplet_loss'])
                                if 'cross_clip_loss' in reid_stats:
                                    metric_log.update(name=f"frame{frame_idx}_reid_cross_clip_loss", value=reid_stats['cross_clip_loss'])
                                
                                if 'num_cross_view_pairs' in reid_stats:
                                    metric_log.update(name=f"frame{frame_idx}_reid_cross_view_pairs", value=float(reid_stats['num_cross_view_pairs']))
                                    metric_log.update(name=f"frame{frame_idx}_reid_noise_pairs", value=float(reid_stats['num_noise_pairs']))
                                    
                                    # 计算跨视角正样本比例
                                    total_pairs = reid_stats['num_cross_view_pairs'] + reid_stats['num_noise_pairs']
                                    if total_pairs > 0:
                                        cross_view_ratio = reid_stats['num_cross_view_pairs'] / total_pairs
                                        metric_log.update(name=f"frame{frame_idx}_reid_cross_view_ratio", value=cross_view_ratio)
                        
                        # 记录可逆性损失（如果有）
                        if frame_reversibility_loss is not None:
                            total_loss_value += frame_reversibility_loss.item()
                        
                        # 记录总损失
                        metric_log.update(name=f"frame{frame_idx}_total_loss", value=total_loss_value)
                        
                        # 释放当前帧的部分变量（保留 frame_total_loss 用于累积）
                        del frame_losses, frame_tracking_loss
                        if frame_reid_loss is not None:
                            del frame_reid_loss
                
                # 整个clip处理完后，统一进行backward
                # 梯度会自然累积，每 accumulation_steps 个 batch 才 optimizer.step()
                if clip_total_loss is not None:
                    # 检查损失是否有效
                    if torch.isnan(clip_total_loss) or torch.isinf(clip_total_loss):
                        logger.show(f"[ERROR] Batch {i}, Clip {clip_idx}: Loss is NaN or Inf! "
                                   f"Loss value: {clip_total_loss.item()}")
                        logger.show(f"[ERROR] Skipping backward for this clip...")
                        del clip_total_loss
                        continue
                    
                    clip_total_loss.backward()
                    del clip_total_loss
                
        # 计算用于日志的总损失（与实际backward的损失一致）
        loss = None
        if loss is None:
            # 已经在帧循环内 backward 了，这里只是为了日志
            if tracking_loss_vals:
                total_tracking_loss_val = sum(tracking_loss_vals)  # 这是 tensor
            else:
                total_tracking_loss_val = torch.tensor(0.0, device=device)
            
            # 只有在ReID训练启用时才统计ReID损失
            if enable_reid_training and reid_loss_vals:
                total_reid_loss_val = sum(reid_loss_vals)  # 这是 tensor
            else:
                total_reid_loss_val = torch.tensor(0.0, device=device)
            
            # 计算总损失（与实际backward的损失组成一致）
            if enable_reid_training and total_reid_loss_val.item() > 0:
                # ReID训练启用：总损失 = 跟踪损失 + ReID损失
                loss = total_tracking_loss_val * w_track_val + total_reid_loss_val * w_reid_val
                # 记录各部分损失
                metric_log.update(name="batch_tracking_loss", value=total_tracking_loss_val.item())
                metric_log.update(name="batch_reid_loss", value=total_reid_loss_val.item())
            else:
                # ReID训练未启用：只有跟踪损失
                loss = total_tracking_loss_val * w_track_val
                # 只记录跟踪损失
                metric_log.update(name="batch_tracking_loss", value=total_tracking_loss_val.item())
    
        # Metrics log - 总损失
        metric_log.update(name="total_loss", value=loss.item())
        # loss.backward() # 已经在分步 backward 中做了

        if (i + 1) % accumulation_steps == 0:
            if max_norm > 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if enable_reid_training and reid_model is not None:
                    reid_grad_clip = config.get("REID_GRAD_CLIP", 1.0)  # 默认1.0
                    reid_grad_norm = torch.nn.utils.clip_grad_norm_(reid_model.parameters(), reid_grad_clip)


                if uncertainty_loss_fn is not None:
                    torch.nn.utils.clip_grad_norm_(uncertainty_loss_fn.parameters(), max_norm)
                
                if reversibility_weight_learner is not None:
                    torch.nn.utils.clip_grad_norm_(reversibility_weight_learner.parameters(), max_norm)
            else:
                pass
            
            # 调试：第一个batch时打印参数更新前后的值
            if i == 0 and enable_reid_training and reid_model is not None:
                # 记录更新前的参数值（取第一个参数的一小部分）
                first_param = next(reid_model.parameters())
                param_before = first_param.data[0][:5].clone() if first_param.numel() >= 5 else first_param.data.clone()
                
                # ⭐ 额外调试：检查梯度是否存在
                grad_exists = first_param.grad is not None
                grad_norm = first_param.grad.norm().item() if grad_exists else 0.0
                logger.show(f"[DEBUG] 更新前 - 梯度存在: {grad_exists}, 梯度范数: {grad_norm:.6f}")
            
            optimizer.step()
            
            # 调试：第一个batch时验证参数是否被更新
            if i == 0 and enable_reid_training and reid_model is not None:
                first_param = next(reid_model.parameters())
                param_after = first_param.data[0][:5] if first_param.numel() >= 5 else first_param.data
                param_diff = (param_after - param_before).abs().max().item()
                
                # 获取当前ReID学习率
                reid_lr = None
                if lr_names is not None:
                    for idx, name in enumerate(lr_names):
                        if name == "lr_reid":
                            reid_lr = optimizer.param_groups[idx]["lr"]
                            break
                
                if param_diff > 1e-8:
                    logger.show(f"[DEBUG] ✅ ReID参数已更新！最大变化: {param_diff:.8f}, 当前LR: {reid_lr}")
                else:
                    logger.show(f"[WARNING] ❌ ReID参数未更新！最大变化: {param_diff:.8f}, 当前LR: {reid_lr}")
                    logger.show(f"[WARNING] 可能原因：1) 梯度为0  2) 学习率太小  3) 梯度裁剪过严")
            
            #打印optimizer的梯度大小
            # print(f"Optimizer gradients: {[param.grad for name, param in model.named_parameters()]}") # 打印每个参数的梯度大小
            optimizer.zero_grad()
            # p.step()
        # For logging - log_dict 已经在帧循环内记录了，这里不需要重复
        iter_end_timestamp = time.time()
        metric_log.update(name="time per iter", value=iter_end_timestamp-iter_start_timestamp)
        
        # Outputs logs
        if i % 100 == 0:
            metric_log.sync()
            max_memory = max([torch.cuda.max_memory_allocated(torch.device('cuda', gpu_id))
                            for gpu_id in range(distributed_world_size())]) // (1024**2)
            second_per_iter = metric_log.metrics["time per iter"].avg
            
            # 添加训练模式标识
            mode_tag = "Track+ReID" if enable_reid_training else "Track-Only"
            
            logger.show(head=f"[Epoch={epoch}, Iter={i}, Mode={mode_tag}, "
                            f"{second_per_iter:.2f}s/iter, "
                            f"{i}/{dataloader_len} iters, "
                            f"rest time: {int(second_per_iter * (dataloader_len - i) // 60)} min, "
                            f"Max Memory={max_memory}MB]",
                        log=metric_log)
            
            # 打印多视角ReID统计信息
            if enable_reid_training and 'frame0_reid_total_samples' in metric_log.metrics:
                total_samples = metric_log.metrics.get('frame0_reid_total_samples', None)
                unique_ids = metric_log.metrics.get('frame0_reid_unique_ids', None)
                multi_view_ids = metric_log.metrics.get('frame0_reid_multi_view_ids', None)
                cross_view_pairs = metric_log.metrics.get('frame0_reid_cross_view_pairs', None)
                noise_pairs = metric_log.metrics.get('frame0_reid_noise_pairs', None)
                cross_view_ratio = metric_log.metrics.get('frame0_reid_cross_view_ratio', None)
                
                if total_samples is not None:
                    reid_info = f"ReID Stats: Samples={total_samples.avg:.1f}, UniqueIDs={unique_ids.avg:.1f}"
                    if multi_view_ids is not None:
                        reid_info += f", MultiViewIDs={multi_view_ids.avg:.1f}"
                    if cross_view_ratio is not None:
                        reid_info += f", CrossViewRatio={cross_view_ratio.avg:.2%}"
                    logger.show(f"  {reid_info}")
            
            logger.write(head=f"[Epoch={epoch}, Iter={i}/{dataloader_len}, Mode={mode_tag}]",
                        log=metric_log, filename="log.txt", mode="a")
            logger.tb_add_metric_log(log=metric_log, steps=train_states["global_iters"], mode="iters")

        if multi_checkpoint:
            if i % 100 == 0 and is_main_process():
                save_checkpoint(
                    model=model,
                    path=os.path.join(logger.logdir[:-5], f"checkpoint_{int(i // 100)}.pth")
                )

        train_states["global_iters"] += 1
        
    # Epoch end
    metric_log.sync()
    epoch_end_timestamp = time.time()
    epoch_minutes = int((epoch_end_timestamp - epoch_start_timestamp) // 60)
    logger.show(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                log=metric_log)
    logger.write(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                log=metric_log, filename="log.txt", mode="a")
    logger.tb_add_metric_log(log=metric_log, steps=epoch, mode="epochs")
    
    # Epoch结束：打印ReID loss详细统计（如果启用了ReID训练）
    if enable_reid_training:
        logger.show("="*80)
        logger.show(f"[Epoch {epoch}] ReID训练统计摘要:")
        
        # 1. 各类损失的平均值
        if 'frame0_reid_loss' in metric_log.metrics:
            reid_loss_avg = metric_log.metrics['frame0_reid_loss'].avg
            logger.show(f"  总ReID Loss: {reid_loss_avg:.4f}")
        
        if 'frame0_id_loss' in metric_log.metrics:
            id_loss_avg = metric_log.metrics['frame0_id_loss'].avg
            logger.show(f"  ID分类Loss: {id_loss_avg:.4f}")
        
        if 'frame0_triplet_loss' in metric_log.metrics:
            triplet_loss_avg = metric_log.metrics['frame0_triplet_loss'].avg
            logger.show(f"  Triplet Loss: {triplet_loss_avg:.4f}")
        
        if 'frame0_cross_clip_loss' in metric_log.metrics:
            cross_clip_loss_avg = metric_log.metrics['frame0_cross_clip_loss'].avg
            logger.show(f"  Cross-Clip Loss: {cross_clip_loss_avg:.4f}")
        
        # 2. 样本统计
        if 'frame0_reid_total_samples' in metric_log.metrics:
            samples_avg = metric_log.metrics['frame0_reid_total_samples'].avg
            unique_ids_avg = metric_log.metrics['frame0_reid_unique_ids'].avg
            logger.show(f"  平均样本数: {samples_avg:.1f}, 平均唯一ID数: {unique_ids_avg:.1f}")
        
        if 'frame0_reid_multi_view_ids' in metric_log.metrics:
            multi_view_avg = metric_log.metrics['frame0_reid_multi_view_ids'].avg
            cross_view_ratio_avg = metric_log.metrics['frame0_reid_cross_view_ratio'].avg
            logger.show(f"  多视角ID数: {multi_view_avg:.1f}, 跨视角正样本比例: {cross_view_ratio_avg:.2%}")
        
        logger.show("="*80)

    return


def get_param_groups(config: dict, model: nn.Module) -> Tuple[List[Dict], List[str]]:
    """
    用于针对不同部分的参数使用不同的 lr 等设置
    Args:
        config: 实验的配置信息
        model: 需要训练的模型

    Returns:
        params_group: a list of params groups.
        lr_names: a list of params groups' lr name, like "lr_backbone".
    """
    def match_keywords(name: str, keywords: List[str]):
        matched = False
        for keyword in keywords:
            if keyword in name:
                matched = True
                break
        return matched
    # keywords
    backbone_keywords = ["backbone.backbone"]
    points_keywords = ["reference_points", "sampling_offsets"]  # 在 transformer 中用于选取参考点和采样点的网络参数关键字
    query_updater_keywords = ["query_updater"]
    param_groups = [
        {   # backbone 学习率设置
            "params": [p for n, p in model.named_parameters() if match_keywords(n, backbone_keywords) and p.requires_grad],
            "lr": config["LR_BACKBONE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, points_keywords)
                       and p.requires_grad],
            "lr": config["LR_POINTS"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, query_updater_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        },
        {
            "params": [p for n, p in model.named_parameters() if not match_keywords(n, backbone_keywords)
                       and not match_keywords(n, points_keywords)
                       and not match_keywords(n, query_updater_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        }
    ]
    return param_groups, ["lr_backbone", "lr_points", "lr_query_updater", "lr"]


def get_param_groups_reid(config, model):
    # 使用ReID独立学习率（如果配置了的话）
    lr_reid = config.get("LR_REID", config["LR"])  # 默认使用主模型LR
    lr_classifier = config.get("LR_CLASSIFIER", lr_reid)
    lr_bottleneck = config.get("LR_BOTTLENECK", lr_reid)

    # 精确取参数（不会误匹配）
    classifier_params = list(model.classifier.parameters())
    bottleneck_params = list(model.bottleneck.parameters())

    # 其他 = 全部参数 - 上面两类
    used = set(id(p) for p in classifier_params + bottleneck_params)
    other_params = [p for p in model.parameters() if id(p) not in used]

    param_groups = [
        {"params": other_params, "lr": lr_reid},  # ⭐ 使用lr_reid
        {"params": bottleneck_params, "lr": lr_bottleneck},
        {"params": classifier_params, "lr": lr_classifier},
    ]
    lr_names = ["lr_reid", "lr_bottleneck", "lr_classifier"]  # ⭐ 改名以区分
    return param_groups, lr_names
