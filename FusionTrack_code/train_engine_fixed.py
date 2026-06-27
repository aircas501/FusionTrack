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
        # without MemoryBank, return window_size zero vectors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return [torch.zeros(feat_dim, device=device) for _ in range(window_size)]
    
    # take last window_size frames from MemoryBank deque
    # last deque element is the newest frame
    recent_frames = list(memory_bank.deque)[-window_size:] if len(memory_bank.deque) >= window_size else list(memory_bank.deque)
    
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # fixed-length list filled with zero vectors
    history_queries = [torch.zeros(feat_dim, device=device) for _ in range(window_size)]
    
    # start index: pad from the end if history < window_size
    # e.g. window_size=5 with 3 frames -> indices [2,3,4]
    start_idx = window_size - len(recent_frames)
    
    # fill recent_frames into aligned slots
    for i, frame_data in enumerate(recent_frames):
        views = frame_data.get("views", {})
        if view_name in views:
            track_data = views[view_name].get(track_id, None)
            if track_data is not None and "feat" in track_data:
                # feat is query feature
                query_feat = track_data["feat"]
                if isinstance(query_feat, torch.Tensor):
                    # fill slot preserving temporal order
                    history_queries[start_idx + i] = query_feat.clone().to(device)
    
    # key: fixed-length list; missing frames stay zero
    # ReID model treats zero vectors as mask=False
    return history_queries


def calculate_reid_loss_with_current_and_memory(current_features, reID_pool, reid_model, triplet_loss_fn, config, device):

    import torch.nn.functional as F
    import random
    from models.utils import get_model
    
    if len(current_features) == 0:
        return None, None
    
    reid_model_unwrapped = get_model(reid_model)
    
    # ========================================
    # Step 1: group by ID (cross-view positives)
    # ========================================
    features_by_id = {}  # {track_id: [(view, feat), ...]}
    
    for (view, track_id), reid_feat in current_features.items():
        if track_id not in features_by_id:
            features_by_id[track_id] = []
        features_by_id[track_id].append((view, reid_feat))
    
    # ========================================
    # Part 1: ID classification loss (all samples)
    # ========================================
    current_id_loss = None
    cls_score_list = []
    gt_labels_list = []
    
    # number of classifier classes
    num_classes = None
    if hasattr(reid_model_unwrapped, 'classifier'):
        if hasattr(reid_model_unwrapped.classifier, 'out_features'):
            num_classes = reid_model_unwrapped.classifier.out_features
        elif hasattr(reid_model_unwrapped.classifier, 'num_classes'):
            num_classes = reid_model_unwrapped.classifier.num_classes
    
    # all views for all IDs
    for track_id, view_feat_list in features_by_id.items():
        # check track_id within class range
        if num_classes is not None and track_id >= num_classes:
            continue
        
        # classification loss per view feature
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
    # Part 2: Triplet loss (prefer cross-view positives)
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
            # pick positive and negative for each anchor
            for anchor_idx, (anchor_view, anchor_feat) in enumerate(anchor_view_feat_list):
                
                # positive: same ID, different view
                positive_candidates = [
                    (v, f) for v, f in anchor_view_feat_list 
                    if v != anchor_view  # different view
                ]
                
                if len(positive_candidates) > 0:
                    # real cross-view positive (higher quality)
                    positive_view, positive_feat = random.choice(positive_candidates)
                    num_cross_view_pairs += 1
                else:
                    # fallback: noise augmentation (lower quality)
                    positive_feat = anchor_feat + torch.randn_like(anchor_feat) * 0.1
                    num_noise_pairs += 1
                
                # negative: any view of a different ID
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
                
                # compute distances
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
    # Part 3: cross-frame contrastive loss vs history
    # ========================================
    cross_clip_loss = None
    num_memory_samples = 0
    
    if reID_pool is not None and len(current_features) > 0:
        # collect historical features from ReIDPool
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
            # current-frame feature matrix (N, C)
            curr_ids_list = []
            curr_feats_list = []
            
            for (view, track_id), reid_feat in current_features.items():
                curr_ids_list.append(track_id)
                curr_feats_list.append(reid_feat)
            
            curr_feats_tensor = torch.stack(curr_feats_list)  # (N, C)
            curr_ids_tensor = torch.tensor(curr_ids_list, device=device)
            
            # history feature matrix (M, C)
            mem_feats_tensor = torch.stack(mem_feats_list)  # (M, C)
            mem_ids_tensor = torch.tensor(mem_ids_list, device=device)
            
            # normalize
            curr_feats_norm = F.normalize(curr_feats_tensor, p=2, dim=1)  # (N, C)
            mem_feats_norm = F.normalize(mem_feats_tensor, p=2, dim=1)    # (M, C)
            
            # similarity matrix (N, M)
            sim_matrix = torch.mm(curr_feats_norm, mem_feats_norm.t())  # (N, M)
            
            # positive/negative masks
            pos_mask = curr_ids_tensor.unsqueeze(1) == mem_ids_tensor.unsqueeze(0)  # (N, M)
            neg_mask = ~pos_mask  # (N, M)
            
            # hardest positive and hardest negative
            margin = config.get("CROSS_CLIP_MARGIN", 0.5)
            
            pos_sim_masked = sim_matrix.clone()
            pos_sim_masked[~pos_mask] = 1e9
            min_pos_sim, _ = pos_sim_masked.min(dim=1)  # (N,)
            
            valid_row_mask = (pos_mask.sum(dim=1) > 0)  # (N,)
            
            neg_sim_masked = sim_matrix.clone()
            neg_sim_masked[~neg_mask] = -1e9
            max_neg_sim, _ = neg_sim_masked.max(dim=1)  # (N,)
            
            # triplet loss
            losses = F.relu(margin - min_pos_sim + max_neg_sim)  # (N,)
            
            if valid_row_mask.sum() > 0:
                cross_clip_loss = losses[valid_row_mask].mean()
    
    # ========================================
    # combine losses
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
    
    # statistics
    num_total_samples = sum(len(view_feat_list) for view_feat_list in features_by_id.values())
    num_unique_ids = len(features_by_id)
    
    # count multi-view IDs
    num_multi_view_ids = sum(1 for view_feat_list in features_by_id.values() if len(view_feat_list) > 1)
    
    stats_dict = {
        'num_current_samples': num_total_samples,  # legacy key name
        'num_total_samples': num_total_samples,
        'num_unique_ids': num_unique_ids,
        'num_multi_view_ids': num_multi_view_ids,
        'num_memory_samples': num_memory_samples,
        **loss_components
    }
    
    # triplet stats: positive source counts
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
        
        # pick negative (different ID)
        negative_ids = [id for id in unique_ids if id != anchor_id]
        if len(negative_ids) == 0:
            continue
        
        negative_id = random.choice(negative_ids)
        negative_feat = features_dict[negative_id]
        
        # positive: same-ID feature + small noise
        positive_feat = anchor_feat + torch.randn_like(anchor_feat) * 0.1
        
        # Normalize
        if normalize_feature:
            anchor_feat = F.normalize(anchor_feat, p=2, dim=-1)
            positive_feat = F.normalize(positive_feat, p=2, dim=-1)
            negative_feat = F.normalize(negative_feat, p=2, dim=-1)
        
        # compute distances
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

    model = build_model(config=config)  # MeMOTR-based FusionTrack with ReID

    # Load pretrained weights (only when not resuming)
    # PRETRAINED_MODEL and RESUME1 should be mutually exclusive
    # - PRETRAINED_MODEL: initial pretrain (e.g. COCO backbone)
    # - RESUME1: resume from saved checkpoint
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
                                        config=config)  # dataset load can be slow

    # multi-view dataset check
    multiview_datasets = ["UAV_V", "CAMPUS", "WILDTRACK", "MvMHAT", "DIVOTrack"]
    if config["DATASET"] in multiview_datasets:
        #viewpoints = dataset_train.viewpoints
        # viewpoints elsewhere are scene-specific; ReID uses config VIEW_POINT count

        viewpoints = ["c00"+str(i+1) for i in range(config["VIEW_POINT"])]
        criterion = {}
        
        # ==================== Staged ReID training ====================
        # first N epochs: tracking only, then ReID
        reid_start_epoch = config.get("REID_START_EPOCH", 0)  # 0 = train ReID from epoch 0
        print("="*80)
        if reid_start_epoch > 0:
            print(f"[Staged training] Training tracking only for the first {reid_start_epoch} epochs")
            print(f"[Staged training] ReID training starts from epoch {reid_start_epoch}")
        else:
            print(f"[Staged training] Training tracking and ReID jointly from epoch 0")
        print("="*80)
        
        # ==================== ReID model choice (two options) ====================
        # 1. SimpleReIDModel: lightweight MLP
        # 2. ReversibleReIDModel: weight-shared reversible (recommended)
        use_simple_reid = config.get("USE_SIMPLE_REID", False)
        
        if use_simple_reid:
            print(f"[ReID model] Using SimpleReIDModel")
            reid_model = build_simple_reid_model(config=config)
            # move SimpleReIDModel to GPU
            if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
                reid_model = reid_model.to(device=torch.device(config["DEVICE"], distributed_rank()))
            else:
                reid_model = reid_model.to(device=torch.device(config["DEVICE"]))
        else:
            print(f"[ReID model] Using ReversibleReIDModel (weight-shared reversible variant)")
            reid_model = build_reversible_reid(config=config)  # device handled internally
        
        # ==================== Freeze ReID at init if needed ====================
        if reid_start_epoch > 0:
            print(f"[Param freeze] ReID model params frozen at init (no updates for first {reid_start_epoch} epochs)")
            for param in reid_model.parameters():
                param.requires_grad = False
            reid_model.eval()  # eval mode

        # build ReIDPool to hold batch ReID features
        keep_all_batch_reid = config.get("KEEP_ALL_BATCH_REID", True)  # currently unused
        reID_pool = build_reid_pool(views=viewpoints, max_forget_length = config["MAX_FORGET_LENGTH"], 
                                    training = True, keep_all_batch_reid=keep_all_batch_reid,reid_update_weight=config.get("REID_UPDATE_WEIGHT", 0.8))
        
        # init MemoryBank
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
        
        # init ReID loss modules
        # 1. triplet loss for ReID features
        triplet_loss_fn = build_triplet_loss(config=config) if config.get("USE_TRIPLET_LOSS", False) else None
        
        # 2. uncertainty weighting for tracking vs ReID
        #    meta-learned task weights
        use_uncertainty_loss = config.get("USE_UNCERTAINTY_LOSS", True)
        uncertainty_loss_fn = None
        if use_uncertainty_loss:
            uncertainty_loss_fn = build_uncertainty_loss(config=config)  # weights applied in forward
            uncertainty_loss_fn = uncertainty_loss_fn.to(device=torch.device("cuda", distributed_rank()) if torch.cuda.is_available() else torch.device("cpu"))
        
        # 3. learnable weights for two reversibility losses
        #    when USE_REVERSIBILITY_LOSS and ReversibleReID
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
        
        for view in viewpoints:  # one criterion per view
            # Criterion
            criterion_one = build_criterion(config=config)
            criterion_one.set_device(torch.device("cuda", distributed_rank()))
            criterion[view] = criterion_one
    else:
        criterion = build_criterion(config=config)
    
    # Optimizer
    param_groups, lr_names = get_param_groups(config=config, model=model)  # per-module LRs


    # multi-view dataset check
    multiview_datasets = ["UAV_V", "CAMPUS", "WILDTRACK", "MvMHAT", "DIVOTrack"]
    if config["DATASET"] in multiview_datasets: 
        param_groups_reid, lr_names_reid = get_param_groups_reid(config=config, model=reid_model)
        param_groups += param_groups_reid
        lr_names += lr_names_reid
        
        # add uncertainty loss params to optimizer
        if uncertainty_loss_fn is not None:
            param_groups.append({
                'params': uncertainty_loss_fn.parameters(),
                'lr': config.get("LR", 2.0e-4),  # main LR
                'name': 'uncertainty_loss'
            })
            lr_names.append('uncertainty_loss')
        
        # add reversibility weight learner to optimizer
        if reversibility_weight_learner is not None:
            param_groups.append({
                'params': reversibility_weight_learner.parameters(),
                'lr': config.get("LR_REVERSIBILITY_WEIGHT", config.get("LR", 2.0e-4)),  # main or dedicated LR
                'name': 'reversibility_weight'
            })
            lr_names.append('reversibility_weight')

    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    
    # ==================== Verify optimizer param groups (debug) ====================
    if config["DATASET"] in multiview_datasets:
        print("="*80)
        print("[Optimizer init verification]")
        print(f"Total param groups: {len(optimizer.param_groups)}")
        for i, (group, name) in enumerate(zip(optimizer.param_groups, lr_names)):
            num_params = len(group['params'])
            lr = group['lr']
            print(f"  Group {i} ({name:25s}): {num_params:4d} params, lr={lr:.2e}")
            
            # extra check: empty ReID param groups
            if 'reid' in name.lower() or 'base' in name.lower() or 'bottleneck' in name.lower() or 'classifier' in name.lower():
                if num_params == 0:
                    print(f"    WARNING: ReID param group '{name}' is empty!")
                else:
                    # check requires_grad on params
                    num_trainable = sum(1 for p in group['params'] if p.requires_grad)
                    num_frozen = num_params - num_trainable
                    print(f"    Trainable: {num_trainable}, frozen: {num_frozen}")
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
    
    # load ReID model if RESUME2 set
    if config["DATASET"] in multiview_datasets and config.get("RESUME2") is not None and config["RESUME2"] not in [None, 'None', '', 'null']:
        print(f"[INFO] Loading ReID model from checkpoint: {config['RESUME2']}")
        load_checkpoint(model=reid_model, path=config["RESUME2"], states=train_states)

    # Set start epoch
    start_epoch = train_states["start_epoch"]

    # ==================== DDP wrap ====================
    reid_start_epoch = config.get("REID_START_EPOCH", 0)
    
    if is_distributed():
        # main model: always has trainable params -> DDP
        model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=True)
        
        # ReID: DDP depends on staged training
        if config["DATASET"] in multiview_datasets:
            has_trainable_params = any(p.requires_grad for p in reid_model.parameters())
            
            if has_trainable_params:
                # trainable params -> wrap DDP (REID_START_EPOCH=0)
                reid_model = DDP(module=reid_model, device_ids=[distributed_rank()], find_unused_parameters=True)
                print(f"[DDP init] ReID model wrapped with DDP (rank={distributed_rank()})")
            else:
                # frozen: GPU only; wrap DDP later in epoch loop
                reid_model = reid_model.to(torch.device(f"cuda:{distributed_rank()}"))
                print(f"[DDP init] ReID model frozen, moved to GPU only (rank={distributed_rank()})")
                print(f"[DDP init] Will wrap with DDP dynamically at epoch={reid_start_epoch}")

    multi_checkpoint = "MULTI_CHECKPOINT" in config and config["MULTI_CHECKPOINT"]

    # Training:
    
    for epoch in range(start_epoch, config["EPOCHS"]):
        # set epoch for distributed sampler and aug (no rebuild)
        if is_distributed():
            sampler_train.set_epoch(epoch)
        dataset_train.set_epoch(epoch)

        # query_updater-only mode: freeze backbone/points/other
        if epoch >= config["ONLY_TRAIN_QUERY_UPDATER_AFTER"]:
            # use lr_names for dynamic LR groups (avoid hard-coded indices)
            for idx, name in enumerate(lr_names):
                if name in ["lr_backbone", "lr_points", "lr"] and name != "lr_query_updater":
                    optimizer.param_groups[idx]["lr"] = 0.0
            # ReID param groups unchanged
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
        
        # ==================== Staged ReID: enable/disable ====================
        reid_start_epoch = config.get("REID_START_EPOCH", 0)
        enable_reid_training = (epoch >= reid_start_epoch)
        
        # ReID LR warmup from reid_start_epoch (linear)
        reid_warmup_epochs = config.get("REID_WARMUP_EPOCHS", 0)
        if enable_reid_training and reid_warmup_epochs > 0:
            epochs_since_reid_start = epoch - reid_start_epoch
            if epochs_since_reid_start < reid_warmup_epochs:
                # warmup: LR 0 -> target linearly
                warmup_factor = (epochs_since_reid_start + 1) / reid_warmup_epochs
                
                # adjust ReID LR groups
                for idx, name in enumerate(lr_names):
                    if name in ["lr_reid", "lr_bottleneck", "lr_classifier"]:
                        # target LR from config
                        if name == "lr_reid":
                            target_lr = config.get("LR_REID", config["LR"])
                        elif name == "lr_bottleneck":
                            target_lr = config.get("LR_BOTTLENECK", config.get("LR_REID", config["LR"]))
                        elif name == "lr_classifier":
                            target_lr = config.get("LR_CLASSIFIER", config.get("LR_REID", config["LR"]))
                        
                        # apply warmup factor
                        optimizer.param_groups[idx]["lr"] = target_lr * warmup_factor
                
                train_logger.show(f"[Epoch {epoch}] ReID LR warmup: {warmup_factor:.2%} "
                                f"(warmup epoch {epochs_since_reid_start+1}/{reid_warmup_epochs})")
        
        # at reid_start_epoch: unfreeze ReID and wrap DDP
        if reid_start_epoch > 0 and epoch == reid_start_epoch:
            train_logger.show(f"="*80)
            train_logger.show(f"[Epoch {epoch}] Starting ReID model training!")
            train_logger.write(f"[Epoch {epoch}] Starting ReID model training!")
            
            # Step 1: unfreeze ReID params
            train_logger.show(f"[Epoch {epoch}]   Step 1/3: Unfreezing ReID params...")
            for param in reid_model.parameters():
                param.requires_grad = True
            
            # Step 2: wrap DDP in distributed mode
            if is_distributed():
                train_logger.show(f"[Epoch {epoch}]   Step 2/3: Wrapping DDP (rank={distributed_rank()})...")
                # reassign reid_model here
                reid_model = DDP(module=reid_model, device_ids=[distributed_rank()], find_unused_parameters=True)
                train_logger.show(f"[Epoch {epoch}]   ReID model wrapped with DDP successfully")
            else:
                train_logger.show(f"[Epoch {epoch}]   Step 2/3: Non-distributed mode, skipping DDP wrap")
            
            # Step 3: train mode
            train_logger.show(f"[Epoch {epoch}]   Step 3/3: Setting train mode...")
            reid_model.train()
            
            # Step 4: verify ReID params in optimizer
            train_logger.show(f"[Epoch {epoch}]   Checking optimizer params...")
            
            # after DDP, use get_model for underlying module
            reid_model_for_check = get_model(reid_model)
            reid_params_set = set(reid_model_for_check.parameters())
            
            reid_params_in_optimizer = sum(1 for group in optimizer.param_groups 
                                           for p in group['params'] 
                                           if p in reid_params_set)
            train_logger.show(f"[Epoch {epoch}]   ReID params in optimizer: {reid_params_in_optimizer}")
            
            # check requires_grad
            num_trainable = sum(1 for p in reid_params_set if p.requires_grad)
            train_logger.show(f"[Epoch {epoch}]   ReID trainable params: {num_trainable}")
            
            train_logger.show(f"[Epoch {epoch}] ReID model fully enabled!")
            train_logger.show(f"="*80)
        
        # log current epoch training mode
        if epoch < reid_start_epoch:
            train_logger.show(f"[Epoch {epoch}] Current mode: tracking only (ReID frozen)")
        else:
            train_logger.show(f"[Epoch {epoch}] Current mode: joint tracking and ReID training")

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
            enable_reid_training=enable_reid_training,  # whether to train ReID
            memory_bank=memory_bank,
            triplet_loss_fn=triplet_loss_fn,
            uncertainty_loss_fn=uncertainty_loss_fn,
            reversibility_weight_learner=reversibility_weight_learner,  # reversibility weight learner
            config=config,
            lr_names=lr_names  # LR group names
        )
        scheduler.step()
        train_states["start_epoch"] += 1
        if multi_checkpoint is True:
            pass
        else:
            if config["DATASET"] == "DanceTrack" or config["EPOCHS"] < 100 or (epoch + 1) % 2 == 0:
                # save main tracking model
                save_checkpoint(
                    model=model,
                    path=os.path.join(config["OUTPUTS_DIR"], f"model_checkpoint_{epoch}.pth"),
                    states=train_states,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                
                # save ReID model (multi-view datasets)
                if config["DATASET"] in multiview_datasets and enable_reid_training:
                    # inference needs weights only, not optimizer/scheduler
                    save_checkpoint(
                        model=reid_model,
                        path=os.path.join(config["OUTPUTS_DIR"], f"reid_checkpoint_{epoch}.pth"),
                        states=train_states
                        # omit optimizer/scheduler for inference checkpoints
                    )
                # # load main model
                # load_checkpoint(model=model, path="model_checkpoint_X.pth")

                # # load ReID model (if needed)
                # load_checkpoint(model=reid_model, path="reid_checkpoint_X.pth")
    return


def train_one_epoch(model: FusionTrack, train_states: dict, max_norm: float,
                    dataloader: DataLoader, criterion: ClipCriterion | dict, optimizer: torch.optim,
                    epoch: int, logger: Logger,
                    accumulation_steps: int = 1, use_dab: bool = False,
                    multi_checkpoint: bool = False,
                    no_grad_frames: int | None = None, reid_model: nn.Module = None , reID_pool: ReIDPool = None,
                    enable_reid_training: bool = True,  # enable ReID training
                    memory_bank: MemoryBank = None, triplet_loss_fn = None, uncertainty_loss_fn = None, 
                    reversibility_weight_learner = None,  # reversibility weight learner
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
    
    # set ReID train/eval from enable_reid_training
    if reid_model is not None:
        if enable_reid_training:
            reid_model.train()
            # confirm params are trainable
            trainable_params = sum(p.requires_grad for p in reid_model.parameters())
            if trainable_params == 0:
                logger.show(f"[WARNING] ReID model has no trainable params! All requires_grad=False")
        else:
            # when ReID disabled, use eval mode
            reid_model.eval()
            logger.show(f"[INFO] ReID model set to eval mode (ReID not trained this epoch)")
    
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device

    dataloader_len = len(dataloader)
    metric_log = MetricLog()
    epoch_start_timestamp = time.time()
    
    cur_frame_view = [0 for _ in criterion.keys()]

    for i, batch in enumerate(dataloader):
        # record batch start time for iter timing
        
        iter_start_timestamp = time.time()
        
        # clear MemoryBank and ReIDPool each batch
        if memory_bank is not None:
            memory_bank.clear()
        if reID_pool is not None:
            reID_pool.clear_all()  # clear ReID features and life counters
        
        if config["DATASET"] == "UAV_V":
            loss_view_dict = {}
            cls_score_list = []
            global_feat_list = []
            gt_labels_list = []
            
            # check batch layout: multi-clip or not
            is_multiclip = 'clips' in batch
            if is_multiclip:
                clips = batch['clips']
            else:
                # backward compat: wrap single clip
                clips = [batch]
            
            # manual loss weights when uncertainty loss off
            w_track_val = 1.0
            w_reid_val = config.get("ID_LOSS_WEIGHT", config.get("REID_LOSS_WEIGHT", 0.5))
            
            # detached losses for logging
            tracking_loss_vals = []
            reid_loss_vals = []

            # iterate clips
            for clip_idx, clip_batch in enumerate(clips):
                # optional: clear MemoryBank at clip start
                if memory_bank is not None and clip_idx > 0:
                    memory_bank.clear()
                
                # reset frame counter per clip
                cur_frame_view = [0 for _ in criterion.keys()]
                
                # accumulate loss over full clip
                clip_total_loss = None
                
                # init tracks and criterion per view
                all_view_tracks = {}
                all_view_batches = {}
                for view_idx, view in enumerate(criterion.keys()):  # init per view
                    view_criterion = criterion[view]
                    view_batch = clip_batch[view]
                    all_view_batches[view] = view_batch
                    
                    # init tracks
                    tracks = TrackInstances.init_tracks(batch=view_batch,
                                                        hidden_dim=get_model(model).hidden_dim,
                                                        num_classes=get_model(model).num_classes,
                                                        device=device, use_dab=use_dab)
                    all_view_tracks[view] = tracks
                    
                    # init criterion
                    view_criterion.init_a_clip(batch=view_batch,
                                        hidden_dim=get_model(model).hidden_dim,
                                        num_classes=get_model(model).num_classes,
                                        device=device)

                # frame count (same across views; use first)
                first_view = list(criterion.keys())[0]
                num_frames = len(all_view_batches[first_view]["imgs"][0])
                
                # process frame by frame
                for frame_idx in range(num_frames):
                    # accumulate per-view losses for this frame
                    frame_losses = []
                    frame_log_dict = {}  # detailed per-frame loss log
                    
                    # collect current-frame ReID features (keep grad)
                    # key (view, track_id) avoids cross-view ID collision
                    current_frame_reid_features = {}  # {(view, track_id): reid_feat} 
                    
                    # tracks/ReID feats for batch MemoryBank update
                    frame_all_view_tracks = {}  # {view: tracks}
                    frame_all_reid_features = {}  # {(view, id): reid_feat}
                    frame_reversibility_losses = []  # collect reversibility losses
                    
                    # 1. forward and process per view
                    for view_idx, view in enumerate(criterion.keys()):
                        view_criterion = criterion[view]
                        view_batch = all_view_batches[view]
                        tracks = all_view_tracks[view]
                        
                        if no_grad_frames is None or frame_idx >= no_grad_frames:
                            frame = [fs[frame_idx] for fs in view_batch["imgs"]]
                            for f in frame:
                                f.requires_grad_(False)
                            frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
                            
                            # view ID from infos (batch_size=1)
                            view_id_tensor = None
                            if len(view_batch["infos"]) > 0 and frame_idx < len(view_batch["infos"][0]):
                                # view_id from first sample, first frame
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
                            
                            # use output_embed from process_single_frame (with grad)
                            id_to_feature_map = {}
                            if config["REID_LOSS"] and enable_reid_training:
                                # 1. existing track features (previous_tracks updated)
                                for b_tracks in previous_tracks:
                                    for track_idx, track_id in enumerate(b_tracks.ids):
                                        if track_id.item() >= 0:
                                            # output_embed slice from transformer, with grad
                                            id_to_feature_map[track_id.item()] = b_tracks.output_embed[track_idx]
                                
                                # 2. new track features (output_embed set in process_single_frame)
                                for b_tracks in new_tracks:
                                    for track_idx, track_id in enumerate(b_tracks.ids):
                                        if track_id.item() >= 0:
                                            id_to_feature_map[track_id.item()] = b_tracks.output_embed[track_idx]
                            
                            if frame_idx < len(view_batch["imgs"][0]) - 1:
                                tracks_updated = get_model(model).postprocess_single_frame(
                                    previous_tracks, new_tracks, unmatched_dets)
                                # update all_view_tracks for next frame
                                all_view_tracks[view] = tracks_updated
                            else:
                                # merge last frame too for feature extraction
                                tracks_updated = get_model(model).postprocess_single_frame(
                                    previous_tracks, new_tracks, unmatched_dets)
                            
                            # extract ReID features via lookup table
                            reid_features_dict = {}
                            if config["REID_LOSS"] and enable_reid_training:
                                for t_idx, track in enumerate(tracks_updated):
                                    for local_idx, track_id in enumerate(track.ids):
                                        tid = track_id.item()
                                        if tid >= 0:
                                            if tid in id_to_feature_map:
                                                # lookup transformer output feature
                                                query_feat = id_to_feature_map[tid]
                                                
                                                # ReID input: single- or multi-frame query
                                                use_multiframe_reid = config.get("USE_MULTIFRAME_REID", False)
                                                
                                                if use_multiframe_reid and memory_bank is not None:
                                                    # multi-frame query list
                                                    window_size = config.get("REID_QUERY_WINDOW_SIZE", 5)
                                                    history_queries = extract_multiframe_queries_from_memorybank(
                                                        memory_bank, tid, view, window_size
                                                    )
                                                    
                                                    # history queries + current query
                                                    # history_queries: List[Tensor(dim)] up to window_size
                                                    query_list = history_queries + [query_feat]  # append current frame
                                                    query_input = query_list  # pass list to ReID model
                                                else:
                                                    # single-frame query (legacy)
                                                    query_input = [query_feat.unsqueeze(0)]
                                                
                                                # ReID model -> embedding
                                                with torch.set_grad_enabled(True):
                                                    use_simple_reid = config.get("USE_SIMPLE_REID", False)
                                                    
                                                    if use_simple_reid:
                                                        # SimpleReID train mode: (cls_score, reid_feat)
                                                        _, reid_feat = get_model(reid_model)(query_input, None)
                                                    else:
                                                        # ReversibleReID train mode: (reid_feat, rev_loss1, rev_loss2)
                                                        reid_output = reid_model(query_input, None)
                                                        if isinstance(reid_output, tuple) and len(reid_output) == 3:
                                                            # train mode: unpack 3 values
                                                            reid_feat, rev_loss1, rev_loss2 = reid_output
                                                            # collect reversibility losses
                                                            frame_reversibility_losses.append((rev_loss1, rev_loss2))
                                                        else:
                                                            # eval mode: feature only
                                                            reid_feat = reid_output
                                                    
                                                    # normalize return format
                                                    if isinstance(reid_feat, tuple):
                                                        reid_feat = reid_feat[1]
                                                    if reid_feat.dim() > 1:
                                                        reid_feat = reid_feat[0]
                                                    
                                                    # L2 normalize (match TripletLoss)
                                                    reid_feat = torch.nn.functional.normalize(reid_feat, p=2, dim=-1)
                                                    
                                                    reid_features_dict[tid] = reid_feat  # per-view dict
                                                    current_frame_reid_features[(view, tid)] = reid_feat  # composite key avoids overwrite
                            
                            # tracks after postprocess
                            if frame_idx < len(view_batch["imgs"][0]) - 1:
                                tracks = tracks_updated
                        else:
                            with torch.no_grad():
                                frame = [fs[frame_idx] for fs in view_batch["imgs"]]
                                for f in frame:
                                    f.requires_grad_(False)
                                frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
                                
                                # view ID
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
                                
                                # {track_id: feature} map (no_grad, output_embed)
                                id_to_feature_map = {}
                                if config["REID_LOSS"] and enable_reid_training:
                                    # 1. collect existing track features
                                    for b_tracks in previous_tracks:
                                        for track_idx, track_id in enumerate(b_tracks.ids):
                                            if track_id.item() >= 0:
                                                id_to_feature_map[track_id.item()] = b_tracks.output_embed[track_idx]
                                    
                                    # 2. collect new track features
                                    for b_tracks in new_tracks:
                                        for track_idx, track_id in enumerate(b_tracks.ids):
                                            if track_id.item() >= 0:
                                                id_to_feature_map[track_id.item()] = b_tracks.output_embed[track_idx]
                                
                                # postprocess
                                if frame_idx < len(view_batch["imgs"][0]) - 1:
                                    tracks_updated = get_model(model).postprocess_single_frame(
                                        previous_tracks, new_tracks, unmatched_dets, no_augment=frame_idx < no_grad_frames-1)
                                    all_view_tracks[view] = tracks_updated
                                else:
                                    # merge last frame too for feature extraction
                                    tracks_updated = get_model(model).postprocess_single_frame(
                                        previous_tracks, new_tracks, unmatched_dets, no_augment=False)
                                
                                # extract ReID for valid IDs only (-1 = unmatched)
                                reid_features_dict = {}
                                if config["REID_LOSS"] and enable_reid_training:
                                    for t_idx, track in enumerate(tracks_updated):
                                        for local_idx, track_id in enumerate(track.ids):
                                            tid = track_id.item()
                                            if tid >= 0:
                                                if tid in id_to_feature_map:
                                                    query_feat = id_to_feature_map[tid]
                                                    
                                                    # ReID input: single- or multi-frame query
                                                    use_multiframe_reid = config.get("USE_MULTIFRAME_REID", False)
                                                    
                                                    if use_multiframe_reid and memory_bank is not None:
                                                        # multi-frame query list
                                                        window_size = config.get("REID_QUERY_WINDOW_SIZE", 5)
                                                        history_queries = extract_multiframe_queries_from_memorybank(
                                                            memory_bank, tid, view, window_size
                                                        )
                                                        
                                                        # history queries + current query
                                                        query_list = history_queries + [query_feat]
                                                        query_input = query_list  # pass list to ReID model
                                                    else:
                                                        # single-frame query (legacy)
                                                        query_input = [query_feat.unsqueeze(0)]
                                                    
                                                    use_simple_reid = config.get("USE_SIMPLE_REID", False)
                                                    
                                                    # both models return features only in no_grad
                                                    reid_feat = get_model(reid_model)(query_input, None)
                                                    
                                                    # handle return value
                                                    if isinstance(reid_feat, tuple):
                                                        reid_feat = reid_feat[1]
                                                    if reid_feat.dim() > 1:
                                                        reid_feat = reid_feat[0]
                                                    reid_features_dict[tid] = reid_feat
                                
                                # tracks after postprocess
                                if frame_idx < len(view_batch["imgs"][0]) - 1:
                                    tracks = tracks_updated
                        
                        if config["REID_LOSS"] and enable_reid_training:
                            # ReIDPool stores history; detach to save memory
                            reid_features_detached = {k: v.detach() for k, v in reid_features_dict.items()} if len(reid_features_dict) > 0 else None
                            
                            # update pool with detached features
                            reID_pool.update_pool(view, tracks, cur_frame_view[view_idx] + frame_idx, 
                                                reid_features=reid_features_detached)
                            
                            # collect tracks/ReID for batch MemoryBank update
                            # MemoryBank also needs detached feats
                            if memory_bank is not None:
                                frame_all_view_tracks[view] = tracks
                                if len(reid_features_dict) > 0:
                                    for track_id, reid_feat in reid_features_dict.items():
                                        # detach features stored in MemoryBank
                                        frame_all_reid_features[(view, track_id)] = reid_feat.detach()
                        
                    # per-view loss
                    loss_dict, log_dict = view_criterion.get_mean_by_n_gts()
                    view_loss = view_criterion.get_sum_loss_dict(loss_dict=loss_dict)
                    
                    # 1. log loss values
                    if view not in loss_view_dict:
                        loss_view_dict[view] = view_loss.item()
                    else:
                        loss_view_dict[view] += view_loss.item()
                    
                    # 2. detached loss for weighting
                    tracking_loss_vals.append(view_loss.detach())
                    
                    # 3. accumulate raw frame losses
                    frame_losses.append(view_loss)
                    
                    # 4. detailed frame loss log
                    # store scalars only, not tensors
                    for log_k, log_v in log_dict.items():
                        if log_k not in frame_log_dict:
                            frame_log_dict[log_k] = []
                        # tensor -> scalar if needed
                        val = log_v[0].item() if isinstance(log_v[0], torch.Tensor) else log_v[0]
                        frame_log_dict[log_k].append(val)
                    
                    # 5. free unused vars
                    del res
                    
                    # after all views: batch MemoryBank update
                    # MemoryBank: temporal update only
                    if memory_bank is not None and config["REID_LOSS"] and enable_reid_training and len(frame_all_view_tracks) > 0:
                        # shared timestamp (first view counter)
                        t = cur_frame_view[0] + frame_idx
                        memory_bank.push_from_views(
                            frame_all_view_tracks,
                            t
                        )
                        
                        # training: update query from ReID via GT ID
                        # after temporal update, enhance query with ReID
                        if len(frame_all_reid_features) > 0:
                            from models.query_update_from_reid import update_query_with_reid_features
                            
                            for view in frame_all_view_tracks.keys():
                                tracks = frame_all_view_tracks[view]
                                # ReID features for this view
                                view_reid_features = {}
                                for (v, track_id), reid_feat in frame_all_reid_features.items():
                                    if v == view:
                                        view_reid_features[track_id] = reid_feat
                                
                                if len(view_reid_features) > 0:
                                    # update query in-place by GT ID
                                    update_query_with_reid_features(
                                        tracks=tracks,
                                        reid_features=view_reid_features,
                                        reid_update_weight=config.get("REID_UPDATE_WEIGHT", 0.1),
                                        use_dab=config["USE_DAB"]
                                    )
                    
                    # total frame loss (tracking + ReID)
                    if len(frame_losses) > 0:
                        # raw tracking loss (unweighted)
                        frame_tracking_loss = sum(frame_losses)
                        
                        # ReID loss: in-frame + cross-frame
                        frame_reid_loss = None
                        reid_stats = None
                        # compute ReID loss only when ReID training enabled
                        if config["REID_LOSS"] and enable_reid_training:
                            # loss from current feats + ReIDPool history
                            frame_reid_loss, reid_stats = calculate_reid_loss_with_current_and_memory(
                                current_features=current_frame_reid_features,  # current frame (with grad)
                                reID_pool=reID_pool,  # history pool (detached)
                                reid_model=reid_model,
                                triplet_loss_fn=triplet_loss_fn,
                                config=config,
                                device=device
                            )
                        
                        # reversibility loss
                        frame_reversibility_loss = None
                        if not config.get("USE_SIMPLE_REID", False) and len(frame_reversibility_losses) > 0:
                            # sum reversibility losses from ReversibleReID
                            total_rev_loss1 = sum([loss[0] for loss in frame_reversibility_losses])
                            total_rev_loss2 = sum([loss[1] for loss in frame_reversibility_losses])
                            
                            # apply learnable weights if enabled
                            if reversibility_weight_learner is not None:
                                frame_reversibility_loss, rev_loss_dict = reversibility_weight_learner(
                                    total_rev_loss1, total_rev_loss2
                                )
                                # log details
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss1_raw", value=total_rev_loss1.item())
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss2_raw", value=total_rev_loss2.item())
                                metric_log.update(name=f"frame{frame_idx}_reversibility_weight1", value=rev_loss_dict['weight1'])
                                metric_log.update(name=f"frame{frame_idx}_reversibility_weight2", value=rev_loss_dict['weight2'])
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss_weighted", value=frame_reversibility_loss.item())
                            else:
                                # sum without learnable weights
                                frame_reversibility_loss = total_rev_loss1 + total_rev_loss2
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss1", value=total_rev_loss1.item())
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss2", value=total_rev_loss2.item())
                                metric_log.update(name=f"frame{frame_idx}_reversibility_loss_total", value=frame_reversibility_loss.item())
                        
                        # weighted total frame loss
                        if frame_reid_loss is not None or frame_reversibility_loss is not None:
                            # uncertainty or manual weighting
                            if uncertainty_loss_fn is not None:
                                # uncertainty_loss_fn weights tracking + ReID
                                frame_total_loss, loss_dict = uncertainty_loss_fn(
                                    frame_tracking_loss, frame_reid_loss if frame_reid_loss is not None else torch.tensor(0.0, device=device)
                                )
                                # add reversibility loss separately
                                if frame_reversibility_loss is not None:
                                    frame_total_loss = frame_total_loss + frame_reversibility_loss
                                
                                # log weights
                                if (i + 1) % 50 == 0 and frame_idx == 0:
                                    weights = uncertainty_loss_fn.get_weights()
                                    logger.show(f"[Iter {i+1}] Uncertainty weights: ω1={weights['omega1']:.3f}, ω2={weights['omega2']:.3f}, "
                                               f"w1={weights['weight1']:.3f}, w2={weights['weight2']:.3f}")
                                    
                                    # log reversibility weights if enabled
                                    if reversibility_weight_learner is not None:
                                        rev_weights = reversibility_weight_learner.get_weights()
                                        logger.show(f"[Iter {i+1}] Reversibility weights: w1={rev_weights['weight1']:.4f}, w2={rev_weights['weight2']:.4f}")
                            else:
                                # manual weighting
                                frame_total_loss = w_track_val * frame_tracking_loss
                                if frame_reid_loss is not None:
                                    frame_total_loss = frame_total_loss + w_reid_val * frame_reid_loss
                                if frame_reversibility_loss is not None:
                                    frame_total_loss = frame_total_loss + frame_reversibility_loss
                        else:
                            # edge case: ReID loss unavailable
                            if uncertainty_loss_fn is not None:
                                # uncertainty weighting with reid_loss = 0
                                frame_total_loss, loss_dict = uncertainty_loss_fn(
                                    frame_tracking_loss, torch.tensor(0.0, device=device)
                                )
                            else:
                                frame_total_loss = w_track_val * frame_tracking_loss
                        
                        # accumulate clip loss; defer backward
                        if clip_total_loss is None:
                            clip_total_loss = frame_total_loss
                        else:
                            clip_total_loss = clip_total_loss + frame_total_loss
                        
                        # logging
                        tracking_loss_vals.append(frame_tracking_loss.detach())
                        if frame_reid_loss is not None:
                            reid_loss_vals.append(frame_reid_loss.detach())
                        
                        # per-frame metrics
                        # 1. detailed tracking losses
                        for log_k, log_vals in frame_log_dict.items():
                            avg_val = sum(log_vals) / len(log_vals) if log_vals else 0.0
                            metric_log.update(name=log_k, value=avg_val)
                        
                        # 2. total tracking loss per frame
                        metric_log.update(name=f"frame{frame_idx}_tracking_loss", value=frame_tracking_loss.item())
                        
                        # 3. ReID loss per frame when enabled
                        total_loss_value = frame_tracking_loss.item() * w_track_val
                        
                        if enable_reid_training and frame_reid_loss is not None:
                            metric_log.update(name=f"frame{frame_idx}_reid_loss", value=frame_reid_loss.item())
                            total_loss_value += frame_reid_loss.item() * w_reid_val
                            
                            # detailed ReID loss stats
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
                                    
                                    # cross-view positive ratio
                                    total_pairs = reid_stats['num_cross_view_pairs'] + reid_stats['num_noise_pairs']
                                    if total_pairs > 0:
                                        cross_view_ratio = reid_stats['num_cross_view_pairs'] / total_pairs
                                        metric_log.update(name=f"frame{frame_idx}_reid_cross_view_ratio", value=cross_view_ratio)
                        
                        # log reversibility loss if any
                        if frame_reversibility_loss is not None:
                            total_loss_value += frame_reversibility_loss.item()
                        
                        # log total loss
                        metric_log.update(name=f"frame{frame_idx}_total_loss", value=total_loss_value)
                        
                        # free frame vars; keep frame_total_loss
                        del frame_losses, frame_tracking_loss
                        if frame_reid_loss is not None:
                            del frame_reid_loss
                
                # backward once per clip
                # grads accumulate; step every accumulation_steps
                if clip_total_loss is not None:
                    # validate loss
                    if torch.isnan(clip_total_loss) or torch.isinf(clip_total_loss):
                        logger.show(f"[ERROR] Batch {i}, Clip {clip_idx}: Loss is NaN or Inf! "
                                   f"Loss value: {clip_total_loss.item()}")
                        logger.show(f"[ERROR] Skipping backward for this clip...")
                        del clip_total_loss
                        continue
                    
                    clip_total_loss.backward()
                    del clip_total_loss
                
        # total loss for logging (matches backward)
        loss = None
        if loss is None:
            # backward done in frame loop; logging only
            if tracking_loss_vals:
                total_tracking_loss_val = sum(tracking_loss_vals)  # tensor sum
            else:
                total_tracking_loss_val = torch.tensor(0.0, device=device)
            
            # ReID loss stats only when ReID training on
            if enable_reid_training and reid_loss_vals:
                total_reid_loss_val = sum(reid_loss_vals)  # tensor sum
            else:
                total_reid_loss_val = torch.tensor(0.0, device=device)
            
            # total loss matching backward composition
            if enable_reid_training and total_reid_loss_val.item() > 0:
                # ReID on: tracking + ReID
                loss = total_tracking_loss_val * w_track_val + total_reid_loss_val * w_reid_val
                # log component losses
                metric_log.update(name="batch_tracking_loss", value=total_tracking_loss_val.item())
                metric_log.update(name="batch_reid_loss", value=total_reid_loss_val.item())
            else:
                # ReID off: tracking only
                loss = total_tracking_loss_val * w_track_val
                # tracking loss only
                metric_log.update(name="batch_tracking_loss", value=total_tracking_loss_val.item())
    
        # metrics log - total loss
        metric_log.update(name="total_loss", value=loss.item())
        # loss.backward() done in per-frame/clip backward

        if (i + 1) % accumulation_steps == 0:
            if max_norm > 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if enable_reid_training and reid_model is not None:
                    reid_grad_clip = config.get("REID_GRAD_CLIP", 1.0)  # default 1.0
                    reid_grad_norm = torch.nn.utils.clip_grad_norm_(reid_model.parameters(), reid_grad_clip)


                if uncertainty_loss_fn is not None:
                    torch.nn.utils.clip_grad_norm_(uncertainty_loss_fn.parameters(), max_norm)
                
                if reversibility_weight_learner is not None:
                    torch.nn.utils.clip_grad_norm_(reversibility_weight_learner.parameters(), max_norm)
            else:
                pass
            
            # debug: param values before/after first batch
            if i == 0 and enable_reid_training and reid_model is not None:
                # snapshot first param slice before step
                first_param = next(reid_model.parameters())
                param_before = first_param.data[0][:5].clone() if first_param.numel() >= 5 else first_param.data.clone()
                
                # debug: check gradient exists
                grad_exists = first_param.grad is not None
                grad_norm = first_param.grad.norm().item() if grad_exists else 0.0
                logger.show(f"[DEBUG] Before update - grad exists: {grad_exists}, grad norm: {grad_norm:.6f}")
            
            optimizer.step()
            
            # debug: verify param update after first batch
            if i == 0 and enable_reid_training and reid_model is not None:
                first_param = next(reid_model.parameters())
                param_after = first_param.data[0][:5] if first_param.numel() >= 5 else first_param.data
                param_diff = (param_after - param_before).abs().max().item()
                
                # current ReID LR
                reid_lr = None
                if lr_names is not None:
                    for idx, name in enumerate(lr_names):
                        if name == "lr_reid":
                            reid_lr = optimizer.param_groups[idx]["lr"]
                            break
                
                if param_diff > 1e-8:
                    logger.show(f"[DEBUG] ReID params updated! Max change: {param_diff:.8f}, current LR: {reid_lr}")
                else:
                    logger.show(f"[WARNING] ReID params NOT updated! Max change: {param_diff:.8f}, current LR: {reid_lr}")
                    logger.show(f"[WARNING] Possible causes: 1) zero grad  2) LR too small  3) grad clip too strict")
            
            # print optimizer grad norms
            # print(f"Optimizer gradients: {[param.grad for name, param in model.named_parameters()]}") # print per-param gradients
            optimizer.zero_grad()
            # p.step()
        # log_dict already filled in frame loop
        iter_end_timestamp = time.time()
        metric_log.update(name="time per iter", value=iter_end_timestamp-iter_start_timestamp)
        
        # Outputs logs
        if i % 100 == 0:
            metric_log.sync()
            max_memory = max([torch.cuda.max_memory_allocated(torch.device('cuda', gpu_id))
                            for gpu_id in range(distributed_world_size())]) // (1024**2)
            second_per_iter = metric_log.metrics["time per iter"].avg
            
            # training mode tag
            mode_tag = "Track+ReID" if enable_reid_training else "Track-Only"
            
            logger.show(head=f"[Epoch={epoch}, Iter={i}, Mode={mode_tag}, "
                            f"{second_per_iter:.2f}s/iter, "
                            f"{i}/{dataloader_len} iters, "
                            f"rest time: {int(second_per_iter * (dataloader_len - i) // 60)} min, "
                            f"Max Memory={max_memory}MB]",
                        log=metric_log)
            
            # multi-view ReID stats
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
    
    # epoch end: ReID loss summary when enabled
    if enable_reid_training:
        logger.show("="*80)
        logger.show(f"[Epoch {epoch}] ReID training summary:")
        
        # 1. mean loss values
        if 'frame0_reid_loss' in metric_log.metrics:
            reid_loss_avg = metric_log.metrics['frame0_reid_loss'].avg
            logger.show(f"  Total ReID loss: {reid_loss_avg:.4f}")
        
        if 'frame0_id_loss' in metric_log.metrics:
            id_loss_avg = metric_log.metrics['frame0_id_loss'].avg
            logger.show(f"  ID classification loss: {id_loss_avg:.4f}")
        
        if 'frame0_triplet_loss' in metric_log.metrics:
            triplet_loss_avg = metric_log.metrics['frame0_triplet_loss'].avg
            logger.show(f"  Triplet Loss: {triplet_loss_avg:.4f}")
        
        if 'frame0_cross_clip_loss' in metric_log.metrics:
            cross_clip_loss_avg = metric_log.metrics['frame0_cross_clip_loss'].avg
            logger.show(f"  Cross-Clip Loss: {cross_clip_loss_avg:.4f}")
        
        # 2. sample stats
        if 'frame0_reid_total_samples' in metric_log.metrics:
            samples_avg = metric_log.metrics['frame0_reid_total_samples'].avg
            unique_ids_avg = metric_log.metrics['frame0_reid_unique_ids'].avg
            logger.show(f"  Avg samples: {samples_avg:.1f}, avg unique IDs: {unique_ids_avg:.1f}")
        
        if 'frame0_reid_multi_view_ids' in metric_log.metrics:
            multi_view_avg = metric_log.metrics['frame0_reid_multi_view_ids'].avg
            cross_view_ratio_avg = metric_log.metrics['frame0_reid_cross_view_ratio'].avg
            logger.show(f"  Multi-view IDs: {multi_view_avg:.1f}, cross-view positive ratio: {cross_view_ratio_avg:.2%}")
        
        logger.show("="*80)

    return


def get_param_groups(config: dict, model: nn.Module) -> Tuple[List[Dict], List[str]]:
    """
    Build param groups with different LR settings per module.
    Args:
        config: experiment config
        model: model to train

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
    points_keywords = ["reference_points", "sampling_offsets"]  # transformer ref-point / sampling-offset params
    query_updater_keywords = ["query_updater"]
    param_groups = [
        {   # backbone LR
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
    # ReID-specific LR if configured
    lr_reid = config.get("LR_REID", config["LR"])  # default: main model LR
    lr_classifier = config.get("LR_CLASSIFIER", lr_reid)
    lr_bottleneck = config.get("LR_BOTTLENECK", lr_reid)

    # exact param groups (no accidental overlap)
    classifier_params = list(model.classifier.parameters())
    bottleneck_params = list(model.bottleneck.parameters())

    # other = all params minus classifier/bottleneck
    used = set(id(p) for p in classifier_params + bottleneck_params)
    other_params = [p for p in model.parameters() if id(p) not in used]

    param_groups = [
        {"params": other_params, "lr": lr_reid},  # use lr_reid
        {"params": bottleneck_params, "lr": lr_bottleneck},
        {"params": classifier_params, "lr": lr_classifier},
    ]
    lr_names = ["lr_reid", "lr_bottleneck", "lr_classifier"]  # distinct LR names
    return param_groups, lr_names
