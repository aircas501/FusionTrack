import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from typing import Tuple, Any, Union, Type, Dict, List
from collections import defaultdict


def collate_fn(batch):
    """
    旧的collate函数，用于单clip多视角数据（向后兼容）
    
    Args:
        batch: List of samples, each sample is {
            'c001': {'imgs': Tensor, 'infos': List},
            'c002': {'imgs': Tensor, 'infos': List},
            ...
        }
    
    Returns:
        {
            'c001': {'imgs': List[Tensor], 'infos': List[List[Dict]]},
            'c002': {'imgs': List[Tensor], 'infos': List[List[Dict]]},
            ...
        }
    """
    collated_batch = {}
    for view in batch[0].keys():
        collated_batch[view] = defaultdict(list)
    for data in batch:
        for view, info in data.items():
            collated_batch[view]["imgs"].append(info["imgs"])
            collated_batch[view]["infos"].append(info["infos"])
    return collated_batch


def collate_fn_multiview_multiclip(batch):
    """
    处理多clip多视角数据的collate函数
    
    Args:
        batch: List of samples, each sample is {
            'clips': [
                {c001: {'imgs': Tensor, 'infos': List}, c002: {...}, ...},  # Clip 1
                {c001: {'imgs': Tensor, 'infos': List}, c002: {...}, ...},  # Clip 2
                ...
            ]
        }
    
    Returns:
        {
            'clips': [
                {  # Clip 1，按view组织（类似旧的collate_fn）
                    'c001': {'imgs': List[Tensor], 'infos': List[List[Dict]]},
                    'c002': {'imgs': List[Tensor], 'infos': List[List[Dict]]},
                    ...
                },
                {  # Clip 2
                    'c001': {'imgs': List[Tensor], 'infos': List[List[Dict]]},
                    'c002': {'imgs': List[Tensor], 'infos': List[List[Dict]]},
                    ...
                },
                ...
            ]
        }
    """
    # 检查batch是否为空
    if len(batch) == 0:
        return {'clips': []}
    
    # 检查数据结构
    if 'clips' not in batch[0]:
        # 如果不是多clip结构，使用旧的collate_fn
        return collate_fn(batch)
    
    num_clips = len(batch[0]['clips'])
    batch_size = len(batch)
    
    # 获取所有view名称（从第一个样本的第一个clip）
    views = list(batch[0]['clips'][0].keys())
    
    # 初始化结果结构：每个clip按view组织
    collated_clips = []
    for clip_idx in range(num_clips):
        collated_clip = {}
        for view in views:
            collated_clip[view] = defaultdict(list)
        collated_clips.append(collated_clip)
    
    # 对每个样本的每个clip进行组织
    for sample_idx, sample in enumerate(batch):
        if 'clips' not in sample:
            raise ValueError(f"Sample {sample_idx} does not have 'clips' key")
        
        for clip_idx, clip_data in enumerate(sample['clips']):
            if clip_idx >= len(collated_clips):
                # 如果clip数量不一致，扩展列表
                collated_clip = {}
                for view in views:
                    collated_clip[view] = defaultdict(list)
                collated_clips.append(collated_clip)
            
            # 对每个view的数据进行collate
            for view, view_data in clip_data.items():
                collated_clips[clip_idx][view]["imgs"].append(view_data["imgs"])
                collated_clips[clip_idx][view]["infos"].append(view_data["infos"])
    
    return {'clips': collated_clips}


def get_collate_fn(dataset_type: str = None):
    """
    根据数据集类型返回合适的collate函数
    
    Args:
        dataset_type: 数据集类型，如 'UAV_V', 'DanceTrack' 等
    
    Returns:
        collate函数
    """
    if dataset_type in ['UAV_V', 'UAV']:
        return collate_fn_multiview_multiclip
    else:
        return collate_fn