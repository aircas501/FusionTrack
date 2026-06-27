import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from typing import Tuple, Any, Union, Type, Dict, List
from collections import defaultdict


def collate_fn(batch):
    """
    Legacy collate function for single-clip multi-view data (backward compatible)
    
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
    Collate function for multi-clip multi-view data
    
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
                {  # Clip 1, organized by view (similar to legacy collate_fn)
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
    # Check whether batch is empty
    if len(batch) == 0:
        return {'clips': []}
    
    # Check data structure
    if 'clips' not in batch[0]:
        # Fall back to legacy collate_fn if not a multi-clip structure
        return collate_fn(batch)
    
    num_clips = len(batch[0]['clips'])
    batch_size = len(batch)
    
    # Get all view names (from the first clip of the first sample)
    views = list(batch[0]['clips'][0].keys())
    
    # Initialize result structure: each clip organized by view
    collated_clips = []
    for clip_idx in range(num_clips):
        collated_clip = {}
        for view in views:
            collated_clip[view] = defaultdict(list)
        collated_clips.append(collated_clip)
    
    # Organize each clip from each sample
    for sample_idx, sample in enumerate(batch):
        if 'clips' not in sample:
            raise ValueError(f"Sample {sample_idx} does not have 'clips' key")
        
        for clip_idx, clip_data in enumerate(sample['clips']):
            if clip_idx >= len(collated_clips):
                # Extend list if clip counts are inconsistent
                collated_clip = {}
                for view in views:
                    collated_clip[view] = defaultdict(list)
                collated_clips.append(collated_clip)
            
            # Collate data for each view
            for view, view_data in clip_data.items():
                collated_clips[clip_idx][view]["imgs"].append(view_data["imgs"])
                collated_clips[clip_idx][view]["infos"].append(view_data["infos"])
    
    return {'clips': collated_clips}


def get_collate_fn(dataset_type: str = None):
    """
    Return the appropriate collate function for the dataset type
    
    Args:
        dataset_type: Dataset type, e.g. 'UAV_V', 'DanceTrack', etc.
    
    Returns:
        collate function
    """
    if dataset_type in ['UAV_V', 'UAV']:
        return collate_fn_multiview_multiclip
    else:
        return collate_fn