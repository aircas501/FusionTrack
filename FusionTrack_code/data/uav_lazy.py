
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
import os
from PIL import Image, ImageDraw
import copy
from . import uav_transforms as T
from .mot import MOTDataset
from collections import defaultdict
import math
import random
import time


class MUAVMotLazy(MOTDataset):

    def __init__(self, config: dict, split: str, transform):
        super(MUAVMotLazy, self).__init__(config=config, split=split, transform=transform)
        
        self.config = config
        self.transform = transform
        
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        self.sample_lengths = config["SAMPLE_LENGTHS"]
        self.sample_modes = config["SAMPLE_MODES"]
        self.sample_intervals = config["SAMPLE_INTERVALS"]
        self.vis = config["VISUALIZE"]
        self.sampler_steps: list = config["SAMPLE_STEPS"] 
        
        self.num_clips_per_sample = config.get("NUM_CLIPS_PER_SAMPLE", 1)
        self.clip_interval = config.get("CLIP_INTERVAL", "consecutive")
        self.clip_interval_frames = config.get("CLIP_INTERVAL_FRAMES", 0)
        self.clip_interval_min = config.get("CLIP_INTERVAL_MIN", 1)
        self.clip_interval_max = config.get("CLIP_INTERVAL_MAX", 30)
        
        self.video_dict = {}
        self.sample_vid_tmax = None
        
        self.viewpoints_num = config["VIEW_POINT"]
        self.viewpoints = ["c00"+str(i+1) for i in range(self.viewpoints_num)]
        
        self.uav_seqs_dir = os.path.join(config["DATA_ROOT"], config["DATASET"], "images", split)
        self.uav_gts_dir = os.path.join(config["DATA_ROOT"], config["DATASET"], "labels", split)
        
        # Get all sequence names and sort them
        self.uav_seq_names = [seq for seq in os.listdir(self.uav_seqs_dir) 
                              if os.path.isdir(os.path.join(self.uav_seqs_dir, seq))]
        self.uav_seq_names.sort()
        
        self.debug = config.get("DEBUG_DATASET", False)
        
        # 1. Scan file paths
        self.gt_file_paths = defaultdict(lambda: defaultdict(dict))
        
        if self.debug:
            import time
            print(f"[DEBUG {time.strftime('%H:%M:%S')}] Scanning file paths...")
            
        for scene_id in self.uav_seq_names:
            uav_gts_scene_dir = os.path.join(self.uav_gts_dir, scene_id)
            if not os.path.exists(uav_gts_scene_dir): continue
            
            for view in self.viewpoints:
                uav_gts_dir = os.path.join(uav_gts_scene_dir, view)
                if not os.path.exists(uav_gts_dir): continue
                
                with os.scandir(uav_gts_dir) as entries:
                    for entry in entries:
                        if entry.name.endswith('.txt'):
                            frame_t = int(entry.name.split('.')[0])
                            self.gt_file_paths[scene_id][view][frame_t] = entry.path

        # Core optimization: caching mechanism
        import json
        try:
            from tqdm import tqdm
        except ImportError:
            # Provide a no-op placeholder if tqdm is unavailable
            def tqdm(iterable, desc=""): return iterable

        cache_file = os.path.join(self.uav_gts_dir, f"uav_global_id_offsets_{split}.json")
        
        self.scene_offset_dict = {}
        self.total_num_classes = 0
        
        cache_loaded = False
        
        # Try loading from cache
        if os.path.exists(cache_file):
            try:
                print(f"[INFO] Found ID mapping cache, loading: {cache_file}")
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.scene_offset_dict = cache_data['offsets']
                    self.total_num_classes = cache_data['total_classes']
                print(f"[INFO] Cache loaded successfully. Total IDs: {self.total_num_classes}")
                cache_loaded = True
            except Exception as e:
                print(f"[WARN] Failed to load cache, will rescan: {e}")
        
        # Perform full scan if cache is missing
        if not cache_loaded:
            print(f"[INFO] No cache found, scanning all annotation files to compute IDs (first run is slower)...")
            
            pbar = tqdm(self.uav_seq_names, desc="Calculating Global IDs")
            
            for scene_id in pbar:
                current_offset = self.total_num_classes
                self.scene_offset_dict[scene_id] = current_offset
                
                max_id_in_scene = -1
                
                # Scan max ID in this scene
                if scene_id in self.gt_file_paths:
                    for view in self.gt_file_paths[scene_id]:
                        for gt_path in self.gt_file_paths[scene_id][view].values():
                            try:
                                with open(gt_path, 'r') as f:
                                    for line in f:
                                        line = line.strip()
                                        if not line: continue
                                        parts = line.split()
                                        tid = -1
                                        if len(parts) == 6: tid = int(parts[1])
                                        elif len(parts) == 5: tid = int(parts[0])
                                        
                                        if tid > max_id_in_scene:
                                            max_id_in_scene = tid
                            except: continue
                
                if max_id_in_scene >= 0:
                    self.total_num_classes += (max_id_in_scene + 1)
                    if hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({"TotalIDs": self.total_num_classes})
            
            # Write cache after scan completes
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'offsets': self.scene_offset_dict,
                        'total_classes': self.total_num_classes
                    }, f, indent=4)
                print(f"[INFO] ID mapping computed and saved to cache: {cache_file}")
            except Exception as e:
                print(f"[WARN] Failed to write cache file: {e}")

        print(f"✅ Final Global Classes (NUM_CLASSES): {self.total_num_classes}")
        
        # Cache configuration
        self._gt_cache = {}
        self._cache_max_size = config.get("GT_CACHE_SIZE", 2000) 
        
        self.set_epoch(epoch=0)
    
    def _read_gt_file(self, gt_path: str) -> list:
        """
        Lazily read a single annotation file (with LRU cache).
        Returns: list of raw rows [[class_id, track_id, x_norm, y_norm, w_norm, h_norm], ...]
        Note: returns Local IDs; offset is applied in get_single_frame.
        """
        # Cache hit
        if gt_path in self._gt_cache:
            return self._gt_cache[gt_path]
        
        # Read file
        gts = []
        try:
            with open(gt_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) == 6:
                            # YOLO MOT format (6 columns): class_id track_id x y w h
                            cls, i, x, y, w, h = parts
                            cls = int(cls)
                        elif len(parts) == 5:
                            # Legacy format (5 columns): track_id x y w h
                            i, x, y, w, h = parts
                            cls = 0
                        else:
                            continue
                        
                        gts.append([
                            cls, int(i), float(x), float(y), float(w), float(h)
                        ])
                    except ValueError:
                        continue
        except Exception as e:
            if self.debug:
                print(f"[ERROR] Failed to read annotation file: {gt_path}, {e}")
            return []
        
        # Cache management (simplified LRU: clear when full)
        if len(self._gt_cache) >= self._cache_max_size:
            # if self.debug: print(f"[DEBUG] Cache full ({self._cache_max_size}), clearing cache")
            self._gt_cache.clear()
        
        self._gt_cache[gt_path] = gts
        return gts
    
    def get_single_frame(self, frame_path: str):
        """Read annotations on demand and apply Global ID offset."""
        if "UAV_V" not in frame_path:
            raise RuntimeError(f"Frame path '{frame_path}' is not from UAV_V dataset")
        
        frame_idx = int(frame_path.split("/")[-1].split(".")[0])
        view = frame_path.split("/")[-2]
        scene_id = frame_path.split("/")[-3]
        
        # Check whether file path exists
        if scene_id not in self.gt_file_paths or view not in self.gt_file_paths[scene_id]:
            raise KeyError(f"Scene '{scene_id}' or view '{view}' not found")
        
        # Get ID offset for the current scene
        scene_offset = self.scene_offset_dict.get(scene_id, 0)
        
        if frame_idx not in self.gt_file_paths[scene_id][view]:
            # Empty frame (no targets)
            gt = []
        else:
            # Lazy load: read annotation file here
            gt_path = self.gt_file_paths[scene_id][view][frame_idx]
            gt = self._read_gt_file(gt_path)
        
        # Open image
        img = Image.open(frame_path)
        size_w, size_h = img.size
        
        info = {
            "boxes": [],
            "obj_ids": [],
            "labels": [],
            "area": [],
            "dataset": "UAV_V"
        }
        
        # Convert normalized coords to absolute coords and apply ID offset
        for cls, i, x_norm, y_norm, w_norm, h_norm in gt:
            # Apply global ID offset
            track_id = i
            if track_id >= 0:
                track_id += scene_offset
            
            w_abs = round(w_norm * size_w)
            h_abs = round(h_norm * size_h)
            x_abs = round(x_norm * size_w - w_abs / 2)
            y_abs = round(y_norm * size_h - h_abs / 2)
            
            info["boxes"].append([float(x_abs), float(y_abs), float(w_abs), float(h_abs)])
            info["area"].append(w_abs * h_abs)
            info["obj_ids"].append(track_id) # Store Global ID
            info["labels"].append(cls)
        
        info["boxes"] = torch.as_tensor(info["boxes"])
        info["area"] = torch.as_tensor(info["area"])
        info["obj_ids"] = torch.as_tensor(info["obj_ids"], dtype=torch.long)
        info["labels"] = torch.as_tensor(info["labels"], dtype=torch.long)
        
        # xywh to xyxy
        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]
        else:
            info["boxes"] = torch.zeros((0, 4))
            info["obj_ids"] = torch.zeros((0,), dtype=torch.long)
            info["labels"] = torch.zeros((0,), dtype=torch.long)
        
        return img, info
    
    def set_epoch(self, epoch: int):
        """Set epoch and generate sampling paths."""
        if self.debug:
            import time
            print(f"[DEBUG {time.strftime('%H:%M:%S')}] set_epoch({epoch}) started")
        
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            assert len(self.sample_lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            return
        
        self.sample_begin_frame_paths = defaultdict(list)
        self.sample_vid_tmax = defaultdict(lambda: defaultdict(int))
        self.sample_stage = 0
        
        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.sample_stage = i + 1
        assert self.sample_stage < len(self.sampler_steps) + 1
        
        self.num_frames_per_batch = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]
        
        for scene_id in self.gt_file_paths.keys():
            for view in self.viewpoints:
                if view in self.gt_file_paths[scene_id]:
                    frames = list(self.gt_file_paths[scene_id][view].keys())
                    if frames:
                        t_min = min(frames)
                        t_max = max(frames)
                        self.sample_vid_tmax[scene_id][view] = t_max
                        
                        frames_per_clip = self.num_frames_per_batch
                        total_frames_needed = frames_per_clip * self.num_clips_per_sample + \
                                            self.clip_interval_frames * (self.num_clips_per_sample - 1)
                        
                        for t in range(t_min, t_max - (total_frames_needed - 1) * self.sample_interval + 1):
                            self.sample_begin_frame_paths[view].append(
                                os.path.join(self.uav_seqs_dir, scene_id, view, str(t).zfill(8) + ".jpg")
                            )
        
        self.current_epoch = epoch
        
        if self.debug:
            import time
            print(f"[DEBUG {time.strftime('%H:%M:%S')}] set_epoch({epoch}) done, sampling paths: {sum(len(v) for v in self.sample_begin_frame_paths.values())}")
        
        return
    
    def step_epoch(self):
        self.set_epoch(self.current_epoch + 1)
    
    def _sample_frame_indices(self, scene_id: str, view: str, begin_t: int) -> list:
        if self.sample_mode == "random_interval":
            assert self.num_frames_per_batch > 1, "Sample Length is less than 2."
            remain_frames = self.sample_vid_tmax[scene_id][view] - begin_t
            max_interval = math.floor(remain_frames / (self.num_frames_per_batch - 1))
            interval = min(random.randint(1, self.sample_interval), max_interval)
            
            all_clips_frame_indices = []
            current_t = begin_t
            
            for clip_idx in range(self.num_clips_per_sample):
                frame_indices = [current_t + interval * i for i in range(self.num_frames_per_batch)]
                all_clips_frame_indices.append(frame_indices)
                
                if clip_idx < self.num_clips_per_sample - 1:
                    if self.clip_interval == "consecutive":
                        current_t = frame_indices[-1] + interval
                    elif self.clip_interval == "random":
                        current_t = frame_indices[-1] + random.randint(1, self.clip_interval_frames + 1)
                    elif self.clip_interval == "random_range":
                        clip_gap = random.randint(self.clip_interval_min, self.clip_interval_max)
                        current_t = frame_indices[-1] + clip_gap
                    elif self.clip_interval == "fixed":
                        current_t = frame_indices[-1] + self.clip_interval_frames
                    else:
                        current_t = frame_indices[-1] + interval
            
            return all_clips_frame_indices
        else:
            raise NotImplementedError(f"Do not support sample mode '{self.sample_mode}'.")
    
    def _generate_frame_paths(self, scene_id: str, view: str, sampled_frame_indices: list) -> list:
        all_clips_frame_paths = []
        for frame_indices in sampled_frame_indices:
            frame_paths = [os.path.join(self.uav_seqs_dir, scene_id, view, str(t).zfill(8) + ".jpg") 
                          for t in frame_indices]
            all_clips_frame_paths.append(frame_paths)
        return all_clips_frame_paths
    
    def get_multi_frames(self, frame_paths: list):
        return zip(*[self.get_single_frame(frame_path=path) for path in frame_paths])
    
    def get_multi_clips(self, all_clips_frame_paths: list):
        clips_data = []
        for clip_frame_paths in all_clips_frame_paths:
            imgs, infos = self.get_multi_frames(clip_frame_paths)
            clips_data.append((imgs, infos))
        return clips_data
    
    def __getitem__(self, item):
        all_clips_data = []
        
        first_view = self.viewpoints[0]
        try:
            begin_frame_path_first_view = self.sample_begin_frame_paths[first_view][item]
        except IndexError as e:
            raise IndexError(f"Sample index {item} out of range for view {first_view}.")
        
        scene_id = begin_frame_path_first_view.split("/")[-3]
        begin_t = int(begin_frame_path_first_view.split("/")[-1].split(".")[0])
        
        sampled_frame_indices = self._sample_frame_indices(scene_id, first_view, begin_t)

        for view_idx, view in enumerate(self.viewpoints): 
            all_clips_frame_paths = self._generate_frame_paths(scene_id, view, sampled_frame_indices)
            clips_data = self.get_multi_clips(all_clips_frame_paths=all_clips_frame_paths)
            
            view_clips_transformed = []
            for clip_idx, (imgs, infos) in enumerate(clips_data):
                transformed_imgs, transformed_infos = self.transform["UAV_V"](imgs, infos)
                view_clips_transformed.append({
                    'imgs': transformed_imgs,
                    'infos': transformed_infos
                })
            
            while len(all_clips_data) < len(view_clips_transformed):
                all_clips_data.append({})
            
            for clip_idx, clip_data in enumerate(view_clips_transformed):
                for info in clip_data['infos']:
                    info['view_id'] = view_idx
                    info['view_name'] = view
                all_clips_data[clip_idx][view] = clip_data
        
        return {'clips': all_clips_data}

    def __len__(self):
        assert self.sample_begin_frame_paths is not None, "Please use set_epoch to init Dataset."
        return len(list(self.sample_begin_frame_paths.values())[0])


# ==================== Transform and build functions ====================

def make_transforms_for_uav(image_set, config=None):
    coco_size = config["COCO_SIZE"]
    overflow_bbox = config["OVERFLOW_BBOX"]
    reverse_clip = config["REVERSE_CLIP"]
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600] if coco_size else [800, 1000, 1200]),
                    T.FixedMotRandomCrop(384 if coco_size else 800, 600 if coco_size else 1200),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MultiHSV(),
            normalize,
            T.MultiReverseClip(reverse=reverse_clip)
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset2transform(image_set, config=None):
    uav_train = make_transforms_for_uav('train', config)
    uav_test = make_transforms_for_uav('val', config)

    dataset2transform_train = {'UAV_V': uav_train}
    dataset2transform_val = {'UAV_V': uav_test}
    
    if image_set == 'train':
        return dataset2transform_train
    elif image_set == 'val':
        return dataset2transform_val
    else:
        raise NotImplementedError()


def build(config, split):
    """
    Build the lazy-loading UAV dataset.
    """
    dataset2transform = build_dataset2transform(split, config)
    if split == 'train':
        return MUAVMotLazy(
            config=config,
            split=split,
            transform=dataset2transform
        )
    # Other splits can be added as needed
    raise NotImplementedError(f"Split '{split}' not implemented for lazy loading yet")