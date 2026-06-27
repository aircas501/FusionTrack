
import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

from tqdm import tqdm
from os import path
from typing import List
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from models import build_model
from models.utils import load_checkpoint, get_model
from models.runtime_tracker import RuntimeTracker
from utils.utils import yaml_to_dict, is_distributed, distributed_world_size, distributed_rank, inverse_sigmoid
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union
from log.logger import Logger
from data.seq_dataset import SeqDataset
from data.seq_multiview_dataset import SeqMultiViewDataset
from structures.track_instances import TrackInstances
from models.reid_model_reversible_mlp import build_reversible_reid as build_reid_model
# from models.simple_reid_model import build_simple_reid_model as build_reid_model
from models.reid_pool import build as build_reid_pool, ReIDPool
from models.memory_bank import MemoryBank
from utils.reid_pool_saver import ReIDPoolSaver
from utils.neighbor_filter import build_neighbor_filter



class Submitter:
    def __init__(self, dataset_name: str, split_dir: str, seq_name: str, outputs_dir: str, model: nn.Module,
                 det_score_thresh: float = 0.7, track_score_thresh: float = 0.6, result_score_thresh: float = 0.7,
                 miss_tolerance: int = 5,
                 use_motion: bool = False, motion_lambda: float = 0.5,
                 motion_min_length: int = 3, motion_max_length: int = 5,
                 use_dab: bool = False,
                 visualize: bool = False,
                 config = None,
                 reid_model = None,
                 reID_pool: ReIDPool = None,
                 memory_bank: MemoryBank = None):
        self.dataset_name = dataset_name
        self.seq_name = seq_name
        self.seq_dir = path.join(split_dir, seq_name)
        self.outputs_dir = outputs_dir
        self.predict_dir = path.join(self.outputs_dir, "tracker")
        self.model = model
        self.tracker = RuntimeTracker(det_score_thresh=det_score_thresh, track_score_thresh=track_score_thresh,
                                      miss_tolerance=miss_tolerance,
                                      use_motion=use_motion,
                                      motion_min_length=motion_min_length, motion_max_length=motion_max_length,
                                      visualize=visualize, use_dab=use_dab)
        self.result_score_thresh = result_score_thresh
        self.motion_lambda = motion_lambda
        self.dataset = SeqMultiViewDataset(seq_dir=self.seq_dir,config=config)
        # Optimize data loading: more workers and prefetch for better GPU utilization
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=1, 
            num_workers=12,        # More workers (multi-view needs more workers)
            shuffle=False,
            pin_memory=True,       # pin_memory speeds up GPU transfer
            prefetch_factor=4      # Each worker prefetches 4 batches to hide I/O latency
        )
        self.device = next(self.model.parameters()).device
        self.use_dab = use_dab
        self.use_motion = use_motion
        self.visualize = visualize
        self.reid_model = reid_model
        self.reID_pool = reID_pool
        self.memory_bank = memory_bank  # Store memory_bank instance
        self.config = config
        
        # Initialize ReIDPoolSaver (saves ReID features for all frames)
        self.reid_pool_saver = None
        if config.get("SAVE_REID_POOL_DATA", False):
            save_dir = config.get("REID_POOL_SAVE_DIR", "./outputs/reid_pool_data")
            self.reid_pool_saver = ReIDPoolSaver(
                save_dir=os.path.join(save_dir, seq_name),
                seq_name=seq_name
            )
            print(f"[Submitter] ✅ ReIDPool data will be saved to {save_dir}")
        
        # Initialize neighbor filter
        self.neighbor_filter = build_neighbor_filter(config)
        if self.neighbor_filter is not None:
            print(f"\n{'='*60}")
            print(f"[Submitter] ✅ Neighbor Filter Configuration:")
            print(f"  - Spatial Radius: {self.neighbor_filter.spatial_radius} px")
            print(f"  - K-Top: {self.neighbor_filter.k_top_similarities}")
            print(f"  - Min Ratio: {self.neighbor_filter.min_neighbor_match_ratio}")
            print(f"  - Threshold: {self.neighbor_filter.neighbor_sim_threshold}")
            print(f"  - Metric: {self.neighbor_filter.distance_metric}")
            print(f"{'='*60}\n")
        else:
            print(f"[Submitter] ⚠️  Neighbor filter disabled")
        
        # Prepare output paths
        os.makedirs(self.predict_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_dir, f'{self.seq_name}.txt')):
            os.remove(os.path.join(self.predict_dir, f'{self.seq_name}.txt'))
        self.model.eval()
        if self.reid_model is not None:
            self.reid_model.eval()
        return

    @torch.no_grad()
    def run(self):
        tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                 num_classes=get_model(self.model).num_classes,
                                 use_dab=self.use_dab).to(self.device)]
        bdd100k_results = []    # for bdd100k, will be converted into json file, different from other datasets.
        for i, ((image, ori_image), info) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            # image: (1, C, H, W); ori_image: (1, H, W, C)
            frame = tensor_list_to_nested_tensor([image[0]]).to(self.device)
            res = self.model(frame=frame, tracks=tracks)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(previous_tracks, new_tracks, None)


            if self.use_motion:
                for _ in range(len(tracks[0])):
                    if tracks[0].disappear_time[_].item() > 0:
                        if len(self.tracker.motions[tracks[0].ids[_].item()]) >= \
                               self.tracker.motions[tracks[0].ids[_].item()].min_record_length:
                            tracks[0].ref_pts[_] = inverse_sigmoid(
                                tracks[0].last_appear_boxes[_]
                            ) + self.motion_lambda * self.tracker.motions[tracks[0].ids[_].item()].get_box_delta(
                                miss_length=tracks[0].disappear_time[_].item()
                            ).to(tracks[0].last_appear_boxes.device)

            tracks_result = tracks[0].to(torch.device("cpu"))
            ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
            # box = [x, y, w, h]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            
            # Optional NMS dedup (usually unnecessary for DETR-like models; bipartite matching avoids duplicates)
            if self.config is not None and self.config.get("USE_NMS", False):  # Off by default
                iou_threshold = self.config.get("NMS_IOU_THRESHOLD", 0.7)
                tracks_result = self.nms_tracks(tracks_result, iou_threshold=iou_threshold)
            
            # to xyxy:
            tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
            tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float))
            if self.dataset_name == "BDD100K":
                self.update_results(tracks_result=tracks_result, frame_idx=i, results=bdd100k_results, img_path=info[0])
            else:
                self.write_results(tracks_result=tracks_result, frame_idx=i)

            if self.visualize:
                os.makedirs(f"./outputs/visualize_tmp/frame_{i+1}/", exist_ok=True)
                os.system(f"mv ./outputs/visualize_tmp/query_updater/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/decoder/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/memotr/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/runtime_tracker/ ./outputs/visualize_tmp/frame_{i+1}/")

        if self.visualize:
            visualize_save_dir = os.path.join("./outputs/visualize/", self.seq_name)
            os.makedirs(visualize_save_dir, exist_ok=True)
            os.system(f"mv ./outputs/visualize_tmp/* {visualize_save_dir}")

        if self.dataset_name == "BDD100K":
            with open(os.path.join(self.predict_dir, '{}.json'.format(self.seq_name)), 'w', encoding='utf-8') as f:
                json.dump(bdd100k_results, f)

        return
    
    def extract_reid_features(self, res, tracks, frame_idx, view_idx):

        reid_features_dict = {}
        
        # Iterate tracks (already postprocessed)
        for t_idx, track in enumerate(tracks):
            # Ensure track has output_embed (set by MeMOTR tracker update)
            if not hasattr(track, 'output_embed'):
                print("[WARNING] Track has no output_embed, skipping ReID extraction.")
                continue
                
            
            # Ensure tensors are on the correct device
            track_ids = track.ids.to(self.device)
            output_embeds = track.output_embed.to(self.device)

            for i, track_id in enumerate(track_ids):
                if track_id.item() >= 0:  # Valid ID
                    query_feat = output_embeds[i] 
                    
                    # Prepare ReID model input
                    query_embed_list = [query_feat.unsqueeze(0)]  # [(1, C)]
                    frame_id_list_list = [[frame_idx]]
                    
                    # Run through ReID model
                    use_simple_reid = self.config.get("USE_SIMPLE_REID", False)
                    if use_simple_reid:
                        reid_feat = get_model(self.reid_model)(query_embed_list, None)
                    else:
                        reid_feat = get_model(self.reid_model)(query_embed_list, None)
                    
                    # Handle return value
                    if isinstance(reid_feat, tuple):
                        reid_feat = reid_feat[1]
                    if reid_feat.dim() > 1:
                        reid_feat = reid_feat[0]
                    
                    # L2 normalize: match training (TripletLoss uses normalized features)
                    reid_feat = torch.nn.functional.normalize(reid_feat, p=2, dim=-1)
                    
                    reid_features_dict[track_id.item()] = reid_feat
        
        return reid_features_dict

    @torch.no_grad()
    def run_multi_view(self):

        import time


        bdd100k_results = []    # for bdd100k, will be converted into json file, different from other datasets.
        view_dict_name = self.dataloader.dataset.viewpoints

        # Initialize tracks per view
        tracks_dict = {view: [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                        num_classes=get_model(self.model).num_classes,
                        use_dab=self.use_dab).to(self.device)] for view in view_dict_name}
        
        for i, dict_data in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):  # One video sequence per iteration
            view_list = list(dict_data.keys())  # All views
            tracks_result_dict = {}  # Per-view results for the current frame
            
            # ========================================
            # Step 0: Save image info
            # ========================================
            #print("nms 238" + str(time.perf_counter()))
            is_first_frame = (i == 0)  # Mark whether this is the first frame
            per_view_image_info = {}  # Per-view image info {view: (image, ori_image, info)}
            
            for view in view_list:
                (image, ori_image) = dict_data[view]["imgs"]
                info = dict_data[view]["infos"]
                per_view_image_info[view] = (image, ori_image, info)
            
            # ========================================
            # Step 1: Model forward (no pre-update; MemoryBank updates on push)
            # ========================================
            #print("nms 251" + str(time.perf_counter()))
            per_view_res = {}  # Per-view model outputs {view: res}
            
            for view in view_list:
                (image, ori_image, info) = per_view_image_info[view]
                frame = tensor_list_to_nested_tensor([image[0]]).to(self.device)
                tracks = tracks_dict[view]  # Tracks (optionally refined by MemoryBank)

                # Single forward pass
                res = self.model(frame=frame, tracks=tracks)
                per_view_res[view] = res
                
                # Runtime tracker update
                previous_tracks, new_tracks = self.tracker.update(
                    model_outputs=res,
                    tracks=tracks
                )
                
                # Postprocess
                tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(
                    previous_tracks, new_tracks, None
                )
                
                # Update tracks_dict
                tracks_dict[view] = tracks
            
            # ========================================
            # Step 3: Extract ReID features
            # ========================================
            #print("nms 280" + str(time.perf_counter()))
            per_view_reid_features = {}  # {view: {track_id: reid_feat}}
            
            for view in view_list:
                tracks = tracks_dict[view]
                res = per_view_res[view]
                
                # Extract ReID features
                reid_feats_dict = {}
                if self.reid_model is not None and self.config.get("REID_LOSS", False):
                    view_idx = int(view.replace("c00", "")) - 1 
                    reid_feats_dict = self.extract_reid_features(res, tracks, i, view_idx)
                
                    if len(reid_feats_dict) > 0:
                        reid_feats_dict_on_device = {}
                        for track_id, reid_feat in reid_feats_dict.items():
                            reid_feats_dict_on_device[track_id] = reid_feat.to(self.device)
                        per_view_reid_features[view] = reid_feats_dict_on_device
            
            # ========================================
            # Step 4: Update ReIDPool (per view, using raw IDs)
            # ========================================
            #print("nms 303" + str(time.perf_counter()))
            if self.config.get("REID_LOSS", False):
                for view in view_list:
                    tracks = tracks_dict[view]
                    reid_feats_dict = per_view_reid_features.get(view, {})
                    
                    # Update ReIDPool (store ReID features)
                    # On frame 0 reid_feats_dict may be empty, but tracks still need updating
                    self.reID_pool.update_pool(view, tracks, i, reid_features=reid_feats_dict if len(reid_feats_dict) > 0 else None)
            
            # ========================================
            # Step 5: Update MemoryBank (all views together, raw IDs)
            # After all views, push all tracks and ReID features at once
            # So cross-view updates can see features from all views
            # ========================================
            #print("nms 318" + str(time.perf_counter()))
            if self.memory_bank is not None and self.config.get("REID_LOSS", False):
                # Collect ReID features from all views: {(view, id): reid_feat}
                all_reid_features_for_bank = {}
                for view in view_list:
                    reid_feats_dict = per_view_reid_features.get(view, {})
                    for track_id, reid_feat in reid_feats_dict.items():
                        # Ensure ReID features are on the correct device
                        reid_feat_on_device = reid_feat.to(self.device) if hasattr(reid_feat, 'to') else reid_feat
                        all_reid_features_for_bank[(view, track_id)] = reid_feat_on_device
                
                # Push all views at once (frame 0 stored as basis for later frames)
                # MemoryBank only does temporal (cross-frame) update
                # No cross-view update at inference here (need association first to know same ID)
                self.memory_bank.push_from_views(
                    tracks_dict,  # Tracks from all views
                    t=i
                )

            # ========================================
            # Step 6: Add ID offset for cross-view association
            # ========================================
            #print("nms 340" + str(time.perf_counter()))
            # Build per-view results first
            for view in view_list:
                (image, ori_image) = dict_data[view]["imgs"]
                tracks = tracks_dict[view]
                
                tracks_result = tracks[0].to(torch.device("cpu"))
                ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
                # box = [x, y, w, h]
                tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                    tracks_result.boxes[:, 3] * ori_h
                tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
                tracks_result = self.filter_by_area(tracks_result)
                import time
                # Optional NMS dedup (usually unnecessary for DETR-like models)
                #print("nms start" + str(time.perf_counter()))
                if self.config.get("USE_NMS", False):  # Off by default
                    iou_threshold = self.config.get("NMS_IOU_THRESHOLD", 0.7)
                    tracks_result = self.nms_tracks(tracks_result, iou_threshold=iou_threshold)
                #print("nms end" + str(time.perf_counter()))

                # to xyxy:
                tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
                tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float))
                
                tracks_result_dict[view] = tracks_result

            # Deep copy for cross-view association
                import copy
                tracks_result_dict_copy = copy.deepcopy(tracks_result_dict)
            
            # ========================================
            # Visualization
            # ========================================
            target_frame = self.config.get("VIS_TARGET_FRAME", 520)  # From config, default 520
            if target_frame >= 0 and i == target_frame and self.config.get("REID_LOSS", False):
                print(f"\n{'='*80}")
                print(f"[Visualizer] 📊 Generating visualizations at frame {i}...")
                print(f"{'='*80}\n")
                
                # 1. Collect ReID features from all views for the current frame from ReIDPool
                all_features = []  # List of numpy arrays
                all_ids = []       # List of IDs
                all_views = []     # List of view names
                features_by_view = {}  # {view: {id: feature}} for heatmap
                
                for view in view_list:
                    # Get all features for the current view from ReIDPool
                    # ReIDPool layout: self.view_id_reid_feat_dict_list[view][id] = reid_feat
                    if view not in self.reID_pool.view_id_reid_feat_dict_list:
                        continue
                    
                    view_pool = self.reID_pool.view_id_reid_feat_dict_list[view]
                    features_by_view[view] = {}
                    
                    # Iterate all IDs in the current view pool
                    for track_id, reid_feat in view_pool.items():
                        if reid_feat is not None:
                            # reid_feat may be (1, D) or (D,) tensor
                            if isinstance(reid_feat, torch.Tensor):
                                feat = reid_feat.squeeze().cpu().numpy()  # Ensure 1D
                            else:
                                feat = np.array(reid_feat).flatten()
                            
                            # Collect for t-SNE
                            all_features.append(feat)
                            all_ids.append(track_id)
                            all_views.append(view)
                            
                            # Collect for heatmap
                            features_by_view[view][track_id] = feat
                
                print(f"[Visualizer] Collected {len(all_features)} features from {len(view_list)} views")
                if len(all_views) > 0:
                    unique_views, counts = np.unique(all_views, return_counts=True)
                    print(f"[Visualizer] Feature distribution: {dict(zip(unique_views, counts))}")
                
            
            # Save ReIDPool data for current frame (before cross-view association)
            if self.reid_pool_saver is not None and self.config.get("REID_LOSS", False):
                # Read frame list to save from config
                save_frames = self.config.get("REID_POOL_SAVE_FRAMES", "all")
                should_save = False
                
                if save_frames == "all":
                    # Save all frames
                    should_save = True
                elif isinstance(save_frames, list):
                    # Save only specified frames
                    should_save = (i in save_frames)
                elif isinstance(save_frames, int):
                    # Save every N frames
                    should_save = (i % save_frames == 0)
                
                if should_save:
                    self.reid_pool_saver.add_frame_data(
                        frame_idx=i,
                        reid_pool=self.reID_pool,
                        tracks_result_dict=tracks_result_dict
                    )
            
            # Apply cross-view ReID association (IDs already offset, no conflict)
            #print("nms 382" + str(time.perf_counter()))
            if self.config.get("REID_LOSS", False):
                tracks_result_dict_copy = self.reID_pool.inference_multiview_v3(
                    view_dict_name=view_list, 
                    tracks_dict=tracks_result_dict_copy
                )
                
                # After association: update query from ReIDPool features (for next-frame tracking)
                # Uses cross-view association to strengthen query representation
                if self.config.get("USE_REID_QUERY_UPDATE", False):
                    from models.query_update_from_reid import update_query_with_reid_features
                    
                    # Per view: fetch features from ReIDPool by global ID and update query
                    for view in view_list:
                        if view not in tracks_dict:
                            continue
                        
                        tracks = tracks_dict[view]
                        
                        # Get all ReID features for this view from ReIDPool
                        if view in self.reID_pool.view_id_reid_feat_dict_list:
                            view_reid_features = self.reID_pool.view_id_reid_feat_dict_list[view]
                            
                            if len(view_reid_features) > 0:
                                # Update query in tracks_dict (for next frame)
                                update_query_with_reid_features(
                                    tracks=tracks,
                                    reid_features=view_reid_features,
                                    reid_update_weight=self.config.get("REID_UPDATE_WEIGHT", 0.1),
                                    use_dab=self.config["USE_DAB"]
                                )
            #print("nms 391" + str(time.perf_counter()))

            # ========================================
            # Step 6: Write results
            # ========================================
            if self.dataset_name == "BDD100K":
                for view_name in view_list:
                    info = dict_data[view_name]["infos"]
                    self.update_results(tracks_result=tracks_result_dict_copy[view_name], 
                                      frame_idx=i, results=bdd100k_results, img_path=info[0], view=view_name)
            else:
                for view_name in view_list:
                    # Write cross-view associated results
                    self.write_results(tracks_result=tracks_result_dict_copy[view_name], 
                                     frame_idx=i, view=view_name)
                    # Write single-view results (for comparison)
                    self.write_results2(tracks_result=tracks_result_dict[view_name],
                                      frame_idx=i, view=view_name)
            #print("nms 409" + str(time.perf_counter()))

            if self.visualize:
                os.makedirs(f"./outputs/visualize_tmp/frame_{i+1}/", exist_ok=True)
                os.system(f"mv ./outputs/visualize_tmp/query_updater/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/decoder/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/memotr/ ./outputs/visualize_tmp/frame_{i+1}/")
                os.system(f"mv ./outputs/visualize_tmp/runtime_tracker/ ./outputs/visualize_tmp/frame_{i+1}/")

        if self.visualize:
            visualize_save_dir = os.path.join("./outputs/visualize/", self.seq_name)
            os.makedirs(visualize_save_dir, exist_ok=True)
            os.system(f"mv ./outputs/visualize_tmp/* {visualize_save_dir}")

        if self.dataset_name == "BDD100K":
            with open(os.path.join(self.predict_dir, '{}.json'.format(self.seq_name)), 'w', encoding='utf-8') as f:
                json.dump(bdd100k_results, f)
        
        # Save ReIDPool data to file
        if self.reid_pool_saver is not None:
            print(f"\n{'='*80}")
            print(f"[Submitter] Saving ReIDPool data...")
            self.reid_pool_saver.save()
            
            # Print statistics
            stats = self.reid_pool_saver.get_statistics()
            print(f"[Submitter] Statistics:")
            print(f"  - Total frames: {stats['total_frames']}")
            print(f"  - Total views: {stats['total_views']}")
            print(f"  - View names: {stats['view_names']}")
            print(f"  - Feature dim: {stats['feature_dim']}")
            print(f"  - Avg IDs per frame: {stats['avg_ids_per_frame']:.1f}")
            print(f"  - Max IDs per frame: {stats['max_ids_per_frame']}")
            print(f"  - Min IDs per frame: {stats['min_ids_per_frame']}")
            print(f"{'='*80}\n")

        return

    @staticmethod
    def filter_by_score(tracks: TrackInstances, thresh: float = 0.7):
        keep = torch.max(tracks.scores, dim=-1).values > thresh
        return tracks[keep]

    @staticmethod
    def filter_by_area(tracks: TrackInstances, thresh: int = 100):
        assert len(tracks.area) == len(tracks.ids), f"Tracks' 'area' should have the same dim with 'ids'"
        keep = tracks.area > thresh
        return tracks[keep]

    @staticmethod
    def nms_tracks(tracks: TrackInstances, iou_threshold: float = 0.7):

        if len(tracks) == 0:
            return tracks
        
        # Confidence scores (max class score)
        scores = torch.max(tracks.scores, dim=-1).values  # (N,)
        
        # Sort by score descending
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep_indices = []
        suppressed = torch.zeros(len(tracks), dtype=torch.bool)
        
        for idx in sorted_indices:
            if suppressed[idx]:
                continue
            
            keep_indices.append(idx.item())
            
            # IoU between current box and remaining boxes
            if len(keep_indices) < len(tracks):
                current_box = tracks.boxes[idx].unsqueeze(0)  # (1, 4)
                
                # IoU only with later unsuppressed boxes
                for other_idx in sorted_indices:
                    if other_idx <= idx or suppressed[other_idx]:
                        continue
                    
                    other_box = tracks.boxes[other_idx].unsqueeze(0)  # (1, 4)
                    
                    # IoU (box_iou_union returns iou and union)
                    iou, _ = box_iou_union(
                        box_cxcywh_to_xyxy(current_box),
                        box_cxcywh_to_xyxy(other_box)
                    )
                    iou = iou[0, 0]  # Extract scalar IoU
                    
                    # Suppress lower-score box if IoU exceeds threshold
                    if iou > iou_threshold:
                        suppressed[other_idx] = True
        
        # Keep unsuppressed tracks
        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        return tracks[keep_indices]

    def update_results(self, tracks_result: TrackInstances, frame_idx: int, results: list, img_path: str, view: str = None):
        """
        Update BDD100K results (ID offset applied before cross-view association)
        """
        # Only be used for BDD100K:
        bdd_cls2label = {
            1: "pedestrian",
            2: "rider",
            3: "car",
            4: "truck",
            5: "bus",
            6: "train",
            7: "motorcycle",
            8: "bicycle"
        }
        frame_result = {
            "name": img_path.split("/")[-1],
            "videoName": img_path.split("/")[-1][:-12],
            # "frameIndex": int(img_path.split("/")[-1][:-4].split("-")[-1]) - 1
            "frameIndex": frame_idx,
            "labels": []
        }
        for i in range(len(tracks_result)):
            x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
            
            # Use track ID directly (offset applied before cross-view association)
            track_id = tracks_result.ids[i].item()
            ID = str(track_id)
            
            label = bdd_cls2label[tracks_result.labels[i].item() + 1]
            frame_result["labels"].append(
                {
                    "id": ID,
                    "category": label,
                    "box2d": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                }
            )
        results.append(frame_result)
        return

    def write_results(self, tracks_result: TrackInstances, frame_idx: int, view: str):
        """
        Write tracking results (ID offset applied before cross-view association)
        """
        with open(os.path.join(self.predict_dir, f"{self.seq_name}_{view}.txt"), "a") as file:
            for i in range(len(tracks_result)):
                # Supported dataset list
                supported_datasets = ["DanceTrack", "SportsMOT", "MOT17", "MOT17_SPLIT", 
                                    "UAV_V", "CAMPUS", "WILDTRACK", "MvMHAT", "DIVOTrack"]
                if self.dataset_name in supported_datasets:
                    x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
                    w, h = x2 - x1, y2 - y1
                    
                    # Use track ID directly (offset applied before cross-view association)
                    track_id = tracks_result.ids[i].item()
                    
                    result_line = f"{frame_idx+1}," \
                                  f"{track_id}," \
                                  f"{x1},{y1},{w},{h},1,-1,-1,-1\n"
                else:
                    raise ValueError(f"{self.dataset_name} dataset is not supported for submit process.")
                file.write(result_line)
        return
    def write_results2(self, tracks_result: TrackInstances, frame_idx: int, view: str):
        """
        Write single-view tracking results (for comparison), no view ID offset
        """
        with open(os.path.join(self.predict_dir, f"{self.seq_name}_{view}_single.txt"), "a") as file:
            for i in range(len(tracks_result)):
                # Supported dataset list
                supported_datasets = ["DanceTrack", "SportsMOT", "MOT17", "MOT17_SPLIT", 
                                    "UAV_V", "CAMPUS", "WILDTRACK", "MvMHAT", "DIVOTrack"]
                if self.dataset_name in supported_datasets:
                    x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
                    w, h = x2 - x1, y2 - y1
                    
                    # Single-view results need no ID offset (saved per view)
                    track_id = tracks_result.ids[i].item()
                    
                    result_line = f"{frame_idx+1}," \
                                  f"{track_id}," \
                                  f"{x1},{y1},{w},{h},1,-1,-1,-1\n"
                else:
                    raise ValueError(f"{self.dataset_name} dataset is not supported for submit process.")
                file.write(result_line)
        return

def submit(config: dict):
    submit_logger = Logger(logdir=os.path.join(config["SUBMIT_DIR"], config["SUBMIT_DATA_SPLIT"]), only_main=True)
    submit_logger.show(head="Configs:", log=config)
    submit_logger.write(log=config, filename="config.yaml", mode="w")

    assert config["SUBMIT_DIR"] is not None, f"'--submit-dir' must not be None for submit process."
    assert config["SUBMIT_MODEL1"] is not None, f"'--submit-model' must not be None for submit process."
    assert config["SUBMIT_DATA_SPLIT"] is not None, f"'--submit-data-split' must not be None for submit process."
    train_config = yaml_to_dict(path=path.join(config["SUBMIT_DIR"], "train/config.yaml"))

    data_root = config["DATA_ROOT"]
    dataset_name = train_config["DATASET"]
    config["DATASET"] = dataset_name
    dataset_split = config["SUBMIT_DATA_SPLIT"]
    outputs_dir = path.join(config["SUBMIT_DIR"], dataset_split)
    use_dab = train_config["USE_DAB"]
    det_score_thresh = config["DET_SCORE_THRESH"]
    track_score_thresh = config["TRACK_SCORE_THRESH"]
    result_score_thresh = config["RESULT_SCORE_THRESH"]
    use_motion = config["USE_MOTION"]
    motion_min_length = config["MOTION_MIN_LENGTH"]
    motion_max_length = config["MOTION_MAX_LENGTH"]
    motion_lambda = config["MOTION_LAMBDA"]
    miss_tolerance = config["MISS_TOLERANCE"]

    # Select device
    if config.get("USE_DISTRIBUTED", False):
        device = torch.device(config["DEVICE"], distributed_rank())
    else:
        device = torch.device(config["DEVICE"])
    
    # Build and load main model
    model = build_model(config=train_config)
    load_checkpoint(
        model=model,
        path=path.join(config["SUBMIT_DIR"], config["SUBMIT_MODEL1"])
    )
    # Move main model to the correct device
    model = model.to(device)
    print(f"[INFO] Model moved to device: {device}")

    reid_model = None
    reID_pool = None

    # Multi-view dataset check
    multiview_datasets = ["UAV_V", "CAMPUS", "WILDTRACK", "MvMHAT", "DIVOTrack"]
    if config["DATASET"] in multiview_datasets:
        viewpoints = ["c00"+str(i+1) for i in range(config["VIEW_POINT"])]
        
        # Build and load ReID model
        reid_model = build_reid_model(config=config)
        load_checkpoint(
            model=reid_model,
            path=path.join(config["SUBMIT_DIR"], config["SUBMIT_MODEL2"])
        )
        # Move ReID model to same device as main model
        reid_model = reid_model.to(device)
        print(f"[INFO] ReID model moved to device: {device}")
        
        # Build ReIDPool (pass neighbor filter)
        reID_pool = build_reid_pool(
            views=viewpoints, 
            max_forget_length=config["MAX_FORGET_LENGTH"], 
            training=False,
            use_frame_level_voting=True,
            config=config,
            neighbor_filter=neighbor_filter  # Pass neighbor filter
        )

    if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        data_split_dir = path.join(data_root, dataset_name, dataset_split)
    elif dataset_name == "BDD100K":
        data_split_dir = path.join(data_root, dataset_name, "images/track/", dataset_split)
    elif dataset_name in multiview_datasets:
        data_split_dir = path.join(data_root, dataset_name, "images", dataset_split)
    else:
        data_split_dir = path.join(data_root, dataset_name, "images", dataset_split)
    seq_names = os.listdir(data_split_dir)

    if is_distributed():
        model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=False)
        total_seq_names = seq_names
        seq_names = []
        for i in range(len(total_seq_names)):
            if i % distributed_world_size() == distributed_rank():
                seq_names.append(total_seq_names[i])
    
    # Create neighbor filter (before ReIDPool)
    neighbor_filter = build_neighbor_filter(config)
    if neighbor_filter is not None:
        print(f"\n{'='*80}")
        print(f"[Submit] ✅ Neighbor Filter ENABLED")
        print(f"  - Spatial Radius: {neighbor_filter.spatial_radius} pixels")
        print(f"  - K-Top Similarities: {neighbor_filter.k_top_similarities}")
        print(f"  - Min Match Ratio: {neighbor_filter.min_neighbor_match_ratio}")
        print(f"  - Similarity Threshold: {neighbor_filter.neighbor_sim_threshold}")
        print(f"  - Distance Metric: {neighbor_filter.distance_metric}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"[Submit] ⚠️  Neighbor Filter DISABLED")
        print(f"  Set 'USE_NEIGHBOR_FILTER: True' in config to enable")
        print(f"{'='*80}\n")
    
    # Create MemoryBank (same device as model)
    memory_bank = MemoryBank(
            bank_len=config.get("MEMORY_BANK_LEN", 30),
            hidden_dim=config.get("HIDDEN_DIM", 256),
            use_dab=config.get("USE_DAB", True),
            temporal_k=config.get("TEMPORAL_K", 8),
            decay_alpha=config.get("DECAY_ALPHA", 0.25),
            num_heads=config.get("NUM_HEADS", 8),
            device=str(device),  # Same device as model
            training=False,  # Inference mode
            reid_update_weight=config.get("REID_UPDATE_WEIGHT", 0.1)
        )
    memory_bank = memory_bank.to(device=device)
    print(f"[INFO] MemoryBank moved to device: {device}")

    for seq_name in seq_names:
        # Clear MemoryBank and ReIDPool at start of each sequence
        if memory_bank is not None:
            memory_bank.clear()  # Use clear() not clear_all()
        if reID_pool is not None:
            reID_pool.clear_all()
        
        seq_name = str(seq_name)
        submitter = Submitter(
            dataset_name=dataset_name,
            split_dir=data_split_dir,
            seq_name=seq_name,
            outputs_dir=outputs_dir,
            model=model,
            use_dab=use_dab,
            det_score_thresh=det_score_thresh,
            track_score_thresh=track_score_thresh,
            result_score_thresh=result_score_thresh,
            use_motion=use_motion,
            motion_min_length=motion_min_length,
            motion_max_length=motion_max_length,
            motion_lambda=motion_lambda,
            miss_tolerance=miss_tolerance,
            config=config,
            reid_model=reid_model,
            reID_pool=reID_pool,
            memory_bank=memory_bank
        )
        submitter.run_multi_view()
    return
