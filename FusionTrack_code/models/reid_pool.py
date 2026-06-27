from collections import defaultdict
import torch
from structures.track_instances import TrackInstances
from models.loss.matrix_loss import MatrixLoss
from models.loss.matrix_loss import *
from models.cross_view_association import CrossViewAssociator
from models.cross_view_association import build_cross_view_associator


class ReIDPool:
    def __init__(self, views: list = [str], max_forget_length: int = 20, training: bool = True, 
                 keep_all_batch_reid: bool = True, reid_update_weight: float = 0.3,
                 cross_view_associator: CrossViewAssociator = None,
                 use_frame_level_voting: bool = False, voting_config: dict = None,
                 neighbor_filter = None):
        super(ReIDPool, self).__init__()
    
        self.max_forget_length = max_forget_length
        self.keep_all_batch_reid = keep_all_batch_reid  # kept for backward compatibility
        self.reid_update_weight = reid_update_weight  # weight for new ReID features (old weight = 1 - reid_update_weight)

        self.view_list = views
        self.training = training

        # Compact storage: ReID features only; queries managed by MemoryBank
        self.view_id_reid_feat_dict_list = {}  # reid features for each track in the pool
        self.view_id_life_dict_list = {}  # life-cycle counter
        
        # Cross-view associator
        self.cross_view_associator = cross_view_associator
        
        # Frame-level voting support
        self.use_frame_level_voting = use_frame_level_voting
        self.frame_associator = None
        self.current_frame = 0
        
        if use_frame_level_voting:
            from models.frame_level_association import build_frame_level_associator
            if voting_config is None:
                # default config
                voting_config = {
                    "CROSS_VIEW_STRATEGY": "pairwise_hungarian",
                    "CROSS_VIEW_DISTANCE": "euclidean",
                    "CROSS_VIEW_THRESHOLD": 1.5,
                    "CROSS_VIEW_NORMALIZE": True,
                    "CROSS_VIEW_ONE_TO_ONE": True,
                    "VOTING_STRATEGY": "sliding_window",
                    "WINDOW_SIZE": 10,
                    "MIN_VOTES": 5,
                }#hierarchical_clustering
            # pass neighbor filter to frame-level associator
            self.frame_associator = build_frame_level_associator(voting_config, neighbor_filter=neighbor_filter)

        for view in self.view_list:
            self.view_id_reid_feat_dict_list[view] = {}  # reid features
            self.view_id_life_dict_list[view] = {}  # life counter

    def get(self, view: str, id: int) -> torch.Tensor:
        """
        Get ReID feature for the given view and ID.
        
        """
        if view not in self.view_list:
            raise ValueError(f"View {view} is not in the list of views.")
        
        # return ReID feature, not query
        return self.view_id_reid_feat_dict_list[view].get(id, None)
    
    def inference_multiview(self, view_dict_name, tracks_dict, reid_model=None, normalize_feature=False):
        """
        Multi-view inference: cross-view ID association via ReID features (legacy agglomerative_clustering).
        """
        every_view_num_list = [len(self.view_id_reid_feat_dict_list[view]) for view in view_dict_name]

        all_num = sum(every_view_num_list)

        dist_matrix = torch.ones((all_num, all_num))

        matrix_idx_view_id_list = []  # matrix index -> view-local id
        matrix_idx_view_name_list = []  # matrix index -> view name
        matrix_idx_reid_feature_list = []  # matrix index -> ReID feature

        for view in view_dict_name:
            for id in self.view_id_reid_feat_dict_list[view].keys():
                matrix_idx_view_id_list.append(id)
                matrix_idx_view_name_list.append(view)

                # use stored ReID features directly; no recomputation
                reid_feat = self.view_id_reid_feat_dict_list[view][id]
                if reid_feat.dim() > 1:
                    reid_feat = reid_feat.squeeze(0)  # (C,)
                matrix_idx_reid_feature_list.append(reid_feat)

        if len(matrix_idx_reid_feature_list) == 0:
            return tracks_dict
        
        matrix_idx_reid_feature_list = torch.stack(matrix_idx_reid_feature_list, dim=0)

        if normalize_feature:
            global_feat = normalize(matrix_idx_reid_feature_list, axis=-1)
        else:
            global_feat = matrix_idx_reid_feature_list
        dist_matrix = euclidean_dist(global_feat, global_feat)  # Euclidean distance

        start_idx = 0
        
        for num in every_view_num_list:
            end_idx = start_idx + num
            dist_matrix[start_idx:end_idx, start_idx:end_idx] = float('inf')
            start_idx = end_idx
        
        for i in range(all_num):
            for j in range(0, i+1):
                dist_matrix[i][j] = float('inf')

        cluster_ids = agglomerative_clustering(dist_matrix, every_view_num_list)

        for i, cluster_id_list in enumerate(cluster_ids):
            start_idx = matrix_idx_view_id_list[cluster_id_list[0]]
            all_change_idx = np.array(matrix_idx_view_id_list)[cluster_id_list]
            for j, node_id in enumerate(cluster_id_list):
                for idx, k in enumerate(tracks_dict[matrix_idx_view_name_list[node_id]].ids.tolist()):
                    if k in all_change_idx:
                        tracks_dict[matrix_idx_view_name_list[node_id]].ids[idx] = start_idx

        return tracks_dict
    
    def inference_multiview_v2(self, view_dict_name, tracks_dict, use_legacy=False):
        # fall back to legacy method if no cross-view associator or use_legacy
        if self.cross_view_associator is None or use_legacy:
            return self.inference_multiview(view_dict_name, tracks_dict)
        
        # 1. Collect features and IDs from all views
        view_features = {}
        view_ids = {}
        
        for view in view_dict_name:
            if view not in self.view_id_reid_feat_dict_list:
                continue
            
            features_list = []
            ids_list = []
            
            for id, feat in self.view_id_reid_feat_dict_list[view].items():
                if feat is not None:
                    # ensure features are 1D
                    if feat.dim() > 1:
                        feat = feat.squeeze(0)
                    features_list.append(feat)
                    ids_list.append(id)
            
            if len(features_list) > 0:
                view_features[view] = torch.stack(features_list, dim=0)  # (N_view, C)
                view_ids[view] = ids_list
        
        # return early if no features
        if len(view_features) == 0:
            return tracks_dict
        
        # 2. Associate via cross-view associator
        mapping = self.cross_view_associator.associate(
            view_features=view_features,
            view_ids=view_ids,
            view_names=view_dict_name
        )
        
        # 3. Apply global ID mapping to tracks_dict
        for view_name in view_dict_name:
            if view_name not in tracks_dict:
                continue
            
            tracks = tracks_dict[view_name]
            if not isinstance(tracks, list):
                tracks = [tracks]
            
            for track_list in tracks:
                for idx, local_id in enumerate(track_list.ids.tolist()):
                    key = (view_name, local_id)
                    if key in mapping:
                        global_id = mapping[key]
                        track_list.ids[idx] = global_id
        
        return tracks_dict
    
    def inference_multiview_v3(self, view_dict_name, tracks_dict, use_legacy=False):
        import time
        # fall back to V2 if frame-level voting disabled or use_legacy
        if not self.use_frame_level_voting or use_legacy:
            return self.inference_multiview_v2(view_dict_name, tracks_dict, use_legacy)
        
        # 1. Prepare data for frame-level association (incl. bboxes)
        view_detections = {}
        view_bboxes = {}  # per-view bbox info
        #print("v3 230" + str(time.perf_counter()))
        for view in view_dict_name:

            if view not in self.view_id_reid_feat_dict_list:
                continue
            
            detections = []
            bboxes_list = []
            
            # get current-view tracks from tracks_dict (for bbox extraction)
            view_tracks = tracks_dict.get(view, None)
            if view_tracks is not None:
                if not isinstance(view_tracks, list):
                    view_tracks = [view_tracks]
                
                # build id -> bbox mapping
                id_to_bbox = {}
                for track_list in view_tracks:
                    for idx, track_id in enumerate(track_list.ids.tolist()):
                        if hasattr(track_list, 'boxes') and idx < len(track_list.boxes):
                            id_to_bbox[track_id] = track_list.boxes[idx]
            else:
                id_to_bbox = {}
            
            for id, reid_feat in self.view_id_reid_feat_dict_list[view].items():
                if reid_feat is not None:
                    # ensure feature is a tensor
                    if not isinstance(reid_feat, torch.Tensor):
                        reid_feat = torch.tensor(reid_feat)
                    
                    # ensure features are 1D (squeeze batch dim if 2D)
                    if reid_feat.dim() > 1:
                        if reid_feat.shape[0] == 1:
                            reid_feat = reid_feat.squeeze(0)
                        else:
                            # average if multiple features
                            reid_feat = reid_feat.mean(dim=0)
                    
                    # add to detection list: (local_id, feature_tensor)
                    detections.append((id, reid_feat))
                    
                    # add bbox if available
                    if id in id_to_bbox:
                        bboxes_list.append(id_to_bbox[id])
                    else:
                        # default bbox if missing (neighbor filter will skip)
                        bboxes_list.append(torch.zeros(4))


            if len(detections) > 0:
                view_detections[view] = detections
                # stack bboxes into tensor
                if len(bboxes_list) > 0:
                    view_bboxes[view] = torch.stack(bboxes_list)
        #print("v3 259" + str(view) + str(time.perf_counter()))  
        # return early if no detections
        if len(view_detections) == 0:
            return tracks_dict
        
        # 2. Associate via frame-level associator (voting + neighbor filter)
        self.current_frame += 1
        mapping = self.frame_associator.associate_frame(
            frame_id=self.current_frame,
            view_detections=view_detections,
            view_names=view_dict_name,
            view_bboxes=view_bboxes  # pass bbox info
        )
        #print("v3 271" + str(view) + str(time.perf_counter()))        
        # 3. Apply global ID mapping to tracks_dict (no view offset)
        for view_name in view_dict_name:
            if view_name not in tracks_dict:
                continue
            
            tracks = tracks_dict[view_name]
            if not isinstance(tracks, list):
                tracks = [tracks]
            
            for track_list in tracks:
                for idx, local_id in enumerate(track_list.ids.tolist()):
                    key = (view_name, local_id)
                    if key in mapping:
                        global_id = mapping[key]
                        track_list.ids[idx] = global_id
        #print("v3 286" + str(view) + str(time.perf_counter()))        

        return tracks_dict
    
    def reset_frame_counter(self):
        """
        Reset frame counter and voting history (start of a new video sequence).
        """
        self.current_frame = 0
        if self.frame_associator is not None:
            self.frame_associator.reset()
        


    def update_pool(self, view, tracks: list = [TrackInstances], cur_frame: int = 0, reid_features: dict = None):
        with torch.no_grad():
            # increment life counter for all keys
            for key in list(self.view_id_life_dict_list[view].keys()):
                self.view_id_life_dict_list[view][key] += 1
            
            # iterate tracks and update ReID features
            for t in tracks:
                for idx, id in enumerate(t.ids):
                    id = id.item()
                    if id < 0:
                        continue
                    
                    # reset life counter
                    self.view_id_life_dict_list[view][id] = 0
                    
                    # update ReID feature
                    if reid_features is not None and id in reid_features:
                        new_reid_feat = reid_features[id]
                        
                        # ensure new feature is 1D
                        if new_reid_feat.dim() > 1:
                            if new_reid_feat.shape[0] == 1:
                                new_reid_feat = new_reid_feat.squeeze(0)
                            else:
                                # average if sequence
                                new_reid_feat = new_reid_feat.mean(dim=0)
                        
                        # detach and move to CPU to save GPU memory
                        new_reid_feat = new_reid_feat.detach().cpu()
                        
                        if id in self.view_id_reid_feat_dict_list[view]:
                            # existing feature: fused update
                            old_reid_feat = self.view_id_reid_feat_dict_list[view][id]
                            
                            # ensure old feature is 1D
                            if old_reid_feat.dim() > 1:
                                if old_reid_feat.shape[0] == 1:
                                    old_reid_feat = old_reid_feat.squeeze(0)
                                else:
                                    old_reid_feat = old_reid_feat[-1]
                            
                            # ensure matching dimensions
                            if old_reid_feat.shape != new_reid_feat.shape:
                                updated_reid_feat = new_reid_feat
                            else:
                                # fused update: updated = (1 - w) * old + w * new
                                updated_reid_feat = (1 - self.reid_update_weight) * old_reid_feat + \
                                                    self.reid_update_weight * new_reid_feat
                            
                            # store as (1, C)
                            self.view_id_reid_feat_dict_list[view][id] = updated_reid_feat.unsqueeze(0)
                        else:
                            # new ID: store directly
                            self.view_id_reid_feat_dict_list[view][id] = new_reid_feat.unsqueeze(0)
            
            # remove features past max life
            keys_to_remove = []
            for key in list(self.view_id_life_dict_list[view].keys()):
                if self.view_id_life_dict_list[view][key] > self.max_forget_length:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.view_id_life_dict_list[view].pop(key)
                if key in self.view_id_reid_feat_dict_list[view]:
                    self.view_id_reid_feat_dict_list[view].pop(key)
            
            torch.cuda.empty_cache()
        return
    
    def get_reid_track_query_embed_record(self):
        
        return {}
    
    def get_reid_features(self, view: str, id: int) -> torch.Tensor:
        if view not in self.view_list:
            raise ValueError(f"View {view} is not in the list of views.")
        
        if id not in self.view_id_reid_feat_dict_list[view]:
            return None
        
        # return fused single feature (not a sequence)
        return self.view_id_reid_feat_dict_list[view][id]
    
    def get_all_reid_features_by_id(self, id: int) -> dict:
        res = {}
        for view in self.view_list:
            if id in self.view_id_reid_feat_dict_list[view]:
                # stored as single fused feature; return directly
                res[view] = self.view_id_reid_feat_dict_list[view][id]
        return res
    
    def clear_batch_reid_features(self):

        for view in self.view_list:
            self.view_id_reid_feat_dict_list[view].clear()
    
    def clear_all(self):
        for view in self.view_list:
            # explicitly delete tensors and gradients
            for id_key in list(self.view_id_reid_feat_dict_list[view].keys()):
                if id_key in self.view_id_reid_feat_dict_list[view]:
                    tensor = self.view_id_reid_feat_dict_list[view][id_key]
                    if tensor is not None and hasattr(tensor, 'grad'):
                        tensor.grad = None  # clear gradient
                    del tensor  # drop tensor reference
            
            # clear dicts
            self.view_id_reid_feat_dict_list[view].clear()
            self.view_id_life_dict_list[view].clear()
    
    
def build(views, max_forget_length, training, keep_all_batch_reid=True, reid_update_weight=0.3,
          cross_view_associator=None, use_frame_level_voting=False, voting_config=None, config=None,
          neighbor_filter=None):

    return ReIDPool(views=views, 
                   max_forget_length=max_forget_length, 
                   training=training,
                   keep_all_batch_reid=keep_all_batch_reid,
                   reid_update_weight=reid_update_weight,
                   cross_view_associator=cross_view_associator,
                   use_frame_level_voting=use_frame_level_voting,
                   voting_config=voting_config,
                   neighbor_filter=neighbor_filter)
