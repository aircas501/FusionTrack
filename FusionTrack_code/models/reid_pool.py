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
        self.keep_all_batch_reid = keep_all_batch_reid  # 保留参数以保持兼容性
        self.reid_update_weight = reid_update_weight  # 新ReID特征的权重（旧特征权重 = 1 - reid_update_weight）

        self.view_list = views
        self.training = training

        # ✅ 精简存储：只保留 reid 特征，query 由 MemoryBank 管理
        self.view_id_reid_feat_dict_list = {}  # reid features for each track in the pool
        self.view_id_life_dict_list = {}  # 生命周期计数
        
        # ✅ 跨视角关联器
        self.cross_view_associator = cross_view_associator
        
        # ✅ 帧级投票机制支持
        self.use_frame_level_voting = use_frame_level_voting
        self.frame_associator = None
        self.current_frame = 0
        
        if use_frame_level_voting:
            from models.frame_level_association import build_frame_level_associator
            if voting_config is None:
                # 默认配置
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
            # 传递邻居筛选器给帧级关联器
            self.frame_associator = build_frame_level_associator(voting_config, neighbor_filter=neighbor_filter)

        for view in self.view_list:
            self.view_id_reid_feat_dict_list[view] = {}  # reid features
            self.view_id_life_dict_list[view] = {}  # life counter

    def get(self, view: str, id: int) -> torch.Tensor:
        """
        获取指定视角和ID的ReID特征
        
        """
        if view not in self.view_list:
            raise ValueError(f"View {view} is not in the list of views.")
        
        # 返回reid特征而非query
        return self.view_id_reid_feat_dict_list[view].get(id, None)
    
    def inference_multiview(self, view_dict_name, tracks_dict, reid_model=None, normalize_feature=False):
        """
        多视角推理：使用ReID特征进行跨视角ID关联（旧版本，使用agglomerative_clustering）
        """
        every_view_num_list = [len(self.view_id_reid_feat_dict_list[view]) for view in view_dict_name]

        all_num = sum(every_view_num_list)

        dist_matrix = torch.ones((all_num, all_num))

        matrix_idx_view_id_list = []  # 记录矩阵索引的idx以及对应的view中的id
        matrix_idx_view_name_list = []  # 记录矩阵索引的idx以及对应的view中的name
        matrix_idx_reid_feature_list = []  # 记录矩阵索引的idx以及对应的reid特征

        for view in view_dict_name:
            for id in self.view_id_reid_feat_dict_list[view].keys():
                matrix_idx_view_id_list.append(id)
                matrix_idx_view_name_list.append(view)

                # ✅ 直接使用存储的reid特征，无需重新计算
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
        dist_matrix = euclidean_dist(global_feat, global_feat)  # 计算欧式距离

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
        # 如果没有配置跨视角关联器或使用旧版本，回退到旧方法
        if self.cross_view_associator is None or use_legacy:
            return self.inference_multiview(view_dict_name, tracks_dict)
        
        # 1. 收集所有视角的特征和ID
        view_features = {}
        view_ids = {}
        
        for view in view_dict_name:
            if view not in self.view_id_reid_feat_dict_list:
                continue
            
            features_list = []
            ids_list = []
            
            for id, feat in self.view_id_reid_feat_dict_list[view].items():
                if feat is not None:
                    # 确保特征是1D
                    if feat.dim() > 1:
                        feat = feat.squeeze(0)
                    features_list.append(feat)
                    ids_list.append(id)
            
            if len(features_list) > 0:
                view_features[view] = torch.stack(features_list, dim=0)  # (N_view, C)
                view_ids[view] = ids_list
        
        # 如果没有特征，直接返回
        if len(view_features) == 0:
            return tracks_dict
        
        # 2. 使用跨视角关联器进行关联
        mapping = self.cross_view_associator.associate(
            view_features=view_features,
            view_ids=view_ids,
            view_names=view_dict_name
        )
        
        # 3. 应用全局ID映射到tracks_dict
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
        # 如果没有开启帧级投票或使用旧版本，回退到V2
        if not self.use_frame_level_voting or use_legacy:
            return self.inference_multiview_v2(view_dict_name, tracks_dict, use_legacy)
        
        # 1. 准备帧级关联所需的数据（包括bboxes）
        view_detections = {}
        view_bboxes = {}  # 新增：存储每个视角的bbox信息
        #print("v3 230" + str(time.perf_counter()))
        for view in view_dict_name:

            if view not in self.view_id_reid_feat_dict_list:
                continue
            
            detections = []
            bboxes_list = []
            
            # 从tracks_dict中获取当前视角的tracks（用于提取bbox）
            view_tracks = tracks_dict.get(view, None)
            if view_tracks is not None:
                if not isinstance(view_tracks, list):
                    view_tracks = [view_tracks]
                
                # 构建 id -> bbox 的映射
                id_to_bbox = {}
                for track_list in view_tracks:
                    for idx, track_id in enumerate(track_list.ids.tolist()):
                        if hasattr(track_list, 'boxes') and idx < len(track_list.boxes):
                            id_to_bbox[track_id] = track_list.boxes[idx]
            else:
                id_to_bbox = {}
            
            for id, reid_feat in self.view_id_reid_feat_dict_list[view].items():
                if reid_feat is not None:
                    # 确保特征是tensor
                    if not isinstance(reid_feat, torch.Tensor):
                        reid_feat = torch.tensor(reid_feat)
                    
                    # 确保特征是1D（如果是2D，squeeze掉batch维度）
                    if reid_feat.dim() > 1:
                        if reid_feat.shape[0] == 1:
                            reid_feat = reid_feat.squeeze(0)
                        else:
                            # 如果有多个特征，取平均
                            reid_feat = reid_feat.mean(dim=0)
                    
                    # 添加到检测列表：(local_id, feature_tensor)
                    detections.append((id, reid_feat))
                    
                    # 添加bbox（如果存在）
                    if id in id_to_bbox:
                        bboxes_list.append(id_to_bbox[id])
                    else:
                        # 如果没有bbox，添加一个默认值（邻居筛选会跳过）
                        bboxes_list.append(torch.zeros(4))


            if len(detections) > 0:
                view_detections[view] = detections
                # 将bboxes转换为tensor
                if len(bboxes_list) > 0:
                    view_bboxes[view] = torch.stack(bboxes_list)
        #print("v3 259" + str(view) + str(time.perf_counter()))  
        # 如果没有检测，直接返回
        if len(view_detections) == 0:
            return tracks_dict
        
        # 2. 使用帧级关联器进行关联（带投票机制和邻居筛选）
        self.current_frame += 1
        mapping = self.frame_associator.associate_frame(
            frame_id=self.current_frame,
            view_detections=view_detections,
            view_names=view_dict_name,
            view_bboxes=view_bboxes  # 传递bbox信息
        )
        #print("v3 271" + str(view) + str(time.perf_counter()))        
        # 3. 应用全局ID映射到tracks_dict，这个映射没啥问题，但是就是不能加偏移
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
        重置帧计数器和投票历史（用于新视频序列开始时）
        """
        self.current_frame = 0
        if self.frame_associator is not None:
            self.frame_associator.reset()
        


    def update_pool(self, view, tracks: list = [TrackInstances], cur_frame: int = 0, reid_features: dict = None):
        with torch.no_grad():
            # 遍历所有 key，生命周期 +1
            for key in list(self.view_id_life_dict_list[view].keys()):
                self.view_id_life_dict_list[view][key] += 1
            
            # 遍历 tracks，更新 reid 特征
            for t in tracks:
                for idx, id in enumerate(t.ids):
                    id = id.item()
                    if id < 0:
                        continue
                    
                    # 重置生命周期
                    self.view_id_life_dict_list[view][id] = 0
                    
                    # 更新重识别特征
                    if reid_features is not None and id in reid_features:
                        new_reid_feat = reid_features[id]
                        
                        # 确保新特征是 1D 向量
                        if new_reid_feat.dim() > 1:
                            if new_reid_feat.shape[0] == 1:
                                new_reid_feat = new_reid_feat.squeeze(0)
                            else:
                                # 如果是序列，取平均
                                new_reid_feat = new_reid_feat.mean(dim=0)
                        
                        # ✅ detach 并移到 CPU 以节省显存
                        new_reid_feat = new_reid_feat.detach().cpu()
                        
                        if id in self.view_id_reid_feat_dict_list[view]:
                            # 已有旧特征：融合更新
                            old_reid_feat = self.view_id_reid_feat_dict_list[view][id]
                            
                            # 确保旧特征是 1D 向量
                            if old_reid_feat.dim() > 1:
                                if old_reid_feat.shape[0] == 1:
                                    old_reid_feat = old_reid_feat.squeeze(0)
                                else:
                                    old_reid_feat = old_reid_feat[-1]
                            
                            # 确保维度匹配
                            if old_reid_feat.shape != new_reid_feat.shape:
                                updated_reid_feat = new_reid_feat
                            else:
                                # 融合更新：updated = (1 - w) * old + w * new
                                updated_reid_feat = (1 - self.reid_update_weight) * old_reid_feat + \
                                                    self.reid_update_weight * new_reid_feat
                            
                            # 存储为 (1, C) 格式
                            self.view_id_reid_feat_dict_list[view][id] = updated_reid_feat.unsqueeze(0)
                        else:
                            # 新 ID：直接存储
                            self.view_id_reid_feat_dict_list[view][id] = new_reid_feat.unsqueeze(0)
            
            # 清理超出生命周期的特征
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
        
        # 返回融合后的特征（单特征，不是序列）
        return self.view_id_reid_feat_dict_list[view][id]
    
    def get_all_reid_features_by_id(self, id: int) -> dict:
        res = {}
        for view in self.view_list:
            if id in self.view_id_reid_feat_dict_list[view]:
                # 现在存储的是单特征（融合后的），直接返回
                res[view] = self.view_id_reid_feat_dict_list[view][id]
        return res
    
    def clear_batch_reid_features(self):

        for view in self.view_list:
            self.view_id_reid_feat_dict_list[view].clear()
    
    def clear_all(self):
        for view in self.view_list:
            # 显式删除tensor和梯度
            for id_key in list(self.view_id_reid_feat_dict_list[view].keys()):
                if id_key in self.view_id_reid_feat_dict_list[view]:
                    tensor = self.view_id_reid_feat_dict_list[view][id_key]
                    if tensor is not None and hasattr(tensor, 'grad'):
                        tensor.grad = None  # 清除梯度
                    del tensor  # 删除tensor引用
            
            # 清空字典
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
