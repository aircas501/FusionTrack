import torch
from torch import nn
import numpy as np


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

def find_closest(dist_matrix, thres):
    # 全局最小值的坐标
    min_row = torch.argmin(dist_matrix.min(dim=1)[0]).item()  # 先找每行最小，再找全局最小行
    min_col = torch.argmin(dist_matrix[min_row]).item()       # 在最小行中找列
    min_value = dist_matrix[min_row][min_col].item()
    if(min_value > thres):
        print("找不到了")
        return -1, -1, min_value
    print(f"找到了：{min_row}, {min_col}, {min_value}")
    return min_row, min_col, min_value

def merge_closest_pair(clusters, min_row, min_col, view_idx_bound):
    
    cluster1 = clusters[min_row]
    cluster2 = clusters[min_col]

    mearged_points = cluster1['points'] + cluster2['points']
    if len(mearged_points) > len(view_idx_bound):
        print("合并后点数超过最大值")
        return False
    for i in range(0, len(mearged_points), 1):
        for j in range(i + 1, len(mearged_points), 1):
            if get_view_name_from_idx(view_idx_bound, mearged_points[i]) == get_view_name_from_idx(view_idx_bound, mearged_points[j]):
                print("合并后存在相同视图中的不同点")
                return False
    return True
    

def get_view_name_from_idx(view_idx_bound, cur_idx):
    start = 0
    for idx, i in enumerate(view_idx_bound):
        end = start + i
        if start <= cur_idx < end:
            return idx + 1
        start = end
    return len(view_idx_bound)

def get_idx_range_from_view_idx(view_idx_bound, cur_idx):
    start = 0
    for idx, i in enumerate(view_idx_bound):
        end = start + i
        if start <= cur_idx < end:
            return start, end - 1
        start = end
    return -1, -1

def update_dist_matrix(dist_matrix, min_row, min_col, view_idx_bound):
    start, end = get_idx_range_from_view_idx(view_idx_bound, min_col)
    for i in range(start, end + 1):
        dist_matrix[min_row][i] = float('inf')
    return dist_matrix
    
def agglomerative_clustering(dist_matrix, view_idx_bound):
    """
    自行实现的层次聚类算法
    :param data: 输入数据，格式为 [[x1,y1,...], [x2,y2,...], ...]
    :param k: 目标簇数量
    :param linkage: 连接方式 ('single', 'complete', 'average')
    :return: 聚类标签列表，如 [0, 1, 0, 2, ...]
    """

    # 初始化：每个点作为一个簇
    total_num = dist_matrix.shape[0]
    clusters = [{'points': [i]} for i in range(total_num)]

    # 逐步合并簇，直到剩下k个
    while (True):
        min_row, min_col, min_value = find_closest(dist_matrix=dist_matrix, thres=0.5)
        if min_row == -1:
            break
        # 找出距离最近的两个簇
        if min_row > min_col:
            min_row, min_col = min_col, min_row
        
        if(merge_closest_pair(clusters, min_row, min_col, view_idx_bound)):
            #更新距离矩阵
            dist_matrix = update_dist_matrix(dist_matrix, min_row, min_col, view_idx_bound)#保证同一视角目标中不会有两个和另一视角中的某一目标关联
            # merge两个簇，别忘了id排序

            clusters[min_row]['points'].extend(clusters[min_col]['points'])
            clusters[min_row]['points'].sort()  # 确保id有序
            for update_id in clusters[min_row]['points']:
              clusters[update_id]['points'] = clusters[min_row]['points']
        else:
            dist_matrix[min_row][min_col] = float("inf")

    clusters_id = []
    view_num = len(view_idx_bound)
    for cluster in clusters:
        tmp = cluster['points']
        while (len(tmp) < view_num):
            tmp.append(-1)#这里是保证必须有视角数目的id
        clusters_id.append(tmp)
    uniques = np.unique(np.array(clusters_id), axis=0)
    res = []
    for idx, tmp in enumerate(uniques):
        res.append(uniques[idx][uniques[idx] != -1].tolist())
    return res

import numpy as np
from scipy.optimize import linear_sum_assignment


class UnionFind:
    """
    并查集数据结构，用于高效管理和合并等价集合
    用于跨视角目标关联中的全局ID融合
    """
    def __init__(self, n):
        """
        初始化并查集
        Args:
            n: 元素数量
        """
        self.parent = list(range(n))  # 每个元素的父节点
        self.rank = [0] * n  # 树的秩（用于按秩合并优化）
        self.count = n  # 连通分量数量
    
    def find(self, x):
        """
        查找元素x的根节点（带路径压缩）
        Args:
            x: 元素索引
        Returns:
            根节点索引
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x, y):
        """
        合并元素x和y所在的集合
        Args:
            x: 元素索引
            y: 元素索引
        Returns:
            是否成功合并（如果已在同一集合则返回False）
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
    
    def is_connected(self, x, y):
        """
        判断元素x和y是否在同一集合
        Args:
            x: 元素索引
            y: 元素索引
        Returns:
            是否连通
        """
        return self.find(x) == self.find(y)
    
    def get_groups(self):
        """
        获取所有的连通分量（等价集合）
        Returns:
            dict: {root: [members]}
        """
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups


def cross_view_matching(dist_matrix, view_idx_bound, threshold=0.5):
    """
    基于距离矩阵进行跨视角匈牙利匹配
    :param dist_matrix: 输入的距离矩阵 (N, N)，N是所有视角track的总数
    :param view_idx_bound: 每个视角的起始索引，如[0, 10, 20]表示View1:[0-9], View2:[10-19], View3:[20-...]
    :param threshold: 距离阈值，大于该值的pair不匹配
    :return: 匹配结果列表，每个元素是一个list，表示聚到一起的id集合
    """

    total_num = dist_matrix.shape[0]
    matched = [False] * total_num
    matches = []

    view_num = len(view_idx_bound) - 1

    for i in range(view_num):
        for j in range(i+1, view_num):
            # 分别取出两个视角的目标索引范围
            start_i, end_i = view_idx_bound[i], view_idx_bound[i+1]
            start_j, end_j = view_idx_bound[j], view_idx_bound[j+1]

            sub_matrix = dist_matrix[start_i:end_i, start_j:end_j].copy()

            # 超过阈值的位置设置成很大
            sub_matrix[sub_matrix > threshold] = 1e6

            # 如果sub_matrix都是inf就跳过
            if np.all(sub_matrix == 1e6):
                continue

            # 使用匈牙利算法匹配
            row_ind, col_ind = linear_sum_assignment(sub_matrix)

            for r, c in zip(row_ind, col_ind):
                real_r = start_i + r
                real_c = start_j + c
                if dist_matrix[real_r, real_c] <= threshold:
                    matches.append([real_r, real_c])
                    matched[real_r] = True
                    matched[real_c] = True

    # 把没有被匹配到的单独作为一组
    for idx in range(total_num):
        if not matched[idx]:
            matches.append([idx])

    return matches



class MatrixLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        #dist_mat = cosine_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


