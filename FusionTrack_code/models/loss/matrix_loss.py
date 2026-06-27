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
    # Coordinates of the global minimum
    min_row = torch.argmin(dist_matrix.min(dim=1)[0]).item()  # Min per row, then global min row
    min_col = torch.argmin(dist_matrix[min_row]).item()       # Min column in that row
    min_value = dist_matrix[min_row][min_col].item()
    if(min_value > thres):
        print("No match found")
        return -1, -1, min_value
    print(f"Found: {min_row}, {min_col}, {min_value}")
    return min_row, min_col, min_value

def merge_closest_pair(clusters, min_row, min_col, view_idx_bound):
    
    cluster1 = clusters[min_row]
    cluster2 = clusters[min_col]

    mearged_points = cluster1['points'] + cluster2['points']
    if len(mearged_points) > len(view_idx_bound):
        print("Merged point count exceeds maximum")
        return False
    for i in range(0, len(mearged_points), 1):
        for j in range(i + 1, len(mearged_points), 1):
            if get_view_name_from_idx(view_idx_bound, mearged_points[i]) == get_view_name_from_idx(view_idx_bound, mearged_points[j]):
                print("Merge would combine different points from the same view")
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
    Custom agglomerative clustering implementation.

    :param data: Input data as [[x1,y1,...], [x2,y2,...], ...]
    :param k: Target number of clusters
    :param linkage: Linkage method ('single', 'complete', 'average')
    :return: Cluster label list, e.g. [0, 1, 0, 2, ...]
    """

    # Initialize: each point is its own cluster
    total_num = dist_matrix.shape[0]
    clusters = [{'points': [i]} for i in range(total_num)]

    # Merge clusters until k remain
    while (True):
        min_row, min_col, min_value = find_closest(dist_matrix=dist_matrix, thres=0.5)
        if min_row == -1:
            break
        # Find the two closest clusters
        if min_row > min_col:
            min_row, min_col = min_col, min_row
        
        if(merge_closest_pair(clusters, min_row, min_col, view_idx_bound)):
            # Update distance matrix
            dist_matrix = update_dist_matrix(dist_matrix, min_row, min_col, view_idx_bound)  # Ensure no two targets in the same view link to the same target in another view
            # Merge the two clusters; keep IDs sorted

            clusters[min_row]['points'].extend(clusters[min_col]['points'])
            clusters[min_row]['points'].sort()  # Keep IDs ordered
            for update_id in clusters[min_row]['points']:
              clusters[update_id]['points'] = clusters[min_row]['points']
        else:
            dist_matrix[min_row][min_col] = float("inf")

    clusters_id = []
    view_num = len(view_idx_bound)
    for cluster in clusters:
        tmp = cluster['points']
        while (len(tmp) < view_num):
            tmp.append(-1)  # Pad to have one entry per view
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
    Union-Find (disjoint set) for efficient equivalence-class management.
    Used for global ID fusion in cross-view target association.
    """
    def __init__(self, n):
        """
        Initialize union-find structure.

        Args:
            n: Number of elements
        """
        self.parent = list(range(n))  # Parent of each element
        self.rank = [0] * n  # Tree rank (for union-by-rank)
        self.count = n  # Number of connected components
    
    def find(self, x):
        """
        Find root of element x (with path compression).

        Args:
            x: Element index
        Returns:
            Root index
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """
        Merge the sets containing x and y.

        Args:
            x: Element index
            y: Element index
        Returns:
            True if merged; False if already in the same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
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
        Check whether x and y are in the same set.

        Args:
            x: Element index
            y: Element index
        Returns:
            Whether they are connected
        """
        return self.find(x) == self.find(y)
    
    def get_groups(self):
        """
        Get all connected components (equivalence sets).

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
    Cross-view Hungarian matching from a distance matrix.

    :param dist_matrix: Distance matrix (N, N), N = total tracks across all views
    :param view_idx_bound: Per-view start indices, e.g. [0, 10, 20] => View1:[0-9], View2:[10-19], View3:[20-...]
    :param threshold: Distance threshold; pairs above this are not matched
    :return: List of matches; each element is a list of IDs grouped together
    """

    total_num = dist_matrix.shape[0]
    matched = [False] * total_num
    matches = []

    view_num = len(view_idx_bound) - 1

    for i in range(view_num):
        for j in range(i+1, view_num):
            # Index ranges for the two views
            start_i, end_i = view_idx_bound[i], view_idx_bound[i+1]
            start_j, end_j = view_idx_bound[j], view_idx_bound[j+1]

            sub_matrix = dist_matrix[start_i:end_i, start_j:end_j].copy()

            # Set positions above threshold to a large value
            sub_matrix[sub_matrix > threshold] = 1e6

            # Skip if sub_matrix is all inf
            if np.all(sub_matrix == 1e6):
                continue

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(sub_matrix)

            for r, c in zip(row_ind, col_ind):
                real_r = start_i + r
                real_c = start_j + c
                if dist_matrix[real_r, real_c] <= threshold:
                    matches.append([real_r, real_c])
                    matched[real_r] = True
                    matched[real_c] = True

    # Unmatched items form their own groups
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

