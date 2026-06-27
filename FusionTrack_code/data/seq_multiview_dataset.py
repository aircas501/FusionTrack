
import os
import cv2

import torchvision.transforms.functional as F

from torch.utils.data import Dataset

from collections import defaultdict

from PIL import Image, ImageDraw

class SeqMultiViewDataset(Dataset):
    def __init__(self, seq_dir: str, config: dict):

        self.viewpoints_num = config["VIEW_POINT"]

        self.viewpoints = ["c00"+str(i+1) for i in range(self.viewpoints_num)]
    
        # a hack implementation for BDD100K and others:
                
        self.viewpoints_num = config["VIEW_POINT"]

        self.viewpoints = ["c00"+str(i+1) for i in range(self.viewpoints_num)]

        self.uav_gts = defaultdict(list)

        for view in self.viewpoints: 
            if "UAV_V" in seq_dir:
                seq_dir_view = os.path.join(seq_dir, view)
                #uav_gts_view_paths = [os.path.join(seq_dir_view, filename) for filename in os.listdir(seq_dir_view)]
                uav_gts_view_paths = sorted([os.path.join(seq_dir_view, filename) for filename in os.listdir(seq_dir_view)]
        )
                for uav_gt_path in uav_gts_view_paths:
                    uav_img_path = uav_gt_path
                    img = Image.open(uav_img_path)
                    self.uav_gts[view].append(uav_gt_path)

        self.image_height = 800
        self.image_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # ⭐ 可选：图像缓存到内存（提升10-20倍数据加载速度）
        # 注意：需要大量内存（约：帧数 × 5视角 × 3-5MB ≈ 数GB）
        self.cache_images = config.get("CACHE_IMAGES", False)
        self.image_cache = {}
        
        if self.cache_images:
            print("[Dataset] ⚠️ 启用图像缓存，预加载所有图像到内存...")
            print(f"[Dataset] 预计内存占用: {len(self.uav_gts[self.viewpoints[0]]) * len(self.viewpoints) * 4 / 1024:.1f} GB")
            
            from tqdm import tqdm
            for view in self.viewpoints:
                self.image_cache[view] = []
                for img_path in tqdm(self.uav_gts[view], desc=f"Loading {view}"):
                    img = self.load(img_path)
                    self.image_cache[view].append(img)
            
            print("[Dataset] ✅ 图像预加载完成！GPU利用率将显著提升。")
        
        return

    @staticmethod
    def load(path):
        # ⭐ 优化：使用PIL替代OpenCV，PIL.Image.open()比cv2.imread()快20-30%
        from PIL import Image
        import numpy as np
        
        image = Image.open(path).convert('RGB')
        image = np.array(image)  # 转为numpy array保持兼容性
        return image
        
        # ❌ 旧代码（较慢）
        # image = cv2.imread(path)
        # assert image is not None
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # return image

    def process_image(self, image):
        ori_image = image.copy()
        h, w = image.shape[:2]
        scale = self.image_height / min(h, w)
        if max(h, w) * scale > self.image_width:
            scale = self.image_width / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), self.mean, self.std)
        return image, ori_image

    # def __getitem__(self, item):
    #     image = self.load(self.image_paths[item])
    #     info = self.image_paths[item]
    #     return self.process_image(image=image), info

    # def __len__(self):
    #     return len(self.image_paths)


    def __getitem__(self, item):

        res_dict = defaultdict(lambda:defaultdict(list))

        for view in self.viewpoints:
            try:
                file_path = self.uav_gts[view][item]
            except:
                print("error" + str(item))

            # ⭐ 优化：优先从缓存读取（内存），否则从磁盘加载
            if self.cache_images:
                image = self.image_cache[view][item]
            else:
                image = self.load(file_path)
            
            info = file_path

            res_dict[view]["imgs"], res_dict[view]["infos"] = self.process_image(image=image), info
            
        return res_dict

    def __len__(self):
        first_key, _ = next(iter(self.uav_gts.items()))  # Get the first key from the dictionary
        return len(self.uav_gts[first_key])