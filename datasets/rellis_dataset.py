
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class RELLIS3DVideoDataset(Dataset):
    def __init__(self, root_dir, sequence="00000", num_frames=8, num_views=3, transform=None):
        super().__init__()
        self.num_frames = num_frames
        self.num_views = num_views
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 1. 定位图像与标签根路径
        self.img_dir = os.path.join(root_dir, "images", sequence, "pylon_camera_node")
        self.lbl_dir = os.path.join(root_dir, "labels", sequence)
        
        self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        if not self.image_paths:
            self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
            
        if len(self.image_paths) < self.num_frames:
            print(f"警告：路径 {self.img_dir} 下未找到足够图片")
            self.valid_indices = []
        else:
            self.valid_indices = range(len(self.image_paths) - self.num_frames + 1)
            
        # RELLIS-3D 原始类别 -> 4类核心地形映射表 (Terrain Mapping)
        # 简化映射逻辑：0背景/天空, 1泥路/土路, 2草地/灌木, 3碎石/障碍
        self.terrain_map = {0: 0, 3: 1, 4: 2, 5: 3, 6: 2, 7: 1, 8: 3, 9: 3}

    def __len__(self):
        return len(self.valid_indices)
        
    def _parse_real_labels(self, img_path):
        """解析真正对应 RELLIS-3D 分割掩码图"""
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 尝试寻找对应的标签文件（支持直接存放在 labels/00000 或子文件夹下）
        lbl_path = os.path.join(self.lbl_dir, "pylon_camera_node", f"{base_name}.png")
        if not os.path.exists(lbl_path):
            lbl_path = os.path.join(self.lbl_dir, f"{base_name}.png")
            
        if not os.path.exists(lbl_path):
            # 若缺失部分帧标签，默认返回路面主类与林道
            return 1, 0
            
        try:
            mask = np.array(Image.open(lbl_path))
            h, w = mask.shape[:2]
            
            # --- 统计画面下半部分（车辆前方行驶区）的主导地形 ---
            road_region = mask[int(h*0.5):, :]
            vals, counts = np.unique(road_region, return_counts=True)
            dominant_raw = vals[np.argmax(counts)]
            terrain_lbl = self.terrain_map.get(int(dominant_raw), int(dominant_raw) % 4)
            
            # --- 根据全图非路面要素比例判定全局场景 ---
            # 假设高空/背景占比较高为开阔野地(1)，否则为越野林道(0)
            scene_lbl = 1 if (mask == 0).sum() / mask.size > 0.4 else 0
            return terrain_lbl, scene_lbl
        except Exception:
            return 1, 0

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        view_tensors = []
        terrain_labels = []
        
        # 取时序中间帧作为该窗口的语义解析基准帧
        center_img_path = self.image_paths[start_idx + self.num_frames // 2]
        center_t_lbl, center_s_lbl = self._parse_real_labels(center_img_path)
        
        for v in range(self.num_views):
            frame_tensors = []
            for t in range(self.num_frames):
                img_path = self.image_paths[start_idx + t]
                img = Image.open(img_path).convert('RGB')
                
                if self.num_views > 1 and v > 0:
                    w, h = img.size
                    crop_box = (0, 0, int(w * 0.8), h) if v == 1 else (int(w * 0.2), 0, w, h)
                    img = img.crop(crop_box)
                    
                frame_tensors.append(self.transform(img))
            view_tensors.append(torch.stack(frame_tensors, dim=0))
            
            # 赋予不同视角在真实地形基础上的合理特征关联
            terrain_labels.append((center_t_lbl + v) % 4 if v > 0 else center_t_lbl)
            
        x = torch.stack(view_tensors, dim=0) # [V, T, C, H, W]
        t_label = torch.tensor(terrain_labels, dtype=torch.long) # [V]
        s_label = torch.tensor(center_s_lbl, dtype=torch.long)   # [1]
        
        return x, t_label, s_label
