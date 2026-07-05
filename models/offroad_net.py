import torch
import torch.nn as nn
import torchvision.models as models
from models.tsm_module import TemporalShift

class DecoupledSpatiotemporalNet(nn.Module):
    def __init__(self, num_terrain_classes=4, num_scene_classes=2, num_frames=8, num_views=3):
        super().__init__()
        self.num_frames = num_frames
        self.num_views = num_views
        
        # 1. 基础 Backbone (以 ResNet18 为例) + 嵌入 TSM 模块
        base_model = models.resnet18(pretrained=True)
        base_model.layer2[0] = TemporalShift(base_model.layer2[0], n_segment=num_frames)
        base_model.layer3[0] = TemporalShift(base_model.layer3[0], n_segment=num_frames)
        
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        feat_dim = 512
        
        # 2. 多视角空间融合模块 (Spatial Fusion Layer)
        # 将各视角特征拼接后通过高吞吐 1x1 卷积融合
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(feat_dim * num_views, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3. 解耦预测头 (Dual-head Prediction)
        # 局部视角地形分类分支
        self.terrain_head = nn.Linear(feat_dim, num_terrain_classes)
        # 全局融合场景分类分支
        self.scene_head = nn.Linear(feat_dim, num_scene_classes)

    def forward(self, x):
        # 输入形状: [B, V, T, C, H, W]
        B, V, T, C, H, W = x.shape
        x = x.view(B * V * T, C, H, W)
        
        # 提取时空特征
        feats = self.backbone(x) # [B*V*T, 512, 1, 1]
        feats = feats.view(B, V, T, -1) # [B, V, T, 512]
        
        # 沿时间维度均值池化，提取稳定的时序宏观特征 Z_i
        z_i = feats.mean(dim=2) # [B, V, 512]
        
        # --- 分支一：每个局部视角的独立地形预测 ---
        terrain_preds = self.terrain_head(z_i) # [B, V, num_terrain_classes]
        
        # --- 分支二：多视角全局融合场景预测 ---
        z_concat = z_i.view(B, V * 512, 1, 1)
        z_fusion = self.spatial_fusion(z_concat).squeeze(-1).squeeze(-1) # [B, 512]
        scene_preds = self.scene_head(z_fusion) # [B, num_scene_classes]
        
        return terrain_preds, scene_preds