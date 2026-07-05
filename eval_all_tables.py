
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets.rellis_dataset import RELLIS3DVideoDataset
from models.offroad_net import DecoupledSpatiotemporalNet

def evaluate_config(model, dataloader, device, view_mask=None, disable_tsm=False, num_samples=60):
    """
    通用测评函数：支持选择开启/屏蔽特定视角，或屏蔽 TSM 时序模块
    view_mask: 例如 [0] 表示仅保留前向视角(索引0)，置零左/右视角
    """
    all_t_preds, all_t_targets = [], []
    all_s_preds, all_s_targets = [], []
    latencies = []
    
    with torch.no_grad():
        for idx, (imgs, t_lbls, s_lbls) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            # --- 多视角消融处理 ---
            if view_mask is not None:
                masked_imgs = torch.zeros_like(imgs)
                for v_idx in view_mask:
                    masked_imgs[:, v_idx, ...] = imgs[:, v_idx, ...]
                imgs = masked_imgs
            
            # --- 测速与前向推理 ---
            start_t = time.perf_counter()
            # 若需消融 TSM，简单用把序列维度复制首帧模拟静态无时空感知
            if disable_tsm:
                imgs_static = imgs[:, :, :1, ...].repeat(1, 1, imgs.shape[2], 1, 1, 1)
                t_preds, s_preds = model(imgs_static)
            else:
                t_preds, s_preds = model(imgs)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)
            
            # 若进行了单/双视角屏蔽，计算准确率时也只统计生效视角的局部标签
            if view_mask is not None:
                t_preds_eval = t_preds[:, view_mask, :].reshape(-1, t_preds.shape[-1])
                t_lbls_eval = t_lbls[:, view_mask].reshape(-1)
            else:
                t_preds_eval = t_preds.reshape(-1, t_preds.shape[-1])
                t_lbls_eval = t_lbls.reshape(-1)
                
            all_t_preds.extend(t_preds_eval.argmax(dim=-1).cpu().numpy())
            all_t_targets.extend(t_lbls_eval.numpy())
            all_s_preds.extend(s_preds.argmax(dim=-1).cpu().numpy())
            all_s_targets.extend(s_lbls.numpy())
            
            if idx >= num_samples:
                break
                
    t_acc = accuracy_score(all_t_targets, all_t_preds) * 100
    s_acc = accuracy_score(all_s_targets, all_s_preds) * 100
    f1 = f1_score(all_t_targets, all_t_preds, average="macro", zero_division=0) * 100
    lat = np.mean(latencies[1:])
    return t_acc, s_acc, f1, lat

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在准备一键生成【表1/表2/表3/表4】指标 (运行设备: {device})...\n")
    
    dataset = RELLIS3DVideoDataset(root_dir="dataset/rellis", sequence="00000", num_frames=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = DecoupledSpatiotemporalNet().to(device)
    try:
        model.load_state_dict(torch.load("checkpoint_offroad_net.pth", map_location=device))
        print(">> 成功加载已训练模型权重：checkpoint_offroad_net.pth\n")
    except FileNotFoundError:
        print(">> 警告：未找到已训练权重，当前使用初始权重跑通各项表格基准指标！\n")
        
    model.eval()
    
    # 计算模型参数量
    total_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # ================= 跑表 1 (Table 1: 完整主方法) =================
    t_acc, s_acc, f1, lat = evaluate_config(model, dataloader, device)
    print("=" * 60)
    print("【表 1 (Table 1) - 主力方法测评输出】")
    print(f"  Method       : Ours (Multi-camera Video)")
    print(f"  Terrain Acc  : {t_acc:.2f} %")
    print(f"  Scene Acc    : {s_acc:.2f} %")
    print(f"  Overall F1   : {f1:.2f}")
    print(f"  Latency      : {lat:.2f} ms")
    print("=" * 60)

    # ================= 跑表 2 (Table 2: 时序建模消融) =================
    t_acc_cnn, s_acc_cnn, f1_cnn, _ = evaluate_config(model, dataloader, device, disable_tsm=True)
    print("\n【表 2 (Table 2) - 时序建模作用消融 (Effect of Temporal Modeling)】")
    print(f"  [CNN Backbone] -> Terrain Acc: {t_acc_cnn:.2f}% | Scene Acc: {s_acc_cnn:.2f}% | F1: {f1_cnn:.2f}")
    print(f"  [CNN + TSM   ] -> Terrain Acc: {t_acc:.2f}% | Scene Acc: {s_acc:.2f}% | F1: {f1:.2f}")
    print("=" * 60)

    # ================= 跑表 3 (Table 3: 多视角相机消融) =================
    print("\n【表 3 (Table 3) - 多视角融合消融 (Effect of Multi-camera Fusion)】")
    views_map = {
        "Front Camera (单前视)": [0],
        "Left Camera  (单左视)": [1],
        "Right Camera (单右视)": [2],
        "Front + Left (前+左) ": [0, 1],
        "Front + Right(前+右) ": [0, 2],
        "Front+Left+Right(三视)": [0, 1, 2]
    }
    for name, v_mask in views_map.items():
        ta, sa, f1_v, _ = evaluate_config(model, dataloader, device, view_mask=v_mask)
        print(f"  {name} | Terrain: {ta:5.2f}% | Scene: {sa:6.2f}% | F1: {f1_v:5.2f}")
    print("=" * 60)

    # ================= 跑表 4 (Table 4: 运行效率与资源统计) =================
    fps = 1000.0 / lat if lat > 0 else 0
    print("\n【表 4 (Table 4) - 部署运行效率分析 (Runtime Analysis)】")
    print(f"  Method       : Ours")
    print(f"  Params (M)   : {total_params_m:.2f} M")
    print(f"  FPS          : {fps:.1f} fps")
    print(f"  Latency (ms) : {lat:.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    main()
