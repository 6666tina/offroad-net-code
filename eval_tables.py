import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets.rellis_dataset import RELLIS3DVideoDataset
from models.offroad_net import DecoupledSpatiotemporalNet

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在启动论文评估脚本 (测试设备: {device})...")
    
    dataset = RELLIS3DVideoDataset(root_dir="dataset/rellis", sequence="00000", num_frames=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = DecoupledSpatiotemporalNet().to(device)
    # 自动尝试加载训练好的权重
    try:
        model.load_state_dict(torch.load("checkpoint_offroad_net.pth", map_location=device))
        print("成功加载已训练权重 checkpoint_offroad_net.pth")
    except FileNotFoundError:
        print("警告：未找到权重文件，将使用初始化模型进行性能基准与时延测试。")
        
    model.eval()
    all_t_preds, all_t_targets = [], []
    all_s_preds, all_s_targets = [], []
    latencies = []
    
    with torch.no_grad():
        for idx, (imgs, t_lbls, s_lbls) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            # 精准测量推理延迟 (加入 GPU 同步保障计时准确)
            start_t = time.perf_counter()
            t_preds, s_preds = model(imgs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)
            
            all_t_preds.extend(t_preds.argmax(dim=-1).view(-1).cpu().numpy())
            all_t_targets.extend(t_lbls.view(-1).numpy())
            all_s_preds.extend(s_preds.argmax(dim=-1).cpu().numpy())
            all_s_targets.extend(s_lbls.numpy())
            
            # 只抽取前 100 组进行快速测试
            if idx >= 100:
                break
            
    terrain_acc = accuracy_score(all_t_targets, all_t_preds) * 100
    scene_acc = accuracy_score(all_s_targets, all_s_preds) * 100
    macro_f1 = f1_score(all_t_targets, all_t_preds, average="macro", zero_division=0) * 100
    avg_latency = np.mean(latencies[1:]) # 剔除首次 GPU 加热延迟
    
    print("\n" + "="*55)
    print(" >>> 论文 Table 1 指标一键生成完成 <<<")
    print("="*55)
    print(f"  Terrain Accuracy (%) : {terrain_acc:.2f} %")
    print(f"  Scene Accuracy (%)   : {scene_acc:.2f} %")
    print(f"  Overall F1-score     : {macro_f1:.2f}")
    print(f"  Latency (ms)         : {avg_latency:.2f} ms")
    print("="*55)

if __name__ == "__main__":
    run_evaluation()