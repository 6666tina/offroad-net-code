import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.rellis_dataset import RELLIS3DVideoDataset
from models.offroad_net import DecoupledSpatiotemporalNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前训练运行设备: {device}")
    
    # 初始化数据集
    dataset = RELLIS3DVideoDataset(root_dir="dataset/rellis", sequence="00000", num_frames=8)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # 初始化模型与损失函数
    model = DecoupledSpatiotemporalNet(num_terrain_classes=4, num_scene_classes=2).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 多任务损失权重平衡系数
    lambda_t, lambda_s = 1.0, 1.0
    
    model.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (imgs, t_lbls, s_lbls) in enumerate(dataloader):
            imgs, t_lbls, s_lbls = imgs.to(device), t_lbls.to(device), s_lbls.to(device)
            
            optimizer.zero_grad()
            t_preds, s_preds = model(imgs)
            
            # 计算局部视角地形损失 L_t 与全局场景损失 L_s
            loss_t = criterion_ce(t_preds.view(-1, 4), t_lbls.view(-1))
            loss_s = criterion_ce(s_preds, s_lbls)
            
            loss = lambda_t * loss_t + lambda_s * loss_s
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        print(f"===> Epoch [{epoch+1}/{epochs}] 完成 | 平均多任务损失: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "checkpoint_offroad_net.pth")
    print("模型训练已完成，权重成功保存至 checkpoint_offroad_net.pth！")

if __name__ == "__main__":
    main()