import argparse
import os
import time

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets.rellis_dataset import RELLIS3DVideoDataset
from models.offroad_net import DecoupledSpatiotemporalNet


def parse_args():
    parser = argparse.ArgumentParser(description="Configurable training script for offroad net")
    parser.add_argument('--config', type=str, default=None, help='path to yaml config file')
    parser.add_argument('--root', type=str, default='dataset/rellis', help='dataset root')
    parser.add_argument('--sequence', type=str, default='00000', help='sequence id')
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--num-views', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--max-samples', type=int, default=0, help='limit dataset samples (0=no limit)')
    parser.add_argument('--use-uncertainty', action='store_true', help='use learned uncertainty weighting for multi-task loss')
    parser.add_argument('--log-interval', type=int, default=50, help='print loss every N batches')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    return args


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, use_uncertainty=False):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        if self.use_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, loss_t, loss_s):
        if not self.use_uncertainty:
            return loss_t + loss_s
        precision_t = torch.exp(-self.log_vars[0])
        precision_s = torch.exp(-self.log_vars[1])
        return precision_t * loss_t + self.log_vars[0] + precision_s * loss_s + self.log_vars[1]


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"当前训练运行设备: {device}")
    print(f"训练配置: {args}")

    dataset = RELLIS3DVideoDataset(root_dir=args.root, sequence=args.sequence, num_frames=args.num_frames, num_views=args.num_views)
    if args.max_samples and args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset appears empty: {args.root}/{args.sequence}")

    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    if train_len == 0:
        train_len = len(dataset) - 1
        val_len = 1

    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DecoupledSpatiotemporalNet(num_terrain_classes=4, num_scene_classes=2, num_frames=args.num_frames, num_views=args.num_views).to(device)
    criterion = nn.CrossEntropyLoss()
    multitask_wrapper = MultiTaskLossWrapper(use_uncertainty=args.use_uncertainty).to(device)
    params = list(model.parameters()) + (list(multitask_wrapper.parameters()) if args.use_uncertainty else [])
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_t_acc = 0.0
        running_s_acc = 0.0
        start_time = time.time()

        for batch_idx, (imgs, t_lbls, s_lbls) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)
            t_lbls = t_lbls.to(device)
            s_lbls = s_lbls.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                t_preds, s_preds = model(imgs)
                loss_t = criterion(t_preds.view(-1, 4), t_lbls.view(-1))
                loss_s = criterion(s_preds, s_lbls)
                loss = multitask_wrapper(loss_t, loss_s)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            running_t_acc += (t_preds.argmax(dim=-1).eq(t_lbls).sum().item() / (imgs.size(0) * args.num_views)) * imgs.size(0)
            running_s_acc += (s_preds.argmax(dim=-1).eq(s_lbls).sum().item() / imgs.size(0)) * imgs.size(0)

            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_time = time.time() - start_time

        model.eval()
        val_loss = 0.0
        val_t_acc = 0.0
        val_s_acc = 0.0
        with torch.no_grad():
            for imgs, t_lbls, s_lbls in val_loader:
                imgs = imgs.to(device)
                t_lbls = t_lbls.to(device)
                s_lbls = s_lbls.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    t_preds, s_preds = model(imgs)
                    loss_t = criterion(t_preds.view(-1, 4), t_lbls.view(-1))
                    loss_s = criterion(s_preds, s_lbls)
                    loss = multitask_wrapper(loss_t, loss_s)

                val_loss += loss.item() * imgs.size(0)
                val_t_acc += (t_preds.argmax(dim=-1).eq(t_lbls).sum().item() / (imgs.size(0) * args.num_views)) * imgs.size(0)
                val_s_acc += (s_preds.argmax(dim=-1).eq(s_lbls).sum().item() / imgs.size(0)) * imgs.size(0)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] Time: {epoch_time:.1f}s | "
            f"TrainLoss: {running_loss/train_len:.4f} | TrainTAcc: {running_t_acc/train_len:.4f} | TrainSAcc: {running_s_acc/train_len:.4f} | "
            f"ValLoss: {val_loss/val_len:.4f} | ValTAcc: {val_t_acc/val_len:.4f} | ValSAcc: {val_s_acc/val_len:.4f}"
        )

        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'args': vars(args),
            'multitask_state': multitask_wrapper.state_dict() if args.use_uncertainty else None,
        }, checkpoint_path)

    final_path = os.path.join(args.save_dir, 'checkpoint_offroad_net.pth')
    torch.save(model.state_dict(), final_path)
    print(f"模型训练已完成，权重保存至 {final_path}")


if __name__ == '__main__':
    main()
