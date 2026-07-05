import argparse
import os
import time

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets.rellis_dataset import RELLIS3DVideoDataset
from models.offroad_net import DecoupledSpatiotemporalNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model and generate table metrics")
    parser.add_argument('--config', type=str, default=None, help='path to yaml config file')
    parser.add_argument('--root', type=str, default='dataset/rellis', help='dataset root')
    parser.add_argument('--sequence', type=str, default='00000', help='sequence id')
    parser.add_argument('--num-frames', type=int, default=8, help='number of frames per sample')
    parser.add_argument('--num-views', type=int, default=3, help='number of views per sample')
    parser.add_argument('--batch-size', type=int, default=1, help='evaluation batch size')
    parser.add_argument('--num-samples', type=int, default=60, help='max number of evaluation batches')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_offroad_net.pth', help='trained model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='device to run eval on')
    parser.add_argument('--num-workers', type=int, default=2, help='data loader workers')
    parser.add_argument('--disable-tsm', action='store_true', help='disable temporal modeling during evaluation')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    return args


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f">> 警告：未找到模型权重 {checkpoint_path}，将使用随机初始化权重进行评估。")
        return model

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, strict=False)
        print(f">> 成功加载模型权重：{checkpoint_path}")
    except RuntimeError as exc:
        print('>> 警告：加载权重时出现不匹配，尝试部分加载。\n' + str(exc))
        model.load_state_dict(state_dict, strict=False)

    return model


def evaluate_config(model, dataloader, device, view_mask=None, disable_tsm=False, num_samples=60):
    all_t_preds, all_t_targets = [], []
    all_s_preds, all_s_targets = [], []
    latencies = []

    model.eval()
    with torch.no_grad():
        for idx, (imgs, t_lbls, s_lbls) in enumerate(dataloader, start=1):
            imgs = imgs.to(device)
            if view_mask is not None:
                masked_imgs = torch.zeros_like(imgs)
                for v_idx in view_mask:
                    masked_imgs[:, v_idx, ...] = imgs[:, v_idx, ...]
                imgs = masked_imgs

            start_t = time.perf_counter()
            if disable_tsm:
                imgs_static = imgs[:, :, :1, ...].repeat(1, 1, imgs.shape[2], 1, 1, 1)
                t_preds, s_preds = model(imgs_static)
            else:
                t_preds, s_preds = model(imgs)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)

            if view_mask is not None:
                t_preds_eval = t_preds[:, view_mask, :].reshape(-1, t_preds.shape[-1])
                t_lbls_eval = t_lbls[:, view_mask].reshape(-1)
            else:
                t_preds_eval = t_preds.reshape(-1, t_preds.shape[-1])
                t_lbls_eval = t_lbls.reshape(-1)

            all_t_preds.extend(t_preds_eval.argmax(dim=-1).cpu().numpy())
            all_t_targets.extend(t_lbls_eval.cpu().numpy())
            all_s_preds.extend(s_preds.argmax(dim=-1).cpu().numpy())
            all_s_targets.extend(s_lbls.cpu().numpy())

            if idx >= num_samples:
                break

    if len(latencies) == 0:
        return 0.0, 0.0, 0.0, 0.0

    t_acc = accuracy_score(all_t_targets, all_t_preds) * 100.0
    s_acc = accuracy_score(all_s_targets, all_s_preds) * 100.0
    f1 = f1_score(all_t_targets, all_t_preds, average='macro', zero_division=0) * 100.0
    lat = float(np.mean(latencies[1:])) if len(latencies) > 1 else float(np.mean(latencies))
    return t_acc, s_acc, f1, lat


def print_table_heading(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    print(f"正在准备评估表格指标 (设备: {device})")
    print(f"配置: root={args.root}, seq={args.sequence}, frames={args.num_frames}, batch={args.batch_size}, checkpoint={args.checkpoint}\n")

    dataset = RELLIS3DVideoDataset(root_dir=args.root, sequence=args.sequence, num_frames=args.num_frames, num_views=args.num_views)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset appears empty: {args.root}/{args.sequence}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = DecoupledSpatiotemporalNet(num_terrain_classes=4, num_scene_classes=2, num_frames=args.num_frames, num_views=args.num_views).to(device)
    model = load_checkpoint(model, args.checkpoint, device)

    total_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params_m:.2f} M | 样本数: {len(dataset)} | 评估批次上限: {args.num_samples}\n")

    print_table_heading('表 1 (Table 1): 主力方法评估')
    t_acc, s_acc, f1, lat = evaluate_config(model, dataloader, device, num_samples=args.num_samples, disable_tsm=args.disable_tsm)
    print(f"Method       : Ours (Multi-camera Video)")
    print(f"Terrain Acc  : {t_acc:.2f} %")
    print(f"Scene Acc    : {s_acc:.2f} %")
    print(f"Overall F1   : {f1:.2f} %")
    print(f"Latency      : {lat:.2f} ms")

    print_table_heading('表 2 (Table 2): 时序建模消融')
    t_acc_cnn, s_acc_cnn, f1_cnn, _ = evaluate_config(model, dataloader, device, disable_tsm=True, num_samples=args.num_samples)
    print(f"CNN-only (no TSM) -> Terrain Acc: {t_acc_cnn:.2f}% | Scene Acc: {s_acc_cnn:.2f}% | F1: {f1_cnn:.2f} %")
    print(f"Ours (with TSM)   -> Terrain Acc: {t_acc:.2f}% | Scene Acc: {s_acc:.2f}% | F1: {f1:.2f} %")

    print_table_heading('表 3 (Table 3): 多视角消融')
    view_configs = [
        ('Front', [0]),
        ('Left', [1]),
        ('Right', [2]),
        ('Front+Left', [0, 1]),
        ('Front+Right', [0, 2]),
        ('Front+Left+Right', [0, 1, 2]),
    ]
    for name, mask in view_configs:
        ta, sa, f1_v, _ = evaluate_config(model, dataloader, device, view_mask=mask, num_samples=args.num_samples)
        print(f"{name:14s} | Terrain: {ta:5.2f}% | Scene: {sa:5.2f}% | F1: {f1_v:5.2f}%")

    fps = 1000.0 / lat if lat > 0 else 0.0
    print_table_heading('表 4 (Table 4): 运行效率与资源统计')
    print(f"Params (M)   : {total_params_m:.2f}")
    print(f"FPS          : {fps:.1f}")
    print(f"Latency (ms) : {lat:.2f}")

    print('\n评估完成。')


if __name__ == '__main__':
    main()
