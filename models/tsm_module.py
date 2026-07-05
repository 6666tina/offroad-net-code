import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=8):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        
        fold = c // fold_div
        out = torch.zeros_like(x)
        # 1/8 通道沿时间轴前移（交互历史信息）
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # 1/8 通道沿时间轴后移（交互未来信息）
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        # 剩余 3/4 通道保持不变
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        
        return out.view(nt, c, h, w)