import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import os
from torch.utils.data import DataLoader
from dataset import UnderwaterAudioDataset
from stft import collate_fn, SAMPLE_RATE


class ConvolutionalAttentionModule(nn.Module):
    """卷积注意力模块（CAM+SAM）：AMNet核心注意力组件"""
    def __init__(self, in_channels: int):
        super(ConvolutionalAttentionModule, self).__init__()
        # 通道注意力（CAM）
        self.global_avg_pool_cam = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool_cam = nn.AdaptiveMaxPool2d(1)
        self.mlp_cam = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 空间注意力（SAM）
        self.conv_sam = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_sam = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力（CAM）
        avg_pool_cam = self.global_avg_pool_cam(x)
        max_pool_cam = self.global_max_pool_cam(x)
        avg_mlp = self.mlp_cam(avg_pool_cam)
        max_mlp = self.mlp_cam(max_pool_cam)
        cam_weight = self.sigmoid(avg_mlp + max_mlp)
        x_cam = x * cam_weight
        
        # 空间注意力（SAM）
        avg_pool_sam = torch.mean(x_cam, dim=1, keepdim=True)
        max_pool_sam = torch.max(x_cam, dim=1, keepdim=True)[0]
        sam_concat = torch.cat([avg_pool_sam, max_pool_sam], dim=1)
        sam_conv = self.conv_sam(sam_concat)
        sam_conv = self.bn_sam(sam_conv)
        sam_weight = self.sigmoid(sam_conv)
        x_sam = x_cam * sam_weight
        
        return x_sam


class MultiBranchBlock(nn.Module):
    """多分支骨干块：AMNet核心特征提取组件"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(MultiBranchBlock, self).__init__()
        # 1×1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        # 3×3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        # 恒等映射
        self.identity = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch_concat = torch.cat([branch1_out, branch2_out], dim=1)
        residual = self.identity(x)
        out = self.relu(branch_concat + residual)
        return out


class AMNet(nn.Module):
    def __init__(self, model_version: str = "T", num_classes: int = 5):
        super(AMNet, self).__init__()
        self.model_version = model_version.upper()
        self.num_classes = num_classes
        
        # 模型阶段配置（遵循文中表1）
        if self.model_version == "N":
            self.stage_config = [(64,1,2), (64,2,2), (128,3,2), (0,0,0)]
        elif self.model_version == "T":
            self.stage_config = [(48,1,2), (48,2,2), (96,4,2), (192,4,2)]
        elif self.model_version == "S":
            self.stage_config = [(48,1,2), (48,2,2), (96,4,2), (192,14,2)]
        else:
            raise ValueError(f"AMNet版本仅支持'N'/'T'/'S'，当前输入：{model_version}")
        
        # 输入层：适配STFT 1通道输入
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, self.stage_config[0][0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_config[0][0]),
            nn.ReLU(inplace=True)
        )
        
        # 构建Stage和注意力模块
        self.stages = nn.ModuleList()
        self.attentions = nn.ModuleList()
        in_channels = self.stage_config[0][0]
        
        for out_channels, num_blocks, stride in self.stage_config:
            if out_channels == 0:
                continue
            stage_blocks = nn.Sequential()
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                stage_blocks.add_module(
                    f"block_{len(self.stages)+1}_{block_idx+1}",
                    MultiBranchBlock(in_channels, out_channels, block_stride)
                )
                in_channels = out_channels
            self.stages.append(stage_blocks)
            self.attentions.append(ConvolutionalAttentionModule(in_channels))
        
        # 分类头
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入层调整
        x = self.input_conv(x)
        # Stage特征提取 + 注意力加权
        for stage, attention in zip(self.stages, self.attentions):
            x = stage(x)
            x = attention(x)
        # 分类头前向
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x)
        probabilities = self.softmax(logits)
        return logits, probabilities


def init_weights(m: nn.Module) -> None:
    """模型参数初始化"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)