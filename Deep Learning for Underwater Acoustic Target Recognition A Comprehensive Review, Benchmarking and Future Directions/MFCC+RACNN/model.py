import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple  


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation注意力模块（Block_B的核心组件）
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Args:
            in_channels: 输入特征图的通道数
            reduction_ratio: 通道压缩比例
        """
        super(SEBlock, self).__init__()
        # 1. Squeeze操作：全局平均池化（压缩空间维度）
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. Excitation操作：全连接层（学习通道权重）
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.elu = nn.ELU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (batch_size, in_channels, width)
        Returns:
            加权后的特征图，形状与输入一致
        """
        batch_size, channels, _ = x.size()
        
        # Squeeze: (batch_size, channels, width) → (batch_size, channels, 1)
        squeeze = self.global_avg_pool(x)
        squeeze = squeeze.view(batch_size, channels)
        
        # Excitation: 压缩→激活→恢复
        excitation = self.fc1(squeeze)
        excitation = self.elu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        # 恢复通道维度：(batch_size, channels) → (batch_size, channels, 1)
        excitation = excitation.view(batch_size, channels, 1)
        
        # 权重加权
        output = x * excitation
        return output


class BlockA(nn.Module):
    """
    RACNN的Block_A模块
    """
    def __init__(self, in_channels: int, out_channels: int, is_first: bool = False):
        """
        Args:
            in_channels: 输入特征图通道数
            out_channels: 输出特征图通道数
            is_first: 是否为第一个Block_A（第一个用MaxPool，后两个用AvgPool）
        """
        super(BlockA, self).__init__()
        self.is_first = is_first
        
        # 1. 初始卷积层（1×3核，ELU激活，BN层）
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,  # 保持宽度不变
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU(inplace=True)
        
        # 2. 池化层（第一个Block_A用MaxPool，后两个用AvgPool）
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4) if is_first else nn.AvgPool1d(kernel_size=4, stride=4)
        
        # 3. 双通道卷积
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels // 2,  
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (batch_size, in_channels, width)
        Returns:
            残差连接后的输出，形状为 (batch_size, out_channels, width//4)
        """
        # 主干路径：卷积→BN→激活→池化
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.elu(residual)
        residual_pooled = self.pool(residual)  # 池化后宽度变为1/4
        
        # 双通道特征提取
        conv2_out = self.conv2(residual_pooled)
        conv3_out = self.conv3(residual_pooled)
        
        # 通道拼接：(batch_size, C//2, W//4) + (batch_size, C//2, W//4) → (batch_size, C, W//4)
        concat_out = torch.cat([conv2_out, conv3_out], dim=1)
        concat_out = self.bn2(concat_out)
        concat_out = self.elu(concat_out)
        
        # 残差连接（池化后的主干 + 双通道特征）
        output = concat_out + residual_pooled
        return output


class BlockB(nn.Module):
    """
    RACNN的Block_B模块
    """
    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: 输入特征图通道数
        """
        super(BlockB, self).__init__()
        
        # 1. 卷积→BN
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.elu = nn.ELU(inplace=True)
        
        # 2. SE注意力模块
        self.se_block = SEBlock(in_channels=in_channels, reduction_ratio=16)
        
        # 3. 输出卷积→BN
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (batch_size, in_channels, width)
        Returns:
            注意力增强后的输出，形状与输入一致
        """
        # 主干路径：卷积→BN→激活→注意力→卷积→BN
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        
        out = self.se_block(out)  # 注意力加权
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out = self.elu(out + residual)
        return out


class RACNN(nn.Module):
    """
    输入：1D-MFCC特征
    输出：多类别概率分布
    """
    def __init__(self, input_dim: int = 2000, num_classes: int = 5):
        """
        Args:
            input_dim: 输入MFCC特征维度（论文中为2000）
            num_classes: 目标分类数
        """
        super(RACNN, self).__init__()
        
        # 1. 输入维度调整：(batch_size, input_dim) → (batch_size, 1, input_dim)
        self.input_adjust = lambda x: x.unsqueeze(1)
        
        # 2. 3个Block_A模块（通道数：1→256→128→64）
        self.block_a1 = BlockA(in_channels=1, out_channels=256, is_first=True)
        self.block_a2 = BlockA(in_channels=256, out_channels=128, is_first=False)
        self.block_a3 = BlockA(in_channels=128, out_channels=64, is_first=False)
        
        # 3. 2个Block_B模块
        self.block_b1 = BlockB(in_channels=64)
        self.block_b2 = BlockB(in_channels=64)
        
        # 4. 全连接层 + Softmax输出
        # 计算池化后的特征宽度：2000 → 2000//4（block_a1）→ //4（block_a2）→ //4（block_a3）= 2000//64 = 31
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 31, out_features=256)  # 64通道 × 31宽度
        self.relu = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(in_features=256, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  
        """
        Args:
            x: 输入MFCC特征，形状为 (batch_size, input_dim)
        Returns:
            logits
            probabilitie
        """
        # 输入调整
        x = self.input_adjust(x)  # (batch_size, 1, 2000)
        
        # 特征提取：Block_A ×3
        x = self.block_a1(x)  # (batch_size, 256, 500) → 2000//4=500
        x = self.block_a2(x)  # (batch_size, 128, 125) → 500//4=125
        x = self.block_a3(x)  # (batch_size, 64, 31) → 125//4=31
        
        # 注意力增强：Block_B ×2
        x = self.block_b1(x)  # (batch_size, 64, 31)
        x = self.block_b2(x)  # (batch_size, 64, 31)
        
        # 分类头
        x = self.flatten(x)  # (batch_size, 64×31=1984)
        x = self.fc(x)       # (batch_size, 256)
        x = self.relu(x)
        logits = self.output_layer(x)  # (batch_size, num_classes)
        probabilities = self.softmax(logits)  # (batch_size, num_classes)
        
        return logits, probabilities


# 模型参数初始化
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)