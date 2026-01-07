import torch
import torch.nn as nn
from typing import Type, Union, List, Optional

class BasicBlock(nn.Module):
    """ResNet 的基础残差块"""
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class MultiHeadAttentionPooling(nn.Module):
    """
    实现论文 Fig. 2 中的多头注意力池化层
    Reference: "Non-local Neural Networks" (Wang et al., 2018)
    """
    def __init__(self, in_channels: int, num_heads: int = 8):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads
        
        # 论文中描述为 Linear_Q, Linear_K, Linear_V
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        
        # Q, K, V
        proj_query = self.query_conv(x).view(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        proj_key = self.key_conv(x).view(B, self.num_heads, C // self.num_heads, H * W)
        proj_value = self.value_conv(x).view(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)

        # Self-attention
        energy = torch.matmul(proj_query, proj_key)
        attention = self.softmax(energy)
        
        out = torch.matmul(attention, proj_value).permute(0, 1, 3, 2).contiguous()
        out = out.view(B, C, H, W)
        
        out = self.gamma * out + x
        return out

class ResNetWithMultiHeadAttention(nn.Module):
    """
    论文中描述的 ResNet-18 + Multi-head Attention 模型
    """
    def __init__(self, block: Type[BasicBlock], num_blocks: List[int], num_classes: int = 9, in_ch: int = 1):
        super(ResNetWithMultiHeadAttention, self).__init__()
        self.in_planes = 64

        # Stem Layer (对应论文 Fig. 2 的 7x7 Conv 和 3x3 Max pooling)
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four Residual Layers (对应论文中的4个 Basic block 堆叠)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Attention Pooling Layer
        # ResNet-18 最后一个 residual layer 输出 512 个通道
        self.attention = MultiHeadAttentionPooling(512, num_heads=8)
        
        # Classifier Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.attention(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        logits = self.fc(out)
        return logits

def create_model(num_classes: int) -> nn.Module:
    """
    工厂函数，创建 ResNet-18 模型
    """
    return ResNetWithMultiHeadAttention(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)