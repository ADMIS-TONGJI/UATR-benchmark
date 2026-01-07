import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    ResNet基本残差块：Conv-BN-ReLU-Conv-BN + 跳连
    (与论文 Table 1  一致)
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class AttentionPooling(nn.Module):
    """
    注意力池化：将空间特征映射聚合为单向量。
    组合了全局平均池化(GAP)与多头自注意力(MHSA)的查询聚合。
    (与论文 Sec 3.2 [cite: 204] 一致)
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, 1, channels))  # [1, 1, C]
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=False)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        gap = F.adaptive_avg_pool2d(x, output_size=1).view(b, c)  # [B, C]

        tokens = x.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        tokens = tokens.transpose(0, 1)  # [HW, B, C]
        query = self.query.expand(-1, b, -1)  # [1, B, C]
        attn_out, _ = self.mha(query, tokens, tokens)  # [1, B, C]
        attn_out = attn_out.squeeze(0)  # [B, C]
        attn_out = self.out_proj(attn_out)

        pooled = gap + attn_out  # 融合GAP与注意力聚合
        return pooled  # [B, C]


class ResNetAPBackbone(nn.Module):
    """
    前端主干：ResNet + Attention Pooling
    (与论文 Table 1  结构一致)
    Conv7x7(s=2) → BN → ReLU → MaxPool3x3(s=2) →
    [64]x2 → [128]x2 → [256]x2 → [512]x2 → AttentionPooling → [B, 512]
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )

        self.attn_pool = AttentionPooling(channels=512, num_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feats = self.attn_pool(x)  # [B, 512]
        return feats


def _make_expert_head(in_dim: int, hidden_dim: int, num_classes: int) -> nn.Sequential:
    """
    创建专家头 (与论文 Table 1  一致)
    Linear(512→128)→BN→ReLU→Linear(128→C)
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, num_classes)
    )


def _balance_loss_from_assign_and_probs(assign_idx: torch.Tensor, gate_probs: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    论文式负载均衡 (Eq. 5)[cite: 294]: m * sum_j ef_j * ep_j
    - ef_j: 当前batch路由到专家j的样本占比（基于硬路由argmax）
    - ep_j: 当前batch对专家j的平均路由概率
    """
    b = gate_probs.size(0)
    # ef_j: [K]
    one_hot = F.one_hot(assign_idx, num_classes=num_experts).float()  # [B, K]
    ef = one_hot.mean(dim=0)  # [K]
    # ep_j: [K]
    ep = gate_probs.mean(dim=0)  # [K]
    m = float(num_experts)
    # 论文 Eq. 5 [cite: 294]
    loss = m * (ef * ep).sum()
    return loss


class CMoE(nn.Module):
    """
    CMoE/RCMoE（卷积版）：
    - 前端主干：ResNet + Attention Pooling（输出[B, 512]）
    - 路由层：Linear(512 → K)，softmax概率，硬路由argmax (top-1) 派发 [cite: 182, 220]
    - 专家层：K个MLP（Linear(512→128)→BN→ReLU→Linear(128→C)）
    - 可选残差专家（RCMoE）：额外非门控专家输出与专家输出相加 [cite: 283]
    返回：(logits, probabilities, aux_losses)
    """

    def __init__(self,
                 num_classes: int = 9,
                 num_experts: int = 4,
                 gate_temperature: float = 1.0,
                 residual: bool = False,
                 residual_scale: float = 0.5) -> None:
        """
        初始化 CMoE/RCMoE
        参数:
            num_classes (int): 类别数
            num_experts (int): 专家数量 (K)
            gate_temperature (float): softmax 温度系数
            residual (bool): 是否启用 RCMoE (添加残差专家)
            residual_scale (float): 残差专家缩放系数
        """
        super().__init__()
        assert num_experts >= 1 and num_classes > 0
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.gate_temperature = float(max(1e-6, gate_temperature))
        self.use_residual = residual
        self.residual_scale = float(residual_scale)
        
        # 论文 Table 1  中的专家头隐藏维度
        self.expert_hidden_dim = 128
        # 论文 Table 1  中的主干输出维度
        self.backbone_dim = 512

        # 前端主干
        self.backbone = ResNetAPBackbone()

        # 路由层：Linear(512 → K) (Table 1) 
        self.routing = nn.Linear(self.backbone_dim, num_experts)

        # K个专家头 (Table 1) 
        self.experts = nn.ModuleList([
            _make_expert_head(self.backbone_dim, self.expert_hidden_dim, num_classes) 
            for _ in range(num_experts)
        ])

        # 残差专家（RCMoE）(Sec 3.3) [cite: 283]
        if self.use_residual:
            self.residual_expert = _make_expert_head(self.backbone_dim, self.expert_hidden_dim, num_classes)
        else:
            self.residual_expert = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # x: [B, 1, H, W]
        feats = self.backbone(x)  # [B, 512]

        gate_logits = self.routing(feats)  # [B, K]
        gate_probs = F.softmax(gate_logits / self.gate_temperature, dim=-1)  # [B, K]

        # 硬路由 (Top-1, 论文 Sec 3.3 [cite: 220] 和 Alg 1 [cite: 274])
        assign_idx = torch.argmax(gate_probs, dim=-1)  # [B]

        # 高效实现：计算所有专家logits后按索引选择
        # [K, B, C] -> [B, K, C]
        expert_logits_all = torch.stack([expert(feats) for expert in self.experts], dim=1) 
        
        # [B] -> [B, 1, C]
        gather_idx = assign_idx.view(-1, 1, 1).expand(-1, 1, self.num_classes)
        # 选出被指派的专家的 logits
        logits = torch.gather(expert_logits_all, dim=1, index=gather_idx).squeeze(1)  # [B, C]

        # 残差路径（RCMoE）(Sec 3.3) [cite: 285]
        if self.residual_expert is not None:
            logits = logits + self.residual_scale * self.residual_expert(feats)

        probs = F.softmax(logits, dim=-1)

        # 论文式负载均衡 (Eq. 5) [cite: 294, 309]
        lb_loss = _balance_loss_from_assign_and_probs(assign_idx, gate_probs, self.num_experts)
        aux_losses: Dict[str, torch.Tensor] = {"load_balance": lb_loss}
        
        return logits, probs, aux_losses


def init_weights(module: nn.Module) -> None:
    """
    权重初始化：
    - Linear: Xavier uniform + bias零
    - LayerNorm: weight=1, bias=0
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)