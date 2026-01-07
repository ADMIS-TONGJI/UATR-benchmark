import torch
import torch.nn as nn


class UATC_DenseNet(nn.Module):
    def __init__(self, num_classes=12):
        super(UATC_DenseNet, self).__init__()
        
        # -------------------------- 1. 输入预处理层（论文 1-24/25 节）--------------------------
        # 输入形状：(batch, 1, 4096) → 1通道声学信号，4096时间样本
        self.batch_norm_input = nn.BatchNorm1d(num_features=1, eps=1e-5)
        # 跳接1的通道统一：将原始输入1通道→32通道（匹配卷积块输出通道）
        self.skip_conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1, stride=1)

        # -------------------------- 2. 卷积块+跳接连接（论文 1-33/35 节，修正维度逻辑）--------------------------
        # 核心修正：跳接源为“前序模块输出”，而非原始输入；通过1×1卷积统一通道，MaxPool统一长度
        
        # 2.1 第一个卷积块（Conv-Block1）+ 跳接1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)  # padding=3：确保卷积后长度不变
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # padding=1：长度//2（4096→2048→1024→...）
        self.elu1 = nn.ELU()
        # 跳接1：输入归一化后→1×1卷积（通道1→32）→MaxPool（长度4096→2048）
        self.skip_maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(num_features=32+32, eps=1e-5)  # 拼接后64通道（32 conv1+32 skip1）

        # 2.2 第二个卷积块（Conv-Block2）+ 跳接2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)  # 输入64通道（来自上一步拼接）
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.elu2 = nn.ELU()
        # 跳接2：前序模块输出（64通道，2048长度）→MaxPool（2048→1024）
        self.skip_maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(num_features=32+64, eps=1e-5)  # 拼接后96通道（32 conv2+64 skip2）

        # 2.3 第三个卷积块（Conv-Block3）+ 跳接3
        self.conv3 = nn.Conv1d(in_channels=96, out_channels=32, kernel_size=7, stride=1, padding=3)  # 输入96通道
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.elu3 = nn.ELU()
        # 跳接3：前序模块输出（96通道，1024长度）→MaxPool（1024→512）
        self.skip_maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(num_features=32+96, eps=1e-5)  # 拼接后128通道（32 conv3+96 skip3）

        # -------------------------- 3. 输出块（论文 1-36/37 节，修正输入维度）--------------------------
        self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8, padding=0)  # 1×8平均池化
        self.elu_out = nn.ELU()
        # 全连接层输入维度计算：128通道 × 64长度（512经过1×8池化后）= 8192
        self.fc = nn.Linear(in_features=128 * 64, out_features=num_classes)

    def forward(self, x):
        """修正后各层尺寸传递：确保拼接时长度一致"""
        # 1. 输入预处理：(batch,1,4096) → (batch,1,4096)
        x_norm = self.batch_norm_input(x)

        # 2. 第一个卷积块+跳接1：拼接后尺寸 (batch,64,2048)
        # 主干流
        conv1_out = self.conv1(x_norm)  # (batch,32,4096) → padding=3确保长度不变
        maxpool1_out = self.maxpool1(conv1_out)  # (batch,32,2048) → 长度//2
        elu1_out = self.elu1(maxpool1_out)
        # 跳接流
        skip1_conv = self.skip_conv1(x_norm)  # (batch,32,4096) → 通道1→32
        skip1_out = self.skip_maxpool1(skip1_conv)  # (batch,32,2048) → 长度//2（与主干流一致）
        # 拼接（通道维度dim=1）
        concat1_out = torch.cat([elu1_out, skip1_out], dim=1)  # (batch,32+32,2048) = (batch,64,2048)
        bn1_out = self.batch_norm1(concat1_out)  # (batch,64,2048)

        # 3. 第二个卷积块+跳接2：拼接后尺寸 (batch,96,1024)
        # 主干流
        conv2_out = self.conv2(bn1_out)  # (batch,32,2048)
        maxpool2_out = self.maxpool2(conv2_out)  # (batch,32,1024)
        elu2_out = self.elu2(maxpool2_out)
        # 跳接流（前序模块输出→MaxPool）
        skip2_out = self.skip_maxpool2(bn1_out)  # (batch,64,1024) → 长度//2（与主干流一致）
        # 拼接
        concat2_out = torch.cat([elu2_out, skip2_out], dim=1)  # (batch,32+64,1024) = (batch,96,1024)
        bn2_out = self.batch_norm2(concat2_out)  # (batch,96,1024)

        # 4. 第三个卷积块+跳接3：拼接后尺寸 (batch,128,512)
        # 主干流
        conv3_out = self.conv3(bn2_out)  # (batch,32,1024)
        maxpool3_out = self.maxpool3(conv3_out)  # (batch,32,512)
        elu3_out = self.elu3(maxpool3_out)
        # 跳接流（前序模块输出→MaxPool）
        skip3_out = self.skip_maxpool3(bn2_out)  # (batch,96,512) → 长度//2（与主干流一致）
        # 拼接
        concat3_out = torch.cat([elu3_out, skip3_out], dim=1)  # (batch,32+96,512) = (batch,128,512)
        bn3_out = self.batch_norm3(concat3_out)  # (batch,128,512)

        # 5. 输出块：(batch,128,512) → (batch,num_classes)
        avgpool_out = self.avgpool(bn3_out)  # (batch,128,512//8) = (batch,128,64)
        elu_out = self.elu_out(avgpool_out)
        flatten_out = torch.flatten(elu_out, start_dim=1)  # (batch,128×64) = (batch,8192)
        logits = self.fc(flatten_out)  # (batch,num_classes)
        probabilities = nn.functional.softmax(logits, dim=1)  # (batch,num_classes)

        return logits, probabilities