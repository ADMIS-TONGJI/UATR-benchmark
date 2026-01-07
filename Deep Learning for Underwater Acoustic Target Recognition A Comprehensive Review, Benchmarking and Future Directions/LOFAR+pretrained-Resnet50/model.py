import torch
import torch.nn as nn
from torchvision.models import resnet50

class LOFARResNet50(nn.Module):
    def __init__(self, num_classes, local_model_path):
        super(LOFARResNet50, self).__init__()
        
        # 初始化ResNet50模型
        self.resnet = resnet50(pretrained=False)  
        
        # 加载本地ImageNet预训练权重
        try:
            pretrained_weights = torch.load(local_model_path)
            self.resnet.load_state_dict(pretrained_weights)
            print(f"成功加载本地预训练权重: {local_model_path}")
        except Exception as e:
            print(f"加载权重失败: {e}")
            raise
        
        # 冻结特征提取层
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 替换分类头
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=num_classes)
        )
    
    def forward(self, x):
        # x的输入形状: (batchsize, mel_bins, time_steps)
        
        # 扩展为 (batchsize, 1, mel_bins, time_steps)
        x = x.unsqueeze(1)
        
        x = x.repeat(1, 3, 1, 1)
        
        # 通过ResNet50获取logits
        logits = self.resnet(x)
        
        # 计算概率
        probabilities = nn.functional.softmax(logits, dim=1)
        
        return logits, probabilities