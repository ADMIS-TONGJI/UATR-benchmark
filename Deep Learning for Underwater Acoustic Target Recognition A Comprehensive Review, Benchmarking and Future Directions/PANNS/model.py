import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Cnn14_16k


class HydroPANNsCnn14_16k(nn.Module):
    def __init__(self, num_classes, local_model_path, freeze_base=True):
        super(HydroPANNsCnn14_16k, self).__init__()

        self.backbone = Cnn14_16k(
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527
        )

        try:
            checkpoint = torch.load(local_model_path, map_location='cpu')

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"成功加载本地 PANNs 预训练权重: {local_model_path}")

        except Exception as e:
            print(f"加载权重失败: {e}")
            raise

        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        output_dict = self.backbone(x, mixup_lambda=None)
        embedding = output_dict['embedding']   # (batch_size, 2048)

        logits = self.classifier(embedding)
        probabilities = F.softmax(logits, dim=1)

        return logits, probabilities