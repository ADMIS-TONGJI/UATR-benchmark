import torch
import torch.nn as nn
from dinov2_wrapper import DINOv2FeatureExtractor


class DINOv2AudioClassifier(nn.Module):
    """
    Classificateur audio basé sur DINOv2
    """
    
    def __init__(
        self,
        num_classes: int,
        dinov2_model: str = "dinov2_vits14",
        freeze_dinov2: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.dinov2_model = dinov2_model
        
        print(f"\n🏗️  Initialisation DINOv2AudioClassifier")
        print(f"   Classes: {num_classes} | Model: {dinov2_model}")
        print(f"   Freeze: {freeze_dinov2} | Dropout: {dropout}")
        
        # DINOv2 backbone
        self.dinov2 = DINOv2FeatureExtractor(
            model_name=dinov2_model,
            freeze=freeze_dinov2
        )
        
        embed_dim = self.dinov2.embed_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print(f"✅ Modèle créé!\n")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB spectrograms
            
        Returns:
            logits: [B, num_classes]
        """
        # Extract features avec DINOv2
        features = self.dinov2(x)  # [B, embed_dim]
        
        # Classification
        logits = self.classifier(features)  # [B, num_classes]
        
        return logits