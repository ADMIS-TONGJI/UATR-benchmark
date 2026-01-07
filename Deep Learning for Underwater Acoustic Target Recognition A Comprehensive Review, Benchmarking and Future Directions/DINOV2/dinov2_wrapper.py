import torch
import torch.nn as nn


class DINOv2FeatureExtractor(nn.Module):
    """Extracteur de features DINOv2"""
    
    def __init__(
        self,
        model_name: str = "dinov2_vits14",  # vits14, vitb14, vitl14, vitg14
        freeze: bool = True,
        use_registers: bool = False
    ):
        """
        Args:
            model_name: 
                - dinov2_vits14: Small, 384 dim
                - dinov2_vitb14: Base, 768 dim
                - dinov2_vitl14: Large, 1024 dim
                - dinov2_vitg14: Giant, 1536 dim (très lourd)
            freeze: Geler les poids DINOv2
            use_registers: Utiliser version avec registers
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        print(f"📦 Chargement DINOv2: {model_name}")
        
        # Charger modèle depuis torch.hub
        if use_registers:
            model_name_full = f"{model_name}_reg"
        else:
            model_name_full = model_name
        
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name_full)
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            print(f"💡 Essayez: pip install --upgrade torch torchvision")
            raise
        
        # Récupérer dimension
        if "vits14" in model_name:
            self.embed_dim = 384
        elif "vitb14" in model_name:
            self.embed_dim = 768
        elif "vitl14" in model_name:
            self.embed_dim = 1024
        elif "vitg14" in model_name:
            self.embed_dim = 1536
        else:
            raise ValueError(f"Modèle inconnu: {model_name}")
        
        print(f"✅ DINOv2 chargé! Dim: {self.embed_dim}")
        
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print(f"   🔒 Paramètres gelés")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB images
            
        Returns:
            features: [B, embed_dim] CLS token features
        """
        if self.freeze:
            with torch.no_grad():
                # DINOv2 retourne dict avec 'x_norm_clstoken' et 'x_norm_patchtokens'
                out = self.model(x)
                if isinstance(out, dict):
                    features = out['x_norm_clstoken']
                else:
                    features = out
        else:
            out = self.model(x)
            if isinstance(out, dict):
                features = out['x_norm_clstoken']
            else:
                features = out
        
        return features