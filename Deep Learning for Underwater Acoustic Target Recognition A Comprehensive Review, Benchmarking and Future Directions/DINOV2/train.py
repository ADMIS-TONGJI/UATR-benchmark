import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataset import UnderwaterAudioDataset
from feature_extraction import CollateDINOv2
from model import DINOv2AudioClassifier


# ============ Hyperparamètres ============
HPARAMS = {
    "device_id": "7",
    "dataset_root": "/remote-home/share/dmb_nas2/Diallo/keshe/reproduction_datasets/deepship_5s_id",
    "checkpoint_dir": "/remote-home/Diallo/checkpointsDINOv2",
    
    # DINOv2
    "dinov2_model": "dinov2_vits14",  # vits14, vitb14, vitl14
    "freeze_dinov2": True,
    "img_size": 224,
    
    # Training
    "num_classes": 4,
    "batch_size": 32,
    "num_epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 25,
    
    # Classifier
    "hidden_dim": 256,
    "dropout": 0.5,
    "label_smoothing": 0.1,
}


# ============ Arguments ============
def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv2 Audio Classifier")
    parser.add_argument("--dataset_name", type=str, default="deepship_5s_id")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vits14", 
                       choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    parser.add_argument("--no_freeze", action="store_true", help="Fine-tune DINOv2")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=224, choices=[224, 518])
    return parser.parse_args()


# ============ Device ============
def get_device(device_id):
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"🔧 GPU: {device_id}")
    else:
        device = torch.device("cpu")
        print("🔧 CPU")
    return device


# ============ Data Loading ============
def load_data(dataset_root, batch_size, img_size, dataset_name):
    print(f"\n📊 Chargement données...")
    
    train_ds = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=False
    )
    val_ds = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="test",
        is_validation=True
    )
    
    # Collate
    collate = CollateDINOv2(img_size=img_size, dataset_name=dataset_name)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True
    )
    
    class_map = train_ds.get_class_mapping()
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"   Classes: {len(class_map)} → {list(class_map.keys())}")
    
    return train_loader, val_loader, class_map


# ============ Training ============
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses, correct, total = [], 0, 0
    
    pbar = tqdm(loader, desc=f"🔥 Epoch {epoch}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return sum(losses) / len(losses), correct / total


# ============ Validation ============
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, correct, total = [], 0, 0
    
    for imgs, labels in tqdm(loader, desc="✅ Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return sum(losses) / len(losses), correct / total


# ============ Main ============
def main():
    args = parse_args()
    
    # Update params
    if args.batch_size:
        HPARAMS["batch_size"] = args.batch_size
    if args.epochs:
        HPARAMS["num_epochs"] = args.epochs
    if args.lr:
        HPARAMS["lr"] = args.lr
    
    HPARAMS["dinov2_model"] = args.dinov2_model
    HPARAMS["freeze_dinov2"] = not args.no_freeze
    HPARAMS["img_size"] = args.img_size
    
    print("\n" + "="*70)
    print("🚀 TRAINING DINOV2 AUDIO CLASSIFICATION")
    print("="*70)
    print(f"📁 Dataset: {args.dataset_name}")
    print(f"🎨 DINOv2: {HPARAMS['dinov2_model']}")
    print(f"🔒 Freeze: {HPARAMS['freeze_dinov2']}")
    print(f"🖼️  Image size: {HPARAMS['img_size']}x{HPARAMS['img_size']}")
    print(f"📦 Batch: {HPARAMS['batch_size']} | Epochs: {HPARAMS['num_epochs']}")
    print(f"📈 LR: {HPARAMS['lr']} | WD: {HPARAMS['weight_decay']}")
    print("="*70)
    
    device = get_device(HPARAMS["device_id"])
    
    # Load data
    train_loader, val_loader, class_map = load_data(
        HPARAMS["dataset_root"],
        HPARAMS["batch_size"],
        HPARAMS["img_size"],
        args.dataset_name
    )
    HPARAMS["num_classes"] = len(class_map)
    
    # Model
    print(f"\n🏗️  Création modèle...")
    model = DINOv2AudioClassifier(
        num_classes=HPARAMS["num_classes"],
        dinov2_model=HPARAMS["dinov2_model"],
        freeze_dinov2=HPARAMS["freeze_dinov2"],
        dropout=HPARAMS["dropout"],
        hidden_dim=HPARAMS["hidden_dim"]
    ).to(device)
    
    # Params count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total:,}")
    print(f"   Trainable: {trainable:,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=HPARAMS["label_smoothing"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"]
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Save dir
    save_dir = os.path.join(
        HPARAMS["checkpoint_dir"], 
        args.dataset_name,
        HPARAMS["dinov2_model"]
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    
    print(f"\n💾 Checkpoint dir: {save_dir}\n")
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, HPARAMS["num_epochs"] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{HPARAMS['num_epochs']}")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        print(f"\n📊 Results:")
        print(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"   LR    → {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "hparams": HPARAMS
            }, save_path)
            
            print(f"\n   ✅ Best model saved! Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏸️  No improvement ({patience_counter}/{HPARAMS['patience']})")
            
            if patience_counter >= HPARAMS["patience"]:
                print(f"\n⏹️  Early stopping!")
                break
    
    print(f"\n{'='*70}")
    print(f"🎉 Training complete!")
    print(f"🏆 Best Val Acc: {best_val_acc:.4f}")
    print(f"💾 Model saved: {save_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()