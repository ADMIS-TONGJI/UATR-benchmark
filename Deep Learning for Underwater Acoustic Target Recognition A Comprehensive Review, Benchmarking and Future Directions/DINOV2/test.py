import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pandas as pd
import seaborn as sns
import logging
from datetime import datetime
from tqdm import tqdm

from dataset import UnderwaterAudioDataset
from feature_extraction import CollateDINOv2
from model import DINOv2AudioClassifier


# ============ Paramètres ============
HPARAMS = {
    "device_id": "7",
    "dataset_root": "/remote-home/share/dmb_nas2/Diallo/keshe/reproduction_datasets/deepship_5s_id",
    "checkpoint_path": "/remote-home/Diallo/checkpointsDINOv2/deepship_5s_id/dinov2_vits14/best_model.pth",
    "result_dir": "/remote-home/Diallo/DINOv2/test_results",
    "dinov2_model": "dinov2_vits14",
    "img_size": 224,
    "batch_size": 32,
    "num_classes": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test DINOv2 Audio Classifier")
    parser.add_argument("--dataset_name", type=str, default="deepship_5s_id")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vits14")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def setup_logging(result_dir):
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def get_device(device_id):
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"✅ GPU: {device_id}")
        return device
    else:
        print("⚠️ CPU")
        return torch.device("cpu")


def load_test_data(dataset_root, batch_size, img_size, dataset_name, logger):
    logger.info(f"📦 Chargement test dataset...")
    
    test_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="test",
        is_validation=True
    )
    
    collate_fn = CollateDINOv2(img_size=img_size, dataset_name=dataset_name)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    class_to_idx = test_dataset.get_class_mapping()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    logger.info(f"   Samples: {len(test_dataset)} | Classes: {len(class_to_idx)}")
    
    return test_loader, class_to_idx, idx_to_class


def init_model(num_classes, dinov2_model, checkpoint_path, device, logger):
    logger.info("🏗️ Initialisation modèle...")
    
    model = DINOv2AudioClassifier(
        num_classes=num_classes,
        dinov2_model=dinov2_model,
        freeze_dinov2=True
    ).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    logger.info("✅ Modèle chargé")
    return model


@torch.no_grad()
def infer(model, test_loader, device, logger):
    logger.info("🔍 Inférence...")
    model.eval()
    
    y_true, y_pred = [], []
    
    pbar = tqdm(test_loader, desc="🧪 Test")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            pbar.set_postfix({'acc': f'{acc:.4f}'})
    
    logger.info("✅ Inférence terminée")
    return np.array(y_true), np.array(y_pred)


def compute_metrics(y_true, y_pred, idx_to_class, result_dir, logger):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        output_dict=True,
        zero_division=0
    )
    
    df_report = pd.DataFrame(report).transpose()
    csv_path = os.path.join(result_dir, "classification_report.csv")
    df_report.to_csv(csv_path)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📈 METRICS")
    logger.info(f"{'='*70}")
    logger.info(f"   Accuracy:  {acc:.4f}")
    logger.info(f"   Precision: {prec:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info(f"   F1-macro:  {f1_macro:.4f}")
    logger.info(f"{'='*70}\n")
    
    return acc, f1_macro, cm


def plot_confusion_matrix(cm, idx_to_class, save_path):
    labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    cmn = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=axes[0])
    sns.heatmap(cmn, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels, cmap="Reds", ax=axes[1])
    
    axes[0].set_title("Confusion Matrix (Raw)")
    axes[1].set_title("Confusion Matrix (%)")
    
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    if args.batch_size: HPARAMS["batch_size"] = args.batch_size
    if args.checkpoint_path: HPARAMS["checkpoint_path"] = args.checkpoint_path
    if args.result_dir: HPARAMS["result_dir"] = args.result_dir
    HPARAMS["dinov2_model"] = args.dinov2_model
    HPARAMS["img_size"] = args.img_size
    
    logger = setup_logging(HPARAMS["result_dir"])
    device = get_device(HPARAMS["device_id"])
    
    test_loader, class_to_idx, idx_to_class = load_test_data(
        HPARAMS["dataset_root"],
        HPARAMS["batch_size"],
        HPARAMS["img_size"],
        args.dataset_name,
        logger
    )
    
    model = init_model(
        len(class_to_idx), 
        HPARAMS["dinov2_model"],
        HPARAMS["checkpoint_path"],
        device, 
        logger
    )
    
    y_true, y_pred = infer(model, test_loader, device, logger)
    
    acc, f1, cm = compute_metrics(y_true, y_pred, idx_to_class, HPARAMS["result_dir"], logger)
    
    cm_path = os.path.join(HPARAMS["result_dir"], "confusion_matrix.png")
    plot_confusion_matrix(cm, idx_to_class, cm_path)
    logger.info(f"✅ Confusion matrix: {cm_path}")
    
    logger.info(f"\n🏆 Final Accuracy: {acc:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    main()