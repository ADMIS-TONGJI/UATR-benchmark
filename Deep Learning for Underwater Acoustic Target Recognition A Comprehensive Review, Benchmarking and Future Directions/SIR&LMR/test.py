import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
from typing import Tuple

from dataset import UnderwaterAudioDataset
from feature_extraction import get_collate_fn
from model import create_model


HPARAMS = {
    "device_id": "0",
    # mel
    "feature": "mel",
    "sr": 16000,
    "target_seconds": 5.0,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    # 推理
    "batch_size": 32,
    # 结果保存
    "result_dir": "./test_results",
    "cm_fig_name": "confusion_matrix.png",
}


def parse_args():
    parser = argparse.ArgumentParser(description="SIR+LMR 测试入口（禁用增强）")
    parser.add_argument("--dataset_name", type=str, default="shipsear9_5s",
                        help="选择数据集名称（将从 data/{dataset_name} 读取）")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="数据集根路径（若不提供，将按项目目录下 data/{dataset_name} 推断）")
    parser.add_argument("--device_id", type=str, default=None, help="CUDA 设备编号，如 '0' 或 '0,1'")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="模型权重路径（若不提供，将默认从 checkpoints/{dataset_name}/best_val_model.pth 读取）")
    return parser.parse_args()


def get_device(device_id: str = None) -> torch.device:
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"使用GPU设备: {device_id} (共{torch.cuda.device_count()}张GPU)")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    return device


def load_test_data(dataset_root: str, batch_size: int) -> Tuple[DataLoader, dict, dict]:
    test_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="test",
        is_validation=False,
    )

    collate_fn = get_collate_fn(
        feature=HPARAMS["feature"],
        is_train=False,
        sr=HPARAMS["sr"],
        target_seconds=HPARAMS["target_seconds"],
        n_mels=HPARAMS["n_mels"],
        n_fft=HPARAMS["n_fft"],
        hop_length=HPARAMS["hop_length"],
        win_length=HPARAMS["win_length"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    class_to_idx = test_dataset.get_class_mapping()
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(f"\n测试集信息:")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别映射: {class_to_idx}")
    print(f"使用特征: {HPARAMS['feature']}")

    return test_loader, class_to_idx, idx_to_class


def init_model(num_classes: int, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型权重文件不存在: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = create_model(num_classes=num_classes)
    model.to(device)
    model.eval()
    model.load_state_dict(state_dict)
    print(f"\n模型权重已加载: {checkpoint_path}")
    print(f"加载的模型轮次: {state.get('epoch', '未知')}")
    print(f"最优验证集准确率: {state.get('best_val_acc', '未知')}")
    return model


@torch.no_grad()
def infer(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    print(f"\n开始推理测试集...")
    for batch_idx, (features, labels) in enumerate(test_loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        if (batch_idx + 1) % 10 == 0:
            print(f"  处理批次 {batch_idx+1:3d}/{len(test_loader)}")
    return np.array(y_true), np.array(y_pred)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict) -> Tuple[dict, str, np.ndarray]:
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[i] for i in labels]
    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return overall, report, cm


def save_confusion_matrix(cm: np.ndarray, idx_to_class: dict, save_dir: str, fig_name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=[idx_to_class[i] for i in range(len(idx_to_class))],
                yticklabels=[idx_to_class[i] for i in range(len(idx_to_class))])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    save_path = os.path.join(save_dir, fig_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    if args.dataset_root:
        dataset_root = args.dataset_root
    else:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        dataset_root = os.path.join(project_dir, "data", dataset_name)

    device_id = args.device_id if args.device_id is not None else HPARAMS["device_id"]
    device = get_device(device_id)

    test_loader, class_to_idx, idx_to_class = load_test_data(dataset_root, HPARAMS["batch_size"])

    checkpoint_path = args.checkpoint_path or os.path.join("./checkpoints", dataset_name, "best_val_model.pth")

    model = init_model(num_classes=len(class_to_idx), checkpoint_path=checkpoint_path, device=device)

    y_true, y_pred = infer(model, test_loader, device)

    overall, report, cm = calculate_metrics(y_true, y_pred, idx_to_class)

    print("\n整体指标:")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    print("\n各类别报告:\n" + report)

    save_confusion_matrix(cm, idx_to_class, HPARAMS["result_dir"], HPARAMS["cm_fig_name"])


if __name__ == "__main__":
    main()