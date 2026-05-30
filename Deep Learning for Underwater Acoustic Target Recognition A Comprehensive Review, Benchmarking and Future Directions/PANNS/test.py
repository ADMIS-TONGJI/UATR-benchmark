import os
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dataset import UnderwaterAudioDataset
from model import HydroPANNsCnn14_16k


HPARAMS = {
    "device_id": "1",

    "dataset_root": "./shipsear9_5s",

    "checkpoint_path": "./PANNs/final_epoch_model.pth",
    "num_classes": 9,
    "local_model_path": "./Cnn14_16k_mAP=0.438.pth",

    "sample_rate": 16000,
    "duration": 5,
    "target_length": 16000 * 5,   

    "batch_size": 16,
    "num_workers": 4,

    "result_dir": "./test_results",
    "cm_fig_name": "confusion_matrix.png"
}


def get_device(device_id: str = None) -> torch.device:
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"使用GPU设备: {device_id} (共{torch.cuda.device_count()}张GPU)")
    else:
        device = torch.device("cpu")
        print("使用CPU设备（推理速度较慢，建议使用GPU）")
    return device


DEVICE = get_device(HPARAMS["device_id"])


def collate_fn_waveform(batch):
    target_sr = HPARAMS["sample_rate"]
    target_length = HPARAMS["target_length"]

    waveforms = []
    labels = []

    for audio_path, label in batch:
        waveform, sr = torchaudio.load(audio_path)   # (channels, samples)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # -> (samples,)

        if sr != target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=target_sr
            )

        if waveform.size(0) < target_length:
            pad_len = target_length - waveform.size(0)
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:target_length]

        waveforms.append(waveform)
        labels.append(label)

    waveforms = torch.stack(waveforms, dim=0)   # (B, 80000)
    labels = torch.tensor(labels, dtype=torch.long)

    return waveforms, labels


def load_test_data(dataset_root: str, batch_size: int) -> Tuple[DataLoader, dict, dict]:
    test_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="test",
        is_validation=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_waveform,
        num_workers=HPARAMS["num_workers"],
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    class_to_idx = test_dataset.get_class_mapping()
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    if len(test_dataset) > 0:
        sample_waveform, sample_label = collate_fn_waveform([test_dataset[0]])
        feature_shape = sample_waveform[0].shape
    else:
        feature_shape = "未知（测试集无样本）"

    print(f"\n=== 测试集信息 ===")
    print(f"特征类型: waveform")
    print(f"波形维度: {feature_shape}")
    print(f"测试集总样本数: {len(test_dataset)}")
    print(f"类别映射（名称→标签）: {class_to_idx}")

    return test_loader, class_to_idx, idx_to_class

def init_model(num_classes: int, local_model_path: str, checkpoint_path: str,
               device: torch.device) -> HydroPANNsCnn14_16k:
    model = HydroPANNsCnn14_16k(
        num_classes=num_classes,
        local_model_path=local_model_path,
        freeze_base=False
    )
    model.to(device)
    model.eval()

    print(f"已初始化 HydroPANNsCnn14_16k 模型，设备: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"训练权重文件不存在: {checkpoint_path}")
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"PANNs预训练权重文件不存在: {local_model_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if next(iter(state_dict.keys()), "").startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print(f"\n已加载训练权重: {os.path.basename(checkpoint_path)}")
    print(f"训练轮次: {checkpoint.get('epoch', '未知')}")
    if "best_val_acc" in checkpoint:
        print(f"训练时最优验证准确率: {checkpoint['best_val_acc']:.4f}")

    return model


@torch.no_grad()
def infer(model: HydroPANNsCnn14_16k, test_loader: DataLoader,
          device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []

    print(f"\n=== 开始推理测试集 ===")
    for batch_idx, (waveforms, labels) in enumerate(test_loader):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        _, probabilities = model(waveforms)
        pred_labels = torch.argmax(probabilities, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred_labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  已处理批次: {batch_idx + 1:3d}/{len(test_loader)}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f"推理完成！共处理样本数: {len(y_true)}")

    return y_true, y_pred


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      idx_to_class: dict) -> Tuple[dict, np.ndarray]:
    overall_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

    cm = confusion_matrix(y_true, y_pred, labels=sorted(idx_to_class.keys()))
    return overall_metrics, cm


def plot_confusion_matrix(cm: np.ndarray, idx_to_class: dict, save_path: str,
                          cmap: str = "Blues"):
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    num_classes = len(class_names)

    fig_size = (12 + num_classes // 3, 10 + num_classes // 3) if num_classes > 5 else (12, 10)
    plt.figure(figsize=fig_size)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 8 if num_classes > 5 else 10}
    )

    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.title("HydroPANNsCnn14_16k 测试集混淆矩阵", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n混淆矩阵已保存至: {os.path.abspath(save_path)}")


def print_results(overall_metrics: dict, cm: np.ndarray, idx_to_class: dict):

    print("\n【1. 整体评估指标】")
    print(f"Accuracy:      {overall_metrics['accuracy']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall:    {overall_metrics['recall']:.4f}")
    print(f"F1-score:  {overall_metrics['f1-score']:.4f}")

    print("\n【2. 混淆矩阵】")
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]

    print(" " * 18 + " ".join([f"{name[:8]:^8s}" for name in class_names]))
    for i, idx in enumerate(sorted(idx_to_class.keys())):
        cls_name = idx_to_class[idx]
        row_data = cm[i]
        print(f"{cls_name[:18]:<18s}" + " ".join([f"{val:^8d}" for val in row_data]))


def main():
    test_loader, class_to_idx, idx_to_class = load_test_data(
        dataset_root=HPARAMS["dataset_root"],
        batch_size=HPARAMS["batch_size"]
    )

    model = init_model(
        num_classes=HPARAMS["num_classes"],
        local_model_path=HPARAMS["local_model_path"],
        checkpoint_path=HPARAMS["checkpoint_path"],
        device=DEVICE
    )

    y_true, y_pred = infer(
        model=model,
        test_loader=test_loader,
        device=DEVICE
    )

    overall_metrics, cm = calculate_metrics(y_true, y_pred, idx_to_class)

    print_results(overall_metrics, cm, idx_to_class)
    cm_save_path = os.path.join(HPARAMS["result_dir"], HPARAMS["cm_fig_name"])
    plot_confusion_matrix(cm, idx_to_class, cm_save_path)

    print(f"结果保存目录: {os.path.abspath(HPARAMS['result_dir'])}")


if __name__ == "__main__":
    main()
