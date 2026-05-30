import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataset import UnderwaterAudioDataset
from model import HydroPANNsCnn14_16k

HPARAMS_DEVICE = {
    "device_id": "2"
}
os.environ["CUDA_VISIBLE_DEVICES"] = HPARAMS_DEVICE["device_id"]


HPARAMS = {

    "device_id": HPARAMS_DEVICE["device_id"],

    "dataset_root": "./shipsear9_5s",
    "sample_rate": 16000,
    "duration": 5.0,

    "batch_size": 32,
    "epochs": 100,
    "init_lr": 1e-3,
    "lr_decay_step": 20,
    "lr_decay_gamma": 0.4,
    "weight_decay": 1e-5,
    "num_workers": 4,

    "val_split_ratio": 0.15,
    "random_seed": 42,

    "early_stop_patience": 20,

    "num_classes": 9,
    "local_model_path": "./Cnn14_16k_mAP=0.438.pth",
    "freeze_base": True,

    "checkpoint_dir": "./checkpoints",
    "best_ckpt_name": "best_val_model.pth",
    "final_ckpt_name": "final_epoch_model.pth",

    "result_dir": "./results",
    "cm_fig_name": "test_confusion_matrix.png"
}


def get_device(device_id: str = None) -> torch.device:
    if device_id and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU设备: {device_id} (当前可见GPU数: {torch.cuda.device_count()})")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    return device


DEVICE = get_device(HPARAMS["device_id"])


def load_data(dataset_root: str, batch_size: int):
    train_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=False,
        val_split_ratio=HPARAMS["val_split_ratio"],
        random_seed=HPARAMS["random_seed"],
        target_sample_rate=HPARAMS["sample_rate"],
        target_duration=HPARAMS["duration"]
    )

    val_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=True,
        val_split_ratio=HPARAMS["val_split_ratio"],
        random_seed=HPARAMS["random_seed"],
        target_sample_rate=HPARAMS["sample_rate"],
        target_duration=HPARAMS["duration"]
    )

    test_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="test",
        is_validation=False,
        val_split_ratio=HPARAMS["val_split_ratio"],
        random_seed=HPARAMS["random_seed"],
        target_sample_rate=HPARAMS["sample_rate"],
        target_duration=HPARAMS["duration"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=HPARAMS["num_workers"],
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=HPARAMS["num_workers"],
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=HPARAMS["num_workers"],
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别映射: {train_dataset.get_class_mapping()}")

    if len(train_dataset) > 0:
        sample_waveform, sample_label = train_dataset[0]
        print(f"单个样本waveform shape: {sample_waveform.shape}")
        print(f"单个样本标签: {sample_label}")

    return train_loader, val_loader, test_loader, train_dataset.get_class_mapping()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (waveforms, labels) in enumerate(loader):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = model(waveforms)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = waveforms.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        if (batch_idx + 1) % 10 == 0:
            batch_acc = (preds == labels).sum().item() / batch_size
            print(f"  批次 {batch_idx+1:3d}/{len(loader)} | 批损失: {loss.item():.4f} | 批准确率: {batch_acc:.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    y_true = []
    y_pred = []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = model(waveforms)
        loss = criterion(logits, labels)

        batch_size = waveforms.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, np.array(y_true), np.array(y_pred)


def save_checkpoint(save_path, model, optimizer, scheduler, epoch, best_val_acc=None):
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "num_classes": HPARAMS["num_classes"],
        "local_model_path": HPARAMS["local_model_path"]
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"\nCheckpoint已保存至: {os.path.abspath(save_path)}")


def load_checkpoint_for_test(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    if next(iter(state_dict.keys()), "").startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print(f"已加载测试模型权重: {checkpoint_path}")
    return model


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


def plot_confusion_matrix(cm, idx_to_class, save_path, title="Test Confusion Matrix"):
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    num_classes = len(class_names)

    plt.figure(figsize=(12 + num_classes // 3, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存至: {os.path.abspath(save_path)}")


def print_test_results(metrics, cm, idx_to_class):

    print("\n【1. 整体评估指标】")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1-score']:.4f}")

    print("\n【2. 混淆矩阵】")
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    print(" " * 18 + " ".join([f"{name[:8]:^8s}" for name in class_names]))
    for i, idx in enumerate(sorted(idx_to_class.keys())):
        cls_name = idx_to_class[idx]
        row_data = cm[i]
        print(f"{cls_name[:18]:<18s}" + " ".join([f"{val:^8d}" for val in row_data]))


def main():
    print("=== 开始加载数据 ===")
    train_loader, val_loader, test_loader, class_to_idx = load_data(
        dataset_root=HPARAMS["dataset_root"],
        batch_size=HPARAMS["batch_size"]
    )
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    print("\n=== 初始化模型 ===")
    model = HydroPANNsCnn14_16k(
        num_classes=HPARAMS["num_classes"],
        local_model_path=HPARAMS["local_model_path"],
        freeze_base=HPARAMS["freeze_base"]
    )
    model.to(DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"启用多GPU训练: 共{torch.cuda.device_count()}张GPU")

    print("\n=== 初始化训练组件 ===")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HPARAMS["init_lr"],
        weight_decay=HPARAMS["weight_decay"]
    )

    scheduler = StepLR(
        optimizer=optimizer,
        step_size=HPARAMS["lr_decay_step"],
        gamma=HPARAMS["lr_decay_gamma"]
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    no_improve_epochs = 0
    best_ckpt_path = os.path.join(HPARAMS["checkpoint_dir"], HPARAMS["best_ckpt_name"])

    print(f"\n=== 开始训练（共{HPARAMS['epochs']}轮）===")
    for epoch in range(1, HPARAMS["epochs"] + 1):
        print(f"\n【轮次 {epoch:2d}/{HPARAMS['epochs']}】")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE
        )

        val_loss, val_acc, _, _ = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )

        print("轮次总结:")
        print(f"  训练集 - 损失: {train_loss:.4f} | 准确率: {train_acc:.4f}")
        print(f"  验证集 - 损失: {val_loss:.4f} | 准确率: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            print(f"  验证集准确率提升，更新 best_val_acc = {best_val_acc:.4f}")

            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc
            )
        else:
            no_improve_epochs += 1
            print(f"  验证集准确率未提升，连续无提升轮次: {no_improve_epochs}")

        if no_improve_epochs >= HPARAMS["early_stop_patience"]:
            print(f"\n早停触发！连续 {HPARAMS['early_stop_patience']} 轮验证集准确率无提升")
            break

        scheduler.step()

    final_epoch = epoch
    final_ckpt_path = os.path.join(HPARAMS["checkpoint_dir"], HPARAMS["final_ckpt_name"])
    save_checkpoint(
        save_path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=final_epoch,
        best_val_acc=best_val_acc
    )

    test_model = HydroPANNsCnn14_16k(
        num_classes=HPARAMS["num_classes"],
        local_model_path=HPARAMS["local_model_path"],
        freeze_base=False
    )
    test_model.to(DEVICE)
    test_model = load_checkpoint_for_test(test_model, best_ckpt_path, DEVICE)

    test_loss, test_acc, y_true, y_pred = evaluate_one_epoch(
        model=test_model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE
    )

    metrics, cm = calculate_metrics(y_true, y_pred)
    print_test_results(metrics, cm, idx_to_class)

    cm_save_path = os.path.join(HPARAMS["result_dir"], HPARAMS["cm_fig_name"])
    plot_confusion_matrix(
        cm=cm,
        idx_to_class=idx_to_class,
        save_path=cm_save_path,
        title="HydroPANNsCnn14_16k Test Confusion Matrix"
    )

    print(f"\n=== 全部流程完成 ===")
    print(f"最优验证集准确率: {best_val_acc:.4f}")
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"最佳模型路径: {os.path.abspath(best_ckpt_path)}")
    print(f"最后一轮模型路径: {os.path.abspath(final_ckpt_path)}")
    print(f"测试结果目录: {os.path.abspath(HPARAMS['result_dir'])}")


if __name__ == "__main__":
    main()