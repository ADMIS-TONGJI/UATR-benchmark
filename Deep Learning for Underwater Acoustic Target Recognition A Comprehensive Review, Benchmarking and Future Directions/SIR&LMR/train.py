import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from typing import Tuple

from dataset import UnderwaterAudioDataset
from feature_extraction import get_collate_fn
from model import create_model


HPARAMS = {
    "device_id": "0",
    "batch_size": 16,
    "epochs": 60,
    "init_lr": 1e-3,
    "lr_decay_step": 20,
    "lr_decay_gamma": 0.2,
    # mel
    "feature": "mel",
    "sr": 16000,
    "target_seconds": 5.0,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    # LMR
    "lmr_p": 0.7,
    "lmr_num_patches": 2,
    "lmr_max_h_ratio": 0.2,
    "lmr_max_w_ratio": 0.2,
    "lmr_inter_prob": 0.3,
    # SIR
    "lambda_sir": 0.5,
    "sir_amp_scale": 0.1,
    "sir_max_time_shift_ratio": 0.02,
    "sir_max_freq_shift_ratio": 0.02,
    "sir_noise_std": 0.01,
    # 早停
    "early_stop_patience": 60,
    # 路径
    "checkpoint_dir": "./checkpoints",
}


def parse_args():
    parser = argparse.ArgumentParser(description="SIR+LMR 训练入口")
    parser.add_argument("--dataset_name", type=str, default="shipsear9_5s",
                        help="选择数据集名称（将从 data/{dataset_name} 读取）")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="数据集根路径（若不提供，将按项目目录下 data/{dataset_name} 推断）")
    parser.add_argument("--device_id", type=str, default=None, help="CUDA 设备编号，如 '0' 或 '0,1'")
    return parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_id: str = None) -> torch.device:
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"使用GPU设备: {device_id} (共{torch.cuda.device_count()}张GPU)")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    return device


def load_data(dataset_root: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=False,
        val_split_ratio=0.15,
        random_seed=42,
    )
    val_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=True,
        val_split_ratio=0.15,
        random_seed=42,
    )

    lmr_cfg = {
        "p": HPARAMS["lmr_p"],
        "num_patches": HPARAMS["lmr_num_patches"],
        "max_h_ratio": HPARAMS["lmr_max_h_ratio"],
        "max_w_ratio": HPARAMS["lmr_max_w_ratio"],
        "inter_prob": HPARAMS["lmr_inter_prob"],
    }
    sir_cfg = {
        "amp_scale": HPARAMS["sir_amp_scale"],
        "max_time_shift_ratio": HPARAMS["sir_max_time_shift_ratio"],
        "max_freq_shift_ratio": HPARAMS["sir_max_freq_shift_ratio"],
        "noise_std": HPARAMS["sir_noise_std"],
    }

    collate_train = get_collate_fn(
        feature=HPARAMS["feature"],
        is_train=True,
        sr=HPARAMS["sr"],
        target_seconds=HPARAMS["target_seconds"],
        n_mels=HPARAMS["n_mels"],
        n_fft=HPARAMS["n_fft"],
        hop_length=HPARAMS["hop_length"],
        win_length=HPARAMS["win_length"],
        lmr_cfg=lmr_cfg,
        sir_sim_cfg=sir_cfg,
    )
    collate_eval = get_collate_fn(
        feature=HPARAMS["feature"],
        is_train=False,
        sr=HPARAMS["sr"],
        target_seconds=HPARAMS["target_seconds"],
        n_mels=HPARAMS["n_mels"],
        n_fft=HPARAMS["n_fft"],
        hop_length=HPARAMS["hop_length"],
        win_length=HPARAMS["win_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_train,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"\n数据集信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"类别映射: {train_dataset.get_class_mapping()}")
    print(f"使用特征: {HPARAMS['feature']}")

    return train_loader, val_loader

def symmetrical_kl_loss(p: torch.Tensor, q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """计算对称 KL 散度损失"""
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return (F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(p.log(), q, reduction='batchmean')) / 2

def mse_prob(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.mean((p - q) ** 2)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device, lambda_sir: float) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        # 训练 collate 返回三元组
        features, labels, features_sim = batch
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features_sim = features_sim.to(device, non_blocking=True)

        logits = model(features)
        logits_sim = model(features_sim)

        loss_cls = criterion(logits, labels)
        p = torch.softmax(logits, dim=1)
        p_sim = torch.softmax(logits_sim, dim=1)
        loss_sir = symmetrical_kl_loss(p, p_sim)
        loss = loss_cls + lambda_sir * loss_sir

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        pred_labels = torch.argmax(logits, dim=1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += features.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        pred_labels = torch.argmax(logits, dim=1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += features.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def save_checkpoint(save_path: str, model: nn.Module, optimizer: optim.Optimizer,
                    scheduler: StepLR, epoch: int, best_val_acc: float = None) -> None:
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint已保存至: {save_path}")


def main():
    args = parse_args()
    set_seed(42)

    dataset_name = args.dataset_name
    if args.dataset_root:
        dataset_root = args.dataset_root
    else:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        dataset_root = os.path.join(project_dir, "data", dataset_name)

    device_id = args.device_id if args.device_id is not None else HPARAMS["device_id"]
    device = get_device(device_id)

    train_loader, val_loader = load_data(dataset_root=dataset_root, batch_size=HPARAMS["batch_size"])

    # 推断类别数
    num_classes = len(train_loader.dataset.get_class_mapping())

    model = create_model(num_classes=num_classes)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"启用多GPU训练: {torch.cuda.device_count()}张GPU")

    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["init_lr"])
    scheduler = StepLR(optimizer=optimizer, step_size=HPARAMS["lr_decay_step"], gamma=HPARAMS["lr_decay_gamma"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    no_improve_epochs = 0
    # 预先定义最优/最后checkpoint路径，避免变量未定义
    best_ckpt_path = os.path.join(HPARAMS["checkpoint_dir"], dataset_name, "best_val_model.pth")
    final_ckpt_path = os.path.join(HPARAMS["checkpoint_dir"], dataset_name, "final_epoch_model.pth")

    print(f"\n=== 开始训练（共{HPARAMS['epochs']}轮）===")
    for epoch in range(1, HPARAMS["epochs"] + 1):
        print(f"\n【轮次 {epoch:2d}/{HPARAMS['epochs']}】 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lambda_sir=HPARAMS["lambda_sir"],
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"轮次总结: 训练损失 {train_loss:.4f} | 训练准确率 {train_acc:.4f} | 验证损失 {val_loss:.4f} | 验证准确率 {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_val_acc)
        else:
            no_improve_epochs += 1
            print(f"  验证集准确率无提升（连续{no_improve_epochs}轮）")

        if no_improve_epochs >= HPARAMS["early_stop_patience"]:
            print(f"\n早停触发！连续{HPARAMS['early_stop_patience']}轮验证集准确率无提升")
            break

        scheduler.step()

    save_checkpoint(final_ckpt_path, model, optimizer, scheduler, epoch, best_val_acc)

    print(f"\n=== 训练完成 ===")
    print(f"最优验证集准确率: {best_val_acc:.4f}")
    print(f"最优模型路径: {os.path.abspath(best_ckpt_path)}")
    print(f"最后一轮模型路径: {os.path.abspath(final_ckpt_path)}")


if __name__ == "__main__":
    main()