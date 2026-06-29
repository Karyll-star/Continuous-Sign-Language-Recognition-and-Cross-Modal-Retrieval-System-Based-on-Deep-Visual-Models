"""
增量训练脚本 —— 在已有模型基础上添加新词汇（如"拿铁"）

功能：
1. 加载已有预训练模型 (best_model.pt)
2. 从 custom.csv 中读取新词，扩展词表
3. 扩展模型输出层（仅新增维度随机初始化，旧维度权重保留）
4. 使用低学习率微调，防止灾难性遗忘

用法：
    # 使用默认路径（项目根目录/data/custom_videos/）
    python finetune_new_word.py

    # 指定自定义数据路径
    python finetune_new_word.py --custom-features data/my_features --custom-label data/my.csv

    训练完成后，模型保存到 output/ctc_lstm/best_model_finetuned.pt
    原有的 best_model.pt 不会覆盖。

目录结构（默认）：
    data/custom_videos/
    ├── *.mp4                ← 手语视频（gitignore 已忽略）
    ├── custom.csv           ← 标注文件
    └── features/            ← extract_custom_features.py 生成的 .npy 特征

环境变量：
    CECSL_DATA_ROOT  — CE-CSL 数据集根目录（覆盖默认路径）
"""

import os
import copy
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ctc_dataset import (
    CSLFeatureDataset,
    Vocabulary,
    collate_fn,
    process_gloss,
)
from ctc_model import TemporalConvBiLSTM, CTCTrainer


# ==================== 项目路径 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CUSTOM_DIR = os.path.join(PROJECT_ROOT, "data", "custom_videos")

# CE-CSL 数据集路径（支持环境变量覆盖）
CECSL_DATA_ROOT = os.environ.get(
    "CECSL_DATA_ROOT",
    r"D:\Aprogress\Shen\dataset\CE-CSL\CE-CSL"
)


# ==================== 配置 ====================
@dataclass
class FinetuneConfig:
    # CE-CSL 数据集路径（旧数据来源）
    DATA_ROOT: str = CECSL_DATA_ROOT
    LABEL_DIR: str = os.path.join(DATA_ROOT, "label")

    # 旧数据的特征目录
    TRAIN_FEATURES: str = os.path.join(DATA_ROOT, "train_features")

    # 自定义数据路径（命令行可覆盖）
    CUSTOM_LABEL: str = os.path.join(DEFAULT_CUSTOM_DIR, "custom.csv")
    CUSTOM_FEATURES: str = os.path.join(DEFAULT_CUSTOM_DIR, "features")

    # 预训练模型路径
    PRETRAINED_MODEL: str = os.path.join("output", "ctc_lstm", "best_model.pt")

    # 输出
    OUTPUT_DIR: str = os.path.join("output", "ctc_lstm")
    OUTPUT_MODEL: str = os.path.join(OUTPUT_DIR, "best_model_finetuned.pt")

    # 微调参数 —— 使用较低学习率防止灾难性遗忘
    BATCH_SIZE: int = 8
    LR: float = 1e-4          # 比从头训练低一个数量级
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 30          # 微调不需要太多轮
    WARMUP_EPOCHS: int = 3
    PATIENCE: int = 10

    # 旧数据抽样数量（防止新词数据过少导致灾难性遗忘）
    MAX_OLD_TRAIN_SAMPLES: int | None = 2000  # 从旧训练集抽样 2000 条
    MAX_OLD_VAL_SAMPLES: int | None = 200

    # 新数据在 batch 中的最小比例（通过加权采样实现）
    NEW_SAMPLE_WEIGHT: float = 5.0  # 新样本的权重倍数（相对于旧样本）

    INPUT_SIZE: int = 512
    HIDDEN_SIZE: int = 256
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = FinetuneConfig()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# ==================== 核心：词表扩展 + 模型扩展 ====================
def load_and_extend_model(
    pretrained_path: str,
    new_tokens: List[str],
    device: torch.device,
) -> tuple:
    """
    加载预训练模型并扩展词表（添加新词）。

    返回：(model, vocab) —— 模型 fc 层已扩展, 新词汇权重随机初始化
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"预训练模型不存在: {pretrained_path}\n"
            "请先完成基础训练: python train_lstm.py"
        )

    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    old_vocab: Vocabulary = ckpt["vocab"]
    old_config: Dict = ckpt["config"]

    old_num_classes = old_config["num_classes"]
    old_model_state = ckpt["model_state_dict"]

    print(f"旧词表大小: {len(old_vocab)}")
    print(f"旧模型 num_classes: {old_num_classes}")

    # ---- 1. 扩展词表（深拷贝，不破坏旧 checkpoint 中的 vocab） ----
    new_vocab = Vocabulary()
    # 把旧词表中的 token 全部复制过来
    for token_id in range(len(old_vocab)):
        token = old_vocab.id2token[token_id]
        # 确保索引对齐（blank=0, unk=1 保持不变）
        if token_id == 0:     # blank
            pass  # 已在 __init__ 中存在
        elif token_id == 1:   # unk
            pass
        else:
            new_vocab.token2id[token] = new_vocab.n_tokens
            new_vocab.id2token[new_vocab.n_tokens] = token
            new_vocab.n_tokens += 1

    # 添加新词
    new_token_ids_start = new_vocab.n_tokens
    new_token_ids = []
    for tok in new_tokens:
        if tok not in new_vocab.token2id:
            new_vocab.token2id[tok] = new_vocab.n_tokens
            new_vocab.id2token[new_vocab.n_tokens] = tok
            new_token_ids.append(new_vocab.n_tokens)
            new_vocab.n_tokens += 1
            print(f"  新增词汇: '{tok}' -> ID={new_token_ids[-1]}")
        else:
            print(f"  [跳过] 词汇 '{tok}' 已存在词表中")

    new_num_classes = len(new_vocab)

    # ---- 2. 创建新模型并扩展 fc 层 ----
    model = TemporalConvBiLSTM(
        input_size=old_config["input_size"],
        hidden_size=old_config["hidden_size"],
        num_layers=old_config["num_layers"],
        num_classes=new_num_classes,
        dropout=cfg.DROPOUT,
    )

    # 加载旧权重（fc 层部分加载）
    old_fc_weight = old_model_state["fc.weight"]  # (old_classes, hidden*2)
    old_fc_bias = old_model_state["fc.bias"]      # (old_classes,)

    new_state = copy.deepcopy(old_model_state)
    # 扩展 fc 层权重
    new_fc_weight = torch.zeros(new_num_classes, old_fc_weight.shape[1])
    new_fc_bias = torch.zeros(new_num_classes)

    # 复制旧词权重
    new_fc_weight[:old_num_classes, :] = old_fc_weight
    new_fc_bias[:old_num_classes] = old_fc_bias

    # 新词权重：用旧词权重的均值 + 小随机噪声初始化
    weight_mean = old_fc_weight.mean(dim=0, keepdim=True)
    weight_std = old_fc_weight.std(dim=0, keepdim=True)
    noise = torch.randn(new_num_classes - old_num_classes, old_fc_weight.shape[1])
    new_fc_weight[old_num_classes:, :] = weight_mean + noise * weight_std * 0.1
    new_fc_bias[old_num_classes:] = 0.0

    new_state["fc.weight"] = new_fc_weight
    new_state["fc.bias"] = new_fc_bias

    model.load_state_dict(new_state)
    model.to(device)
    model.eval()

    print(f"新词表大小: {len(new_vocab)} (新增 {new_num_classes - old_num_classes} 个)")
    print(f"新模型 num_classes: {new_num_classes}")

    return model, new_vocab


# ==================== 从 CSV 中提取新词 ====================
def get_new_words(csv_path: str) -> List[str]:
    """从自定义 CSV 中提取所有 unique 的 Gloss 词"""
    df = None
    for enc in ("utf-8", "gbk"):
        try:
            tmp = pd.read_csv(csv_path, encoding=enc)
            if "Gloss" in tmp.columns:
                df = tmp
                break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise RuntimeError(f"无法读取 {csv_path}")

    words = set()
    for gloss_str in df["Gloss"]:
        tokens = process_gloss(gloss_str)
        for t in tokens:
            words.add(t)
    return sorted(words)


# ==================== 混合数据集（旧数据 + 新数据） ====================
class MixedWeightedDataset(torch.utils.data.Dataset):
    """
    将旧数据集和新数据集混合，通过 WeightedRandomSampler 给新数据更高权重。
    """
    def __init__(
        self,
        old_dataset: CSLFeatureDataset,
        new_dataset: CSLFeatureDataset,
        new_sample_weight: float = 5.0,
    ):
        self.old_dataset = old_dataset
        self.new_dataset = new_dataset
        self.new_weight = new_sample_weight
        self.old_len = len(old_dataset)
        self.new_len = len(new_dataset)
        self.total = self.old_len + self.new_len

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < self.old_len:
            return self.old_dataset[idx]
        else:
            return self.new_dataset[idx - self.old_len]

    def get_sample_weights(self) -> List[float]:
        """返回每个样本的权重（用于 WeightedRandomSampler）"""
        weights = [1.0] * self.old_len + [self.new_weight] * self.new_len
        return weights


# ==================== 主流程 ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="增量训练：为新词汇微调手语识别模型")
    parser.add_argument(
        "--custom-label",
        type=str,
        default=None,
        help=f"自定义标注 CSV 路径（默认: {cfg.CUSTOM_LABEL}）",
    )
    parser.add_argument(
        "--custom-features",
        type=str,
        default=None,
        help=f"自定义特征目录（默认: {cfg.CUSTOM_FEATURES}）",
    )
    args = parser.parse_args()

    # 命令行参数覆盖默认值
    custom_label = args.custom_label or cfg.CUSTOM_LABEL
    custom_features_dir = args.custom_features or cfg.CUSTOM_FEATURES

    print("=" * 60)
    print("增量训练：为新词汇微调手语识别模型")
    print("=" * 60)

    # ---- 1. 提取新词 ----
    if not os.path.exists(custom_label):
        print(f"[错误] 自定义标注文件不存在: {custom_label}")
        print("请先创建。推荐格式：")
        print("  Number,Translator,Chinese Sentences,Gloss,video_file,Note")
        print("  00001,Z,拿铁,拿铁,拿铁_01.mp4,")
        return

    new_words = get_new_words(custom_label)
    print(f"\n从 {custom_label} 中提取到新词: {new_words}")

    if len(new_words) == 0:
        print("[错误] 未找到任何新词，请检查 CSV 的 Gloss 列")
        return

    # ---- 2. 加载预训练模型并扩展词表 ----
    print(f"\n[1/4] 加载预训练模型: {cfg.PRETRAINED_MODEL}")
    model, vocab = load_and_extend_model(
        cfg.PRETRAINED_MODEL,
        new_words,
        cfg.DEVICE,
    )

    # ---- 3. 构建混合数据集 ----
    print(f"\n[2/4] 构建训练数据集...")

    # 旧数据（抽样）
    old_train_label = os.path.join(cfg.LABEL_DIR, "train.csv")
    if not os.path.exists(old_train_label):
        print(f"[错误] CE-CSL 训练标注不存在: {old_train_label}")
        print("请设置环境变量 CECSL_DATA_ROOT 指向正确的 CE-CSL 数据集根目录")
        return

    old_train = CSLFeatureDataset(
        features_dir=cfg.TRAIN_FEATURES,
        label_csv=old_train_label,
        vocab=vocab,
        split="train",
        max_samples=cfg.MAX_OLD_TRAIN_SAMPLES,
    )
    print(f"  旧训练集抽样: {len(old_train)} 条")

    # 新数据
    if not os.path.isdir(custom_features_dir):
        print(f"\n[错误] 自定义特征目录不存在: {custom_features_dir}")
        print("请先运行: python extract_custom_features.py")
        return

    new_train = CSLFeatureDataset(
        features_dir=custom_features_dir,
        label_csv=custom_label,
        vocab=vocab,
        split="custom",
        max_samples=None,
    )
    print(f"  新词数据集: {len(new_train)} 条")

    if len(new_train) == 0:
        print(f"\n[错误] 新词数据集为空！请检查：")
        print(f"  1. 特征目录: {custom_features_dir}")
        print(f"  2. 是否已运行: python extract_custom_features.py")
        return

    # 混合数据集
    mixed_train = MixedWeightedDataset(
        old_train, new_train, new_sample_weight=cfg.NEW_SAMPLE_WEIGHT
    )
    sample_weights = mixed_train.get_sample_weights()
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(mixed_train),
        replacement=True,
    )

    train_loader = DataLoader(
        mixed_train,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 验证集只用旧数据的抽样（评估是否遗忘）
    val_dataset = CSLFeatureDataset(
        features_dir=os.path.join(cfg.DATA_ROOT, "val_features"),
        label_csv=os.path.join(cfg.LABEL_DIR, "dev.csv"),
        vocab=vocab,
        split="dev",
        max_samples=cfg.MAX_OLD_VAL_SAMPLES,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"  验证集（旧数据）: {len(val_dataset)} 条")

    # ---- 4. 开始微调 ----
    print(f"\n[3/4] 开始微调 ({cfg.EPOCHS} epochs, LR={cfg.LR})")

    trainer = CTCTrainer(
        model,
        device=cfg.DEVICE,
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR
    cosine_epochs = max(cfg.EPOCHS - cfg.WARMUP_EPOCHS, 1)
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=cosine_epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_path = cfg.OUTPUT_MODEL
    last_path = os.path.join(cfg.OUTPUT_DIR, "last_finetune.pt")

    for epoch in range(cfg.EPOCHS):
        if epoch < cfg.WARMUP_EPOCHS:
            warmup_factor = float(epoch + 1) / max(cfg.WARMUP_EPOCHS, 1)
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = cfg.LR * warmup_factor
        else:
            scheduler.step()

        print(f"\n===== Epoch {epoch + 1}/{cfg.EPOCHS} =====")
        train_stats = trainer.train_epoch(train_loader, epoch=epoch)
        val_stats = trainer.evaluate(val_loader, epoch=epoch)

        print(f"Train CTC Loss: {train_stats.loss:.4f}")
        print(f" Val  CTC Loss: {val_stats.loss:.4f}")
        if train_stats.token_accuracy is not None:
            print(f"Train token acc: {train_stats.token_accuracy * 100:.2f}%")
        if val_stats.token_accuracy is not None:
            print(f" Val  token acc: {val_stats.token_accuracy * 100:.2f}%")

        if val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "input_size": cfg.INPUT_SIZE,
                        "hidden_size": cfg.HIDDEN_SIZE,
                        "num_layers": cfg.NUM_LAYERS,
                        "num_classes": len(vocab),
                    },
                },
                best_path,
            )
            print(f"[保存最佳模型] {best_path} (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # 保存断点续训文件
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "vocab": vocab,
                "config": {
                    "input_size": cfg.INPUT_SIZE,
                    "hidden_size": cfg.HIDDEN_SIZE,
                    "num_layers": cfg.NUM_LAYERS,
                    "num_classes": len(vocab),
                },
            },
            last_path,
        )

        if epochs_no_improve >= cfg.PATIENCE:
            print(f"\n[Early Stopping] {epochs_no_improve} epochs 未提升")
            break

    print(f"\n[4/4] 微调完成！")
    print(f"  新模型: {best_path}")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"\n下一步：")
    print(f"  测试新词: python demo_infer.py --video <test.mp4>")
    print(f"     --checkpoint {best_path}")


if __name__ == "__main__":
    main()
