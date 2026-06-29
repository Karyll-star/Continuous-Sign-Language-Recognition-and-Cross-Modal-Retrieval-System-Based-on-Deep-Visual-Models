"""
直接从自定义视频特征训练 CTC 模型（不依赖 CE-CSL，不需要增量训练）

用法：
    python train_custom.py

前提：
    - 已运行 extract_custom_features.py 生成特征
    - data/custom_videos/features/ 下有 .npy 文件
    - data/custom_videos/custom.csv 存在

产出：
    output/ctc_lstm/custom_model.pt —— 直接可用于 demo_infer.py
"""

import os
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from ctc_dataset import CSLFeatureDataset, Vocabulary, collate_fn, process_gloss
from ctc_model import TemporalConvBiLSTM, CTCTrainer
import pandas as pd


# ==================== 配置 ====================
CUSTOM_DIR = os.path.join(PROJECT_ROOT, "data", "custom_videos")
FEATURES_DIR = os.path.join(CUSTOM_DIR, "features")
LABEL_CSV = os.path.join(CUSTOM_DIR, "custom.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "ctc_lstm")
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "custom_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
WARMUP_EPOCHS = 5
PATIENCE = 20


def build_vocab_from_csv(csv_path: str) -> Vocabulary:
    vocab = Vocabulary()
    token_seqs = []
    for enc in ("utf-8", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            if "Gloss" in df.columns:
                for gloss_str in df["Gloss"]:
                    tokens = process_gloss(gloss_str)
                    if tokens:
                        token_seqs.append(tokens)
                break
        except UnicodeDecodeError:
            continue
    vocab.build_vocab(token_seqs)
    return vocab


def main():
    print("=" * 50)
    print("从自定义数据训练 CTC 手语识别模型")
    print("=" * 50)

    if not os.path.isdir(FEATURES_DIR):
        print(f"\n[错误] 特征目录不存在: {FEATURES_DIR}")
        print("请先运行: python extract_custom_features.py")
        return

    if not os.path.exists(LABEL_CSV):
        print(f"\n[错误] CSV 不存在: {LABEL_CSV}")
        return

    # 1. 构建词表
    vocab = build_vocab_from_csv(LABEL_CSV)
    tokens = [t for t in vocab.token2id if t not in ("<blank>", "<unk>")]
    print(f"\n词表: {tokens}")
    print(f"总类别数: {len(vocab)}")

    # 2. 数据集
    dataset = CSLFeatureDataset(
        features_dir=FEATURES_DIR,
        label_csv=LABEL_CSV,
        vocab=vocab,
        split="custom",
        max_samples=None,
    )
    print(f"样本数: {len(dataset)}")

    if len(dataset) == 0:
        print("\n[错误] 数据集为空")
        return

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )

    # 3. 模型
    model = TemporalConvBiLSTM(
        input_size=512,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(vocab),
        dropout=DROPOUT,
    ).to(DEVICE)

    trainer = CTCTrainer(model, device=DEVICE, lr=LR, weight_decay=WEIGHT_DECAY)

    from torch.optim.lr_scheduler import CosineAnnealingLR
    cosine_epochs = max(EPOCHS - WARMUP_EPOCHS, 1)
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=cosine_epochs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_loss = float("inf")
    epochs_no_improve = 0

    print(f"\n开始训练 ({EPOCHS} epochs, LR={LR}, device={DEVICE})")
    print("-" * 50)

    for epoch in range(EPOCHS):
        if epoch < WARMUP_EPOCHS:
            factor = float(epoch + 1) / max(WARMUP_EPOCHS, 1)
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = LR * factor
        else:
            scheduler.step()

        stats = trainer.train_epoch(loader, epoch=epoch)

        if (epoch + 1) % 10 == 0 or stats.loss < best_loss:
            token_acc = f"{stats.token_accuracy*100:.1f}%" if stats.token_accuracy else "N/A"
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {stats.loss:.4f} | Token Acc: {token_acc}")

        if stats.loss < best_loss:
            best_loss = stats.loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": {
                    "input_size": 512,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_classes": len(vocab),
                },
            }, OUTPUT_MODEL)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\n{'='*50}")
    print(f"训练完成！模型已保存到: {OUTPUT_MODEL}")
    print(f"最佳 Loss: {best_loss:.4f}")
    print(f"\n测试命令:")
    print(f"  python demo_infer.py --video ..\\data\\custom_videos\\你的视频.mp4")
    print(f"      --checkpoint {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
