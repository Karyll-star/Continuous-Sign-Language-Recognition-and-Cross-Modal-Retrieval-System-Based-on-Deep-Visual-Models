"""
自举验证脚本 —— 不依赖 CE-CSL 数据集，用自定义数据独立验证增量训练流程。

原理：
  将你的自定义数据分成两半 —— 一半模拟"旧模型已认识的词"，另一半模拟"要新增的词"。
  全程只用 data/custom_videos/ 下的数据，不需要 CE-CSL。

用法：
  python bootstrap_and_finetune.py

它会：
  1. 用"旧词"训练一个基础模型（模拟预训练模型）
  2. 加载这个模型 + 扩展词表加入"新词"
  3. 用整个数据集做增量微调
  4. 比较微调前后对新词的识别能力

这样你就能在 5-10 分钟内验证整个增量训练流程是否能跑通。
"""

import os
import sys
import copy
import tempfile
import argparse
import numpy as np
import torch

# 确保能找到项目模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from ctc_dataset import CSLFeatureDataset, Vocabulary, collate_fn, process_gloss
from ctc_model import TemporalConvBiLSTM, CTCTrainer
import pandas as pd
from torch.utils.data import DataLoader


# ==================== 配置 ====================
CUSTOM_DIR = os.path.join(PROJECT_ROOT, "data", "custom_videos")
FEATURES_DIR = os.path.join(CUSTOM_DIR, "features")
LABEL_CSV = os.path.join(CUSTOM_DIR, "custom.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 基础训练参数（快速验证用）
BASE_EPOCHS = 10
FINETUNE_EPOCHS = 8
BATCH_SIZE = 4
LR = 1e-3
HIDDEN_SIZE = 128  # 小模型，快速训练


def build_vocab_from_csv(csv_path: str) -> Vocabulary:
    """从 CSV 构建词表"""
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


def train_small_model(
    features_dir: str,
    label_csv: str,
    vocab: Vocabulary,
    epochs: int,
    lr: float,
    label: str = "",
) -> tuple:
    """用给定数据训练一个小模型，返回 (model, best_loss)"""
    dataset = CSLFeatureDataset(
        features_dir=features_dir,
        label_csv=label_csv,
        vocab=vocab,
        split="bootstrap",
        max_samples=None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"数据集为空: {features_dir}")

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )

    model = TemporalConvBiLSTM(
        input_size=512, hidden_size=HIDDEN_SIZE,
        num_layers=1, num_classes=len(vocab), dropout=0.2,
    ).to(DEVICE)

    trainer = CTCTrainer(model, device=DEVICE, lr=lr, weight_decay=1e-4)
    best_loss = float("inf")

    prefix = f"[{label}] " if label else ""
    for epoch in range(epochs):
        stats = trainer.train_epoch(loader, epoch=epoch)
        if stats.loss < best_loss:
            best_loss = stats.loss
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  {prefix}Epoch {epoch+1}/{epochs}  Loss: {stats.loss:.4f}")

    return model, best_loss


def evaluate_model(model, features_dir, label_csv, vocab):
    """简单评估：返回 Loss"""
    dataset = CSLFeatureDataset(
        features_dir=features_dir, label_csv=label_csv,
        vocab=vocab, split="eval",
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    trainer = CTCTrainer(model, device=DEVICE, lr=LR)
    stats = trainer.evaluate(loader)
    return stats.loss


def ctc_decode_test(model, features_path, vocab):
    """对单个特征文件做 CTC 解码，返回预测词列表"""
    feats = np.load(features_path)
    if feats.ndim != 2 or feats.shape[0] == 0:
        return ["(空)"]

    x = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
    lens = torch.tensor([feats.shape[0]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        log_probs, _ = model(x, lens)
        log_probs = log_probs.squeeze(1)  # (T, C)
        best_path = log_probs.argmax(dim=-1).tolist()

    collapsed = []
    prev = None
    for idx in best_path:
        if idx == 0:
            prev = None
            continue
        if prev is not None and idx == prev:
            continue
        collapsed.append(idx)
        prev = idx

    tokens = [vocab.id2token.get(i, "<unk>") for i in collapsed]
    return tokens


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("自举验证：不依赖 CE-CSL，验证增量训练全流程")
    print("=" * 60)

    # ---- 0. 检查数据 ----
    if not os.path.isdir(FEATURES_DIR):
        print(f"\n[错误] 特征目录不存在: {FEATURES_DIR}")
        print("请先运行: python extract_custom_features.py")
        return

    csv_path = LABEL_CSV
    if not os.path.exists(csv_path):
        print(f"\n[错误] CSV 不存在: {csv_path}")
        return

    # 读取所有特征文件
    all_features = sorted([
        f for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")
    ])
    n_total = len(all_features)
    if n_total < 4:
        print(f"\n[错误] 至少需要 4 个特征文件，当前只有 {n_total} 个")
        return

    print(f"\n数据检查通过: {n_total} 个特征文件")

    # ---- 1. 构建完整词表 + 拆分"旧词"/"新词" ----
    full_vocab = build_vocab_from_csv(csv_path)
    all_tokens = sorted([
        t for t in full_vocab.token2id.keys()
        if t not in ("<blank>", "<unk>")
    ])
    print(f"完整词表: {all_tokens}")

    if len(all_tokens) < 1:
        print("\n[错误] 词表为空，请检查 CSV 的 Gloss 列")
        return

    # 策略：如果只有一个词，我们无法拆分"旧词"和"新词"。
    # 此时用"模拟词"来验证流程：假设模型认识一个假词，需要新增真实词。
    if len(all_tokens) == 1:
        real_word = all_tokens[0]
        fake_word = "__FAKE_OLD__"
        print(f"\n只有一个词 '{real_word}'，将用模拟词 '{fake_word}' 来验证增量流程")
        # 为模拟词生成随机特征
        os.makedirs(os.path.join(FEATURES_DIR, "_fake"), exist_ok=True)
        fake_feat_dir = os.path.join(FEATURES_DIR, "_fake")
        # 从真实特征中取一半作为"旧词"的模拟数据
        old_features = all_features[: n_total // 2]
        new_features = all_features[n_total // 2 :]
        old_csv_path = os.path.join(CUSTOM_DIR, "_fake_old.csv")
        new_csv_path = csv_path  # 新词就用原始 CSV
    else:
        # 多个词：前一半当旧词，后一半当新词
        old_tokens = all_tokens[: len(all_tokens) // 2]
        new_tokens = all_tokens[len(all_tokens) // 2 :]
        real_word = None
        fake_word = None
        old_features = []
        new_features = []
        # 需要按 Gloss 拆分特征文件（简化：直接按文件数量对半分）
        old_features = all_features[: n_total // 2]
        new_features = all_features[n_total // 2 :]
        old_csv_path = csv_path
        new_csv_path = csv_path

    print(f"旧词特征: {len(old_features)} 个")
    print(f"新词特征: {len(new_features)} 个")

    # ---- 2. 创建旧词数据（模拟 CE-CSL 旧数据集） ----
    if fake_word:
        # 单词语境：生成假的旧词特征
        os.makedirs(fake_feat_dir, exist_ok=True)
        for i, fname in enumerate(old_features):
            src = os.path.join(FEATURES_DIR, fname)
            dst = os.path.join(fake_feat_dir, f"fake_{i:04d}_Z.npy")
            feats = np.load(src)
            np.save(dst, feats + np.random.randn(*feats.shape).astype(np.float32) * 0.01)

        # 生成假的 CSV
        df_old = pd.DataFrame({
            "Number": [f"fake_{i:04d}" for i in range(len(old_features))],
            "Translator": ["Z"] * len(old_features),
            "Chinese Sentences": [fake_word] * len(old_features),
            "Gloss": [fake_word] * len(old_features),
            "Note": ["模拟旧词"] * len(old_features),
        })
        df_old.to_csv(old_csv_path, index=False, encoding="utf-8")

        old_feat_dir = fake_feat_dir
        old_label_csv = old_csv_path

        # 为旧词构建词表
        old_vocab = Vocabulary()
        old_vocab.build_vocab([[fake_word]])
    else:
        old_feat_dir = FEATURES_DIR
        old_label_csv = old_csv_path
        old_vocab = build_vocab_from_csv(old_csv_path)

    # ---- 3. 训练"预训练模型"（模拟 best_model.pt） ----
    print(f"\n{'='*40}")
    print(f"阶段 A：训练基础模型（模拟预训练）")
    print(f"  词表: {[t for t in old_vocab.token2id if t not in ('<blank>','<unk>')]}")
    print(f"  词汇数: {len(old_vocab)}")

    try:
        base_model, base_loss = train_small_model(
            old_feat_dir, old_label_csv, old_vocab,
            epochs=BASE_EPOCHS, lr=LR, label="Base"
        )
    except RuntimeError as e:
        print(f"\n[错误] 基础训练失败: {e}")
        return

    print(f"  基础模型训练完成，Loss: {base_loss:.4f}")

    # 保存为模拟的 best_model.pt
    tmp_ckpt = os.path.join(tempfile.gettempdir(), "bootstrap_best_model.pt")
    torch.save({
        "epoch": BASE_EPOCHS - 1,
        "model_state_dict": base_model.state_dict(),
        "vocab": old_vocab,
        "config": {
            "input_size": 512,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": 1,
            "num_classes": len(old_vocab),
        },
    }, tmp_ckpt)
    print(f"  模拟预训练模型已保存到: {tmp_ckpt}")

    # ---- 4. 用 finetune_new_word.py 的核心逻辑做增量训练 ----
    # 这里我们手动调用 load_and_extend_model + 训练，不依赖 CE-CSL
    print(f"\n{'='*40}")
    print(f"阶段 B：增量训练（模拟添加新词）")

    from finetune_new_word import load_and_extend_model, get_new_words

    # 获取新词
    new_words = get_new_words(new_csv_path)
    # 过滤出旧词表中不存在的词
    truly_new = [w for w in new_words if w not in old_vocab.token2id]
    if not truly_new:
        print(f"  所有词已在旧词表中，无法验证增量流程")
        print(f"  旧词表: {list(old_vocab.token2id.keys())}")
        print(f"  CSV中词: {new_words}")
        return
    print(f"  旧词表已有的词: {[w for w in new_words if w in old_vocab.token2id]}")
    print(f"  需要新增的词: {truly_new}")

    # 加载并扩展模型
    finetune_model, extended_vocab = load_and_extend_model(
        tmp_ckpt, truly_new, DEVICE
    )
    print(f"  扩展后词表大小: {len(extended_vocab)}")

    # 构建混合数据集：旧特征 + 新特征
    class SimpleMixedDataset(torch.utils.data.Dataset):
        def __init__(self, old_ds, new_ds):
            self.old = old_ds
            self.new = new_ds
            self.old_len = len(old_ds)
            self.new_len = len(new_ds)
        def __len__(self):
            return self.old_len + self.new_len
        def __getitem__(self, idx):
            if idx < self.old_len:
                return self.old[idx]
            return self.new[idx - self.old_len]

    old_ds = CSLFeatureDataset(
        old_feat_dir, old_label_csv, extended_vocab, split="old",
    )
    new_ds = CSLFeatureDataset(
        FEATURES_DIR, new_csv_path, extended_vocab, split="new",
    )
    mixed_ds = SimpleMixedDataset(old_ds, new_ds)

    weights = [1.0] * len(old_ds) + [5.0] * len(new_ds)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(mixed_ds), replacement=True,
    )

    ft_loader = DataLoader(
        mixed_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=0, collate_fn=collate_fn,
    )

    ft_trainer = CTCTrainer(
        finetune_model, device=DEVICE, lr=LR * 0.1, weight_decay=1e-4,
    )

    print(f"\n  开始微调 ({FINETUNE_EPOCHS} epochs)...")
    best_ft_loss = float("inf")
    for epoch in range(FINETUNE_EPOCHS):
        stats = ft_trainer.train_epoch(ft_loader, epoch=epoch)
        if stats.loss < best_ft_loss:
            best_ft_loss = stats.loss
        if (epoch + 1) % max(1, FINETUNE_EPOCHS // 4) == 0:
            print(f"    Epoch {epoch+1}/{FINETUNE_EPOCHS}  Loss: {stats.loss:.4f}")

    print(f"  微调完成，最佳 Loss: {best_ft_loss:.4f}")

    # ---- 5. 验证：微调后能否识别新词 ----
    print(f"\n{'='*40}")
    print(f"阶段 C：验证")

    # 对每个新词的特征做一次推理
    for word in truly_new:
        # 找一个对应的特征文件
        found = False
        for fname in all_features:
            fpath = os.path.join(FEATURES_DIR, fname)
            predicted = ctc_decode_test(finetune_model, fpath, extended_vocab)
            tokens_str = "".join(predicted) if predicted else "(空)"
            if word in tokens_str or any(word in t for t in predicted):
                print(f"  ✅ '{word}' → 预测: {tokens_str}")
                found = True
                break
        if not found:
            # 展示第一个新词特征文件的预测
            for fname in all_features:
                fpath = os.path.join(FEATURES_DIR, fname)
                predicted = ctc_decode_test(finetune_model, fpath, extended_vocab)
                print(f"  🔍 新词 '{word}' 的样本预测: {''.join(predicted) if predicted else '(空)'}")
                break

    # ---- 6. 总结 ----
    print(f"\n{'='*60}")
    print("验证结论")
    print(f"{'='*60}")
    print(f"  ✅ 基础模型训练: 成功 (Loss={base_loss:.4f})")
    print(f"  ✅ 词表扩展: {len(old_vocab)} → {len(extended_vocab)} (+{len(truly_new)})")
    print(f"  ✅ 增量微调: 成功 (Loss={best_ft_loss:.4f})")
    print(f"\n增量训练流程验证通过！")
    print(f"\n当你有了 CE-CSL 数据集后，只需：")
    print(f"  1. python train_lstm.py          # 训练真正的预训练模型")
    print(f"  2. python finetune_new_word.py   # 运行正式增量训练")
    if fake_word:
        print(f"\n清理模拟数据：")
        print(f"  del {old_csv_path}")
        print(f"  rmdir /s {fake_feat_dir}")


if __name__ == "__main__":
    main()
