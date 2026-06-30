"""
模型批量评估脚本 —— 同时评估新旧两个模型在 CE-CSL 验证集上的表现

用法：
    python eval_models.py

前提：
    - 已提取 val_features（python extract_features.py --splits dev --max-samples 0）
    - 环境变量 CECSL_DATA_ROOT 已设
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from torch.utils.data import DataLoader
from ctc_dataset import CSLFeatureDataset, Vocabulary, collate_fn
from ctc_model import TemporalConvBiLSTM, CTCTrainer

# ==================== 配置 ====================
CECSL = os.environ.get("CECSL_DATA_ROOT",
    os.path.join(os.path.dirname(SCRIPT_DIR), "dataset", "CE-CSL"))

DEVC_CSV = os.path.join(CECSL, "label", "dev.csv")
VAL_FEATURES = os.path.join(CECSL, "val_features")

MODEL_PATHS = {
    "旧模型 (last_checkpoint)": os.path.join(os.path.dirname(SCRIPT_DIR), "Project", "Back", "model", "last_checkpoint.pt"),
    "新模型 (finetuned)": os.path.join(SCRIPT_DIR, "output", "ctc_lstm", "best_model_finetuned.pt"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_SAMPLES = None  # 设为数字可限制评估样本数，None = 全量


def evaluate_one(name: str, ckpt_path: str):
    print(f"\n{'='*50}")
    print(f"评估: {name}")
    print(f"  路径: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        print(f"  ❌ 文件不存在，跳过")
        return None

    # 加载
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    vocab = ckpt["vocab"]
    config = ckpt["config"]

    model = TemporalConvBiLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=0.2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)

    # 数据集
    dataset = CSLFeatureDataset(
        features_dir=VAL_FEATURES,
        label_csv=DEVC_CSV,
        vocab=vocab,
        split="dev",
        max_samples=MAX_SAMPLES,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # 评估
    trainer = CTCTrainer(model, device=DEVICE, lr=1e-4)
    stats = trainer.evaluate(loader)

    result = {
        "loss": stats.loss,
        "token_acc": stats.token_accuracy,
        "seq_acc": stats.seq_accuracy,
        "n_samples": len(dataset),
    }

    print(f"  样本数: {result['n_samples']}")
    print(f"  CTC Loss: {result['loss']:.4f}")
    if result['token_acc'] is not None:
        print(f"  Token 准确率: {result['token_acc']*100:.1f}%")
    if result['seq_acc'] is not None:
        print(f"  整句准确率: {result['seq_acc']*100:.1f}%")

    return result


def main():
    print("=" * 50)
    print("模型批量评估")
    print(f"设备: {DEVICE}")
    print(f"验证集: {VAL_FEATURES}")
    print(f"标注:   {DEVC_CSV}")

    results = {}
    for name, path in MODEL_PATHS.items():
        results[name] = evaluate_one(name, path)

    # 汇总
    print(f"\n{'='*50}")
    print("汇总对比")
    print(f"{'='*50}")
    print(f"{'模型':<30} {'Loss':>10} {'Token Acc':>10} {'Seq Acc':>10}")
    print("-" * 60)
    for name, r in results.items():
        if r is None:
            continue
        tok = f"{r['token_acc']*100:.1f}%" if r['token_acc'] is not None else "N/A"
        seq = f"{r['seq_acc']*100:.1f}%" if r['seq_acc'] is not None else "N/A"
        print(f"{name:<30} {r['loss']:>10.4f} {tok:>10} {seq:>10}")

    # 结论
    if len(results) >= 2:
        old = results.get("旧模型 (last_checkpoint)")
        new = results.get("新模型 (finetuned)")
        if old and new:
            loss_diff = new["loss"] - old["loss"]
            print(f"\nLoss 变化: {loss_diff:+.4f}")
            if loss_diff > 1.0:
                print("⚠ 新模型验证 Loss 显著升高 → 可能存在灾难性遗忘")
                print("  建议：增加 MAX_OLD_TRAIN_SAMPLES 或降低 LR 重新微调")
            elif abs(loss_diff) < 0.5:
                print("✅ Loss 基本持平 → 增量训练未伤害旧词能力")
            else:
                print("  变化在可接受范围内")


if __name__ == "__main__":
    main()
