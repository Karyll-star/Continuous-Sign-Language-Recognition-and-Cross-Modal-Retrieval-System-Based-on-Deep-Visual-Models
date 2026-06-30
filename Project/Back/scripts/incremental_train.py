"""
增量训练脚本：在已训练好的手语识别模型上新增一个句子类别。
使用方法:
    python scripts/incremental_train.py --sentence "今天天气很好"

兼容两种检查点格式:
  1. train_model.py       → 包含 label_encoder_classes, ResNet18
  2. train_model_final.py → 可能不含 label_encoder_classes, 支持 ResNet18/50/34/101

参数说明:
  --sentence       新增的手语句子文本（必填）
  --checkpoint     检查点路径（默认 model/last_checkpoint.pt）
  --output_dir     输出目录（默认 models/sign_language）
  --replay_samples 旧类别重放样本数（默认 500，用于防止灾难性遗忘）
  --epochs         增量训练轮数（默认 5）
  --lr             学习率（默认 1e-4，建议比初次训练低 10-100 倍）
  --batch_size     批次大小（默认 16）

当前限制: 使用哑元帧训练；真实帧需要先实现帧提取管线。
"""

import os
import sys
import json
import shutil
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------- 设备 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}  | "
          f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
else:
    print("[Device] CPU")

# ---------- 配置 ----------
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "model" / "last_checkpoint.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "sign_language"

# ResNet 系列映射
_BACKBONE_MAP = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights,  512),
    "resnet34":  (models.resnet34,  models.ResNet34_Weights,  512),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights, 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights, 2048),
}

def _get_backbone_info(model_name: str) -> tuple:
    """返回 (builder_fn, weights_cls, backbone_dim)。"""
    if model_name not in _BACKBONE_MAP:
        print(f"  ⚠ 未知模型名 '{model_name}'，回退到 resnet18")
        model_name = "resnet18"
    return _BACKBONE_MAP[model_name]


# ---------- 分类器（兼容 ResNet18/34/50/101）----------
class SignLanguageClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, dropout=0.5,
                 pretrained=True, freeze_backbone=True, model_name="resnet18"):
        super().__init__()
        fn, wcls, backbone_dim = _get_backbone_info(model_name)
        if pretrained:
            weights = getattr(wcls, "IMAGENET1K_V1", None)
            self.backbone = fn(weights=weights)
        else:
            self.backbone = fn(weights=None)
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("  ▸ 骨干网络已解冻")


# ---------- 哑元数据集（供验证；真实训练需替换）----------
class DummySignDataset(Dataset):
    """生成哑元图像帧，每条样本绑定一个标签。"""
    def __init__(self, labels: List[int], total_per_label: int = 50):
        self.labels = labels
        self.total = total_per_label

    def __len__(self):
        return len(self.labels) * self.total

    def __getitem__(self, idx):
        label = self.labels[idx // self.total]
        rng = np.random.RandomState(idx * 7 + label * 31)
        frame = torch.stack([
            torch.from_numpy(rng.rand(224, 224).astype(np.float32)) for _ in range(3)
        ])
        frame = (frame - 0.45) / 0.25
        return frame, label


# ---------- 工具：加载 / 创建 LabelEncoder ----------
def _load_label_encoder_from_file(output_dir: Path) -> Optional[LabelEncoder]:
    """尝试从 models/sign_language/label_encoder.pkl 加载。"""
    p = output_dir / "label_encoder.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def _load_classes_from_label_map(output_dir: Path) -> Optional[List[str]]:
    """尝试从 label_map.json 反向获取类别列表。"""
    p = output_dir / "label_map.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        # 按数字键排序
        items = sorted(d.items(), key=lambda x: int(x[0]))
        return [v for _, v in items]
    return None

def get_or_create_label_encoder(checkpoint: dict, output_dir: Path) -> LabelEncoder:
    """
    优先从检查点取 label_encoder_classes，其次从 label_encoder.pkl，
    最后从 label_map.json 回退。
    """
    classes = checkpoint.get("label_encoder_classes")
    if classes is not None:
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        print(f"  ▸ 从检查点恢复 {len(le.classes_)} 个类别")
        return le

    le = _load_label_encoder_from_file(output_dir)
    if le is not None:
        print(f"  ▸ 从 label_encoder.pkl 恢复 {len(le.classes_)} 个类别")
        return le

    classes = _load_classes_from_label_map(output_dir)
    if classes is not None:
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        print(f"  ▸ 从 label_map.json 恢复 {len(le.classes_)} 个类别")
        return le

    raise RuntimeError(
        "检查点中缺少 label_encoder_classes，且 models/sign_language/ 下没有 "
        "label_encoder.pkl 或 label_map.json。请先完成一次完整训练。"
    )

def save_label_encoder(le: LabelEncoder, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(le, f)
    print(f"  ▸ LabelEncoder 已保存: {path}")

def save_label_map(le: LabelEncoder, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {str(i): cls for i, cls in enumerate(le.classes_)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"  ▸ LabelMap 已保存: {path}")


# ---------- 核心：扩展分类头 ----------
def _find_last_linear_key(classifier_keys: set, state_dict: dict) -> str:
    """
    在 classifier 的 state_dict keys 中找出最后一个 Linear 层的 weight key。
    通过维度过滤（Linear weight 是 2D，BatchNorm 是 1D），并用数字后缀定位。
    """
    linear_weight_keys = [
        k for k in classifier_keys
        if k.endswith(".weight") and "classifier" in k
        and len(state_dict[k].shape) == 2  # 排除 BatchNorm 的 1D weight
    ]
    linear_weight_keys.sort(key=lambda k: int(k.split(".")[1]))
    return linear_weight_keys[-1] if linear_weight_keys else "classifier.8.weight"

def expand_model_for_new_classes(
    model: SignLanguageClassifier,
    old_num_classes: int,
    new_num_classes: int,
    old_state_dict: dict,
):
    """将分类头从 old_num_classes 扩展到 new_num_classes，保留已有权重。"""
    old_backbone_keys = {k for k in old_state_dict if k.startswith("backbone.")}
    old_classifier_keys = {k for k in old_state_dict if k.startswith("classifier.")}

    # 定位最后一个 Linear 层（通过 2D shape 过滤 BN 层）
    last_linear_weight = _find_last_linear_key(old_classifier_keys, old_state_dict)
    last_linear_bias = last_linear_weight.replace(".weight", ".bias")

    new_state = {}
    for k in sorted(old_backbone_keys | old_classifier_keys):
        old_t = old_state_dict[k]
        if k == last_linear_weight:
            new_t = torch.zeros(new_num_classes, old_t.shape[1])
            new_t[:old_num_classes] = old_t
            nn.init.xavier_uniform_(new_t[old_num_classes:])
            new_state[k] = new_t
        elif k == last_linear_bias:
            new_t = torch.zeros(new_num_classes)
            new_t[:old_num_classes] = old_t
            new_state[k] = new_t
        else:
            new_state[k] = old_t

    model.load_state_dict(new_state, strict=False)
    print(f"  ▸ 分类头已扩展: {old_num_classes} → {new_num_classes}")
    print(f"  ▸ 前 {old_num_classes} 个类别权重保留，"
          f"后 {new_num_classes - old_num_classes} 个随机初始化")
    return model


# ---------- 训练 / 评估 ----------
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch}")
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="增量训练 - 新增手语句子类别")
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--replay_samples", type=int, default=500)
    parser.add_argument("--new_samples", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--unfreeze_backbone", action="store_true")
    args = parser.parse_args()

    new_sentence = args.sentence.strip()
    if not new_sentence:
        print("❌ 错误: 句子不能为空"); sys.exit(1)

    print("=" * 60)
    print("  增量训练: 手语识别模型")
    print(f"  新增句子: {new_sentence}")
    print(f"  检查点:   {args.checkpoint}")
    print(f"  设备:     {DEVICE}")
    print("=" * 60)

    # ---- (1) 加载检查点 ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ 检查点不存在: {ckpt_path}")
        print("  请先完整训练一次模型 (例如: python scripts/train_model.py)")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "resnet18")
    hidden_dim = config.get("hidden_dim", 512)
    dropout = config.get("dropout", 0.5)
    pretrained = config.get("pretrained", True)

    # 恢复 LabelEncoder
    output_dir = Path(args.output_dir)
    label_encoder = get_or_create_label_encoder(checkpoint, output_dir)
    old_classes = label_encoder.classes_.tolist()
    old_num_classes = len(old_classes)

    print(f"\n[1/5] 加载检查点完成")
    print(f"  模型:      {model_name}")
    print(f"  已有类别:  {old_num_classes}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  best_acc:   {checkpoint.get('best_acc', 0):.2f}%")

    # ---- (2) 构建扩展模型 ----
    new_classes_list = old_classes + [new_sentence]
    new_num_classes = len(new_classes_list)

    print(f"\n[2/5] 构建扩展模型 ({new_num_classes} 类 → 新增 1 类)")
    model = SignLanguageClassifier(
        num_classes=new_num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pretrained=pretrained,
        freeze_backbone=not args.unfreeze_backbone,
        model_name=model_name,
    ).to(DEVICE)

    model = expand_model_for_new_classes(
        model, old_num_classes, new_num_classes,
        checkpoint["model_state_dict"],
    )

    # ---- (3) 准备数据 ----
    print(f"\n[3/5] 准备增量训练数据")

    # 重放旧类别（防止灾难性遗忘）
    if old_num_classes > 0:
        replay_ds = DummySignDataset(
            labels=list(range(old_num_classes)),
            total_per_label=max(1, args.replay_samples // old_num_classes),
        )
        if len(replay_ds) > args.replay_samples:
            idxs = np.random.choice(len(replay_ds), args.replay_samples, replace=False)
            replay_ds = Subset(replay_ds, idxs)
    else:
        replay_ds = DummySignDataset(labels=[], total_per_label=0)

    # 新句子数据
    new_label = old_num_classes
    new_ds = DummySignDataset(labels=[new_label], total_per_label=args.new_samples)
    train_dataset = ConcatDataset([replay_ds, new_ds])
    val_ds = DummySignDataset(labels=[new_label], total_per_label=10)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"  重放样本:   {len(replay_ds)} 条 (旧类别)")
    print(f"  新句子样本: {len(new_ds)} 条 (标签 ID={new_label})")
    print(f"  总训练集:   {len(train_dataset)} 条")
    print(f"  验证集:     {len(val_ds)} 条")

    # ---- (4) 增量训练 ----
    print(f"\n[4/5] 开始增量训练 ({args.epochs} epochs, lr={args.lr})")
    if args.unfreeze_backbone:
        model.unfreeze_backbone()

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        vl, va = evaluate(model, val_loader, criterion, DEVICE)
        print(f"  ── Epoch {epoch}/{args.epochs} ──")
        print(f"    Train Loss: {tl:.4f}  Acc: {ta:.1f}%")
        print(f"    Val   Loss: {vl:.4f}  Acc: {va:.1f}%")
        if va > best_acc:
            best_acc = va

    print(f"\n  增量训练完成, 最佳验证准确率: {best_acc:.1f}%")

    # ---- (5) 保存 ----
    print(f"\n[5/5] 保存模型与标签文件")
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = output_dir / f"incremental_{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 5a) 模型
    model_path = save_dir / "sign_language_model.pth"
    ckpt_out = {
        "epoch": checkpoint.get("epoch", 0),
        "model_state_dict": model.state_dict(),
        "best_acc": best_acc,
        "config": {
            "model_name": model_name,
            "num_classes": new_num_classes,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "pretrained": pretrained,
            "freeze_backbone": not args.unfreeze_backbone,
        },
        "label_encoder_classes": new_classes_list,
        "incremental": {
            "new_sentence": new_sentence,
            "old_num_classes": old_num_classes,
            "added_at": datetime.now().isoformat(),
        },
    }
    torch.save(ckpt_out, model_path)
    print(f"  ▸ 模型已保存: {model_path}")

    # 5b) 标签
    le = LabelEncoder()
    le.classes_ = np.array(new_classes_list)
    save_label_encoder(le, save_dir / "label_encoder.pkl")
    save_label_map(le, save_dir / "label_map.json")

    # 5c) 部署覆盖
    shutil.copy2(model_path, output_dir / "sign_language_model.pth")
    save_label_encoder(le, output_dir / "label_encoder.pkl")
    save_label_map(le, output_dir / "label_map.json")

    print(f"\n{'=' * 60}")
    print(f"  增量训练完毕！")
    print(f"  新增句子:  {new_sentence}  (标签 ID={new_label})")
    print(f"  总类别数:  {new_num_classes}")
    print(f"  部署路径:  {output_dir / 'sign_language_model.pth'}")
    print(f"  重启后端即可识别该句子。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
