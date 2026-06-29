"""
提取自定义词视频的特征（阶段一扩展）

用法：
    # 使用默认路径（项目根目录/data/custom_videos/）
    python extract_custom_features.py

    # 指定自定义数据目录
    python extract_custom_features.py --data-dir D:\\my_videos

    特征会保存到: <data-dir>/features/
    每个视频对应一个 .npy 文件，形状为 (T, 512)

目录结构（默认 data/custom_videos/）：
    data/custom_videos/
    ├── 拿铁_01.mp4          ← 你的手语视频
    ├── 拿铁_02.mp4
    ├── custom.csv            ← 标注文件
    └── features/             ← 自动生成的特征（gitignore 已忽略 *.npy）
        ├── 00001_Z.npy
        └── ...

CSV 格式（支持两种模式）：
    【推荐】扁平模式 — 包含 video_file 列：
        Number,Translator,Chinese Sentences,Gloss,video_file,Note
        00001,Z,拿铁,拿铁,拿铁_01.mp4,

    【兼容】层级模式 — 无 video_file 列，按 CE-CSL 层级查找：
        Number,Translator,Chinese Sentences,Gloss,Note
        custom-00001,Z,拿铁,拿铁,
        （视频需放在 <data-dir>/<Translator>/<Number>.mp4）
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import pandas as pd
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


# ==================== 配置 ====================
class Config:
    # 默认数据目录：项目根目录下的 data/custom_videos/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "custom_videos")

    RESNET_MODEL = "resnet18"
    IMAGE_SIZE = 224
    FEATURE_BATCH_SIZE = 16
    FRAME_SAMPLE_RATE = 1  # 每帧都取（保留完整时序）

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()


# ==================== 特征提取器（复用阶段一逻辑） ====================
class ResNetFeatureExtractor:
    def __init__(self):
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.to(cfg.DEVICE)
        self.feature_dim = 512

    @torch.no_grad()
    def extract(self, frames: list) -> np.ndarray:
        if len(frames) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        processed = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (cfg.IMAGE_SIZE + 32, cfg.IMAGE_SIZE + 32))
            h, w = frame.shape[:2]
            sh = (h - cfg.IMAGE_SIZE) // 2
            sw = (w - cfg.IMAGE_SIZE) // 2
            frame = frame[sh:sh + cfg.IMAGE_SIZE, sw:sw + cfg.IMAGE_SIZE]
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            frame = frame.transpose(2, 0, 1)
            processed.append(frame)

        all_feats = []
        T = len(processed)
        bs = cfg.FEATURE_BATCH_SIZE
        for start in range(0, T, bs):
            end = min(start + bs, T)
            batch_np = np.stack(processed[start:end], axis=0)
            batch = torch.from_numpy(batch_np).float().to(cfg.DEVICE)
            feats = self.backbone(batch)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(feats.size(0), -1)
            all_feats.append(feats.cpu())
            if cfg.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        return torch.cat(all_feats, dim=0).numpy()


def load_video_frames(video_path: str) -> list:
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % cfg.FRAME_SAMPLE_RATE == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def find_video(data_dir: str, row: pd.Series) -> str | None:
    """
    根据 CSV 行查找视频文件，支持两种模式：

    1. 扁平模式：row 中有 'video_file' 列 → 在 data_dir 下直接查找
    2. 层级模式：用 Number + Translator 在 data_dir/<Translator>/<Number>.mp4 查找
    """
    number = str(row["Number"])
    translator = str(row["Translator"])

    # 模式 1：video_file 列（推荐）
    if "video_file" in row.index:
        video_file = str(row["video_file"])
        if pd.notna(video_file) and video_file.strip():
            # 支持相对路径（相对于 data_dir）
            candidate = os.path.join(data_dir, video_file.strip())
            if os.path.exists(candidate):
                return candidate
            # 也尝试绝对路径
            if os.path.exists(video_file.strip()):
                return video_file.strip()

    # 模式 2：层级结构 data_dir/<Translator>/<Number>.mp4
    candidate = os.path.join(data_dir, translator, f"{number}.mp4")
    if os.path.exists(candidate):
        return candidate

    # 模式 3：扁平 data_dir/<Number>.mp4
    candidate = os.path.join(data_dir, f"{number}.mp4")
    if os.path.exists(candidate):
        return candidate

    return None


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(
        description="提取自定义手语词汇的视频特征（阶段一扩展）"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=cfg.DEFAULT_DATA_DIR,
        help=f"自定义数据目录，包含视频和 custom.csv（默认: {cfg.DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV 标注文件路径（默认: <data-dir>/custom.csv）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="特征输出目录（默认: <data-dir>/features/）",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    csv_path = args.csv or os.path.join(data_dir, "custom.csv")
    output_dir = args.output_dir or os.path.join(data_dir, "features")

    if not os.path.exists(csv_path):
        print(f"[错误] CSV 文件不存在: {csv_path}")
        print()
        print("请创建标注文件。推荐格式（包含 video_file 列）：")
        print("  Number,Translator,Chinese Sentences,Gloss,video_file,Note")
        print("  00001,Z,拿铁,拿铁,拿铁_01.mp4,")
        print()
        print(f"视频请放在: {data_dir}/")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 读取标注
    df = None
    for enc in ("utf-8", "gbk"):
        try:
            tmp = pd.read_csv(csv_path, encoding=enc)
            if "Number" in tmp.columns and "Translator" in tmp.columns:
                df = tmp
                break
            if "Column1" in tmp.columns:
                tmp = pd.read_csv(csv_path, encoding=enc, header=1)
                if "Number" in tmp.columns:
                    df = tmp
                    break
        except UnicodeDecodeError:
            continue

    if df is None:
        print("[错误] 无法解析 CSV 文件，请确保包含 Number, Translator 列")
        return

    # 检测模式
    has_video_file = "video_file" in df.columns
    mode_label = "扁平模式 (video_file 列)" if has_video_file else "层级模式 (Translator 目录)"

    print(f"设备: {cfg.DEVICE}")
    print(f"数据目录: {data_dir}")
    print(f"标注文件: {csv_path}")
    print(f"标注数量: {len(df)}")
    print(f"查找模式: {mode_label}")
    print(f"输出目录: {output_dir}")

    extractor = ResNetFeatureExtractor()
    processed = 0
    skipped_missing = 0
    skipped_exists = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
        number = str(row["Number"])
        translator = str(row["Translator"])

        video_path = find_video(data_dir, row)
        if video_path is None:
            tqdm.write(f"[警告] 找不到视频: Number={number}, Translator={translator}")
            skipped_missing += 1
            continue

        feature_name = f"{number}_{translator}.npy"
        feature_path = os.path.join(output_dir, feature_name)

        if os.path.exists(feature_path):
            skipped_exists += 1
            continue

        frames = load_video_frames(video_path)
        if len(frames) == 0:
            tqdm.write(f"[警告] 视频为空: {video_path}")
            skipped_missing += 1
            continue

        feats = extractor.extract(frames)
        np.save(feature_path, feats)
        processed += 1

    print(f"\n完成！处理 {processed} 个视频，"
          f"跳过 {skipped_exists} 个（已存在），"
          f"跳过 {skipped_missing} 个（缺失/空）")
    print(f"特征文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
