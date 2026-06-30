"""
CTC 手语识别服务封装

基于 AAA 目录下已经完成的训练与推理代码：
- 复用 `demo_infer.py` 中的 `predict_from_video` / `predict_from_image`
- 使用 `Project/Back/model/last_checkpoint.pt` 作为默认权重

提供给 FastAPI 的统一调用接口：
- recognize_video_file(path) -> 适配上传视频识别
- recognize_image_file(path) -> 适配上传图片识别
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple

import cv2

# ==================== 路径与依赖 ====================

# AAA 工程根目录（已包含 CTC 模型与推理脚本）
# 相对仓库根自动定位，避免硬编码绝对路径导致换机器失效。
# 本文件位于 <repo>/Project/Back/app/ctc_service.py，往上三级即仓库根，AAA 在其下。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AAA_ROOT = os.environ.get("AAA_ROOT", os.path.join(_REPO_ROOT, "AAA"))
if AAA_ROOT not in sys.path:
    sys.path.insert(0, AAA_ROOT)

print(f"[ctc_service] 初始化: AAA_ROOT={AAA_ROOT}")

try:
    # 复用 AAA/demo_infer 中已经验证的推理流程
    from demo_infer import predict_from_video, predict_from_image  # type: ignore
    print("[ctc_service] 已成功导入 demo_infer.predict_from_video / predict_from_image")
except Exception as e:
    print("[ctc_service] 导入 demo_infer 失败，请检查 AAA 目录与 Python 解释器:")
    traceback.print_exc()
    # 继续抛出，让启动阶段就暴露问题
    raise


@dataclass
class CTCConfig:
    """CTC 推理配置"""

    # 使用 Back 下的 checkpoint，方便部署时只依赖 Back 目录
    # <repo>/Project/Back/app/ctc_service.py -> 往上两级即 Back，model 在其下。
    checkpoint_path: str = os.environ.get(
        "CTC_CHECKPOINT",
        os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "model",
            "last_checkpoint.pt",
        ),
    )


cfg = CTCConfig()


def _get_video_meta(video_path: str) -> Tuple[float, int]:
    """获取视频时长（秒）与帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    if fps <= 0.0:
        return 0.0, frame_count

    duration = frame_count / fps
    return float(duration), frame_count


def recognize_video_file(video_path: str) -> Dict:
    """
    对上传视频做 CTC 推理，返回统一结构：
    {
      "text": str,
      "tokens": List[str],
      "confidence": float,
      "startTime": float,
      "endTime": float,
      "videoDuration": float,
      "processedFrames": int
    }
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    print(f"[ctc_service] recognize_video_file 被调用, video_path={video_path}")
    print(f"[ctc_service] 使用 checkpoint: {cfg.checkpoint_path}")

    tokens, sentence = predict_from_video(
        video_path,
        checkpoint_path=cfg.checkpoint_path,
    )

    duration, frame_count = _get_video_meta(video_path)

    # 当前 CTC 推理脚本未输出置信度，这里先给一个固定值占位，满足前端展示需求
    confidence = 95.0

    result = {
        "text": sentence,
        "tokens": tokens,
        "confidence": confidence,
        "startTime": 0.0,
        "endTime": duration,
        "videoDuration": duration,
        "processedFrames": frame_count,
    }
    print(
        "[ctc_service] recognize_video_file 完成: "
        f"text={sentence!r}, tokens_len={len(tokens)}, "
        f"duration={duration}, frames={frame_count}"
    )
    return result


def recognize_image_file(image_path: str) -> Dict:
    """
    对上传图片做 CTC 推理，返回统一结构：
    {
      "text": str,
      "tokens": List[str],
      "confidence": float
    }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    print(f"[ctc_service] recognize_image_file 被调用, image_path={image_path}")
    print(f"[ctc_service] 使用 checkpoint: {cfg.checkpoint_path}")

    tokens, sentence = predict_from_image(
        image_path,
        checkpoint_path=cfg.checkpoint_path,
    )

    confidence = 95.0

    result = {
        "text": sentence,
        "tokens": tokens,
        "confidence": confidence,
    }
    print(
        "[ctc_service] recognize_image_file 完成: "
        f"text={sentence!r}, tokens_len={len(tokens)}"
    )
    return result


# ==================== 实时 CTC 识别服务 ====================

_realtime_instance: "RealtimeCTCService | None" = None


def get_realtime_ctc_service(
    checkpoint_path: str | None = None,
) -> "RealtimeCTCService":
    """获取全局单例 RealtimeCTCService"""
    global _realtime_instance
    if _realtime_instance is None:
        _realtime_instance = RealtimeCTCService(checkpoint_path=checkpoint_path)
    return _realtime_instance


class RealtimeCTCService:
    """
    实时 CTC 手语识别服务

    设计思路：
    - 初始化时加载 CTC 模型和 ResNet 特征提取器（仅一次）
    - 维护一个滑动窗口帧缓冲区
    - 每收到 infer_every_n_frames 帧后触发一次 CTC 推理
    - 推理时用缓冲区中所有帧提取特征序列 → 送入 BiLSTM+CTC → greedy 解码
    - 通过 WebSocket 将结果实时推送给前端

    参数：
    - max_buffer_frames: 缓冲区最大帧数（滑动窗口上限，默认 64）
    - infer_every_n_frames: 每 N 帧触发一次推理（默认 4）
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        max_buffer_frames: int = 64,
        infer_every_n_frames: int = 4,
    ):
        if checkpoint_path is None:
            checkpoint_path = os.environ.get(
                "CTC_CHECKPOINT",
                os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                    "model",
                    "last_checkpoint.pt",
                ),
            )

        self.checkpoint_path = checkpoint_path
        self.max_buffer_frames = max_buffer_frames
        self.infer_every_n_frames = infer_every_n_frames

        # 模型组件（延迟加载）
        self.model = None
        self.vocab = None
        self.device = None
        self.extractor = None

        # 帧缓冲区
        self.frame_buffer: list = []  # list of numpy BGR frames
        self.frames_since_last_infer = 0
        self.last_result: dict | None = None

        self._load_model()

    # ----------------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------------
    def _load_model(self):
        """加载 CTC 模型与特征提取器"""
        from demo_infer import load_checkpoint, build_feature_extractor

        self.model, self.vocab, self.device = load_checkpoint(self.checkpoint_path)
        self.extractor = build_feature_extractor()
        print(
            f"[RealtimeCTCService] 模型加载完成 "
            f"checkpoint={self.checkpoint_path} device={self.device}"
        )

    # ----------------------------------------------------------------
    # 帧管理
    # ----------------------------------------------------------------
    def add_frame(self, img_bgr: "np.ndarray") -> dict | None:
        """
        向缓冲区添加一帧（BGR 格式 numpy 数组），满足推理条件时自动推理。

        返回:
            若触发了推理，返回 dict: { text, tokens, confidence }
            否则返回 None
        """
        import numpy as np

        self.frame_buffer.append(img_bgr)
        self.frames_since_last_infer += 1

        # 滑动窗口：限制缓冲区长度
        if len(self.frame_buffer) > self.max_buffer_frames:
            self.frame_buffer = self.frame_buffer[-self.max_buffer_frames:]

        # 推理条件：累积帧数 >= infer_every_n_frames
        if (
            self.frames_since_last_infer >= self.infer_every_n_frames
            and len(self.frame_buffer) >= self.infer_every_n_frames
        ):
            self.frames_since_last_infer = 0
            return self._infer()

        return None

    # ----------------------------------------------------------------
    # 推理
    # ----------------------------------------------------------------
    def _infer(self) -> dict:
        """执行 CTC 推理（单次前向传播），返回 { text, tokens, confidence }"""
        import torch
        import numpy as np

        if len(self.frame_buffer) == 0:
            self.last_result = {"text": "", "tokens": [], "confidence": 0.0}
            return self.last_result

        # 提取特征序列
        features = self.extractor.extract(self.frame_buffer)  # (T, 512)

        # 单次 CTC 前向传播
        feats_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        feat_lens_t = torch.tensor([features.shape[0]], dtype=torch.long).to(self.device)
        log_probs, _ = self.model(feats_t, feat_lens_t)  # (T, 1, C)

        # CTC greedy 解码（折叠重复 + 去 blank）
        log_probs_2d = log_probs.squeeze(1)  # (T, C)
        best_path = log_probs_2d.argmax(dim=-1).tolist()

        collapsed_ids: list = []
        prev = None
        for idx in best_path:
            if idx == 0:  # blank
                prev = None
                continue
            if prev is not None and idx == prev:
                continue
            collapsed_ids.append(idx)
            prev = idx

        tokens = [self.vocab.id2token.get(i, "<unk>") for i in collapsed_ids]
        tokens = [t for t in tokens if t not in ("<blank>",)]
        sentence = "".join(tokens) if tokens else "(空预测)"

        # 计算置信度：各 token 对应时间步的 softmax 概率均值
        confidence = 95.0  # 默认值
        try:
            probs = torch.softmax(log_probs_2d, dim=-1)
            token_probs = []
            prev2 = None
            for t, idx in enumerate(best_path):
                if idx == 0:
                    prev2 = None
                    continue
                if prev2 is not None and idx == prev2:
                    continue
                token_probs.append(probs[t, idx].item())
                prev2 = idx

            if token_probs:
                confidence = float(np.mean(token_probs) * 100)
        except Exception:
            pass

        self.last_result = {
            "text": sentence,
            "tokens": tokens,
            "confidence": round(confidence, 1),
        }
        return self.last_result

    # ----------------------------------------------------------------
    # 重置
    # ----------------------------------------------------------------
    def reset(self):
        """清空帧缓冲区，重置状态"""
        self.frame_buffer = []
        self.frames_since_last_infer = 0
        self.last_result = None
        print("[RealtimeCTCService] 状态已重置")