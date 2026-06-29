"""
从已有的 metadata_dev.json 生成 embeddings_dev.npy 的快速脚本

用途：当没有 CE-CSL 视频数据集时，利用 metadata 中已有的句子文本
      重新编码句向量，使 video_rag 检索接口可以正常工作。

使用：
    cd Project/Back
    python scripts/rebuild_embeddings_from_meta.py
"""
import json
import os
import sys

import numpy as np

# 工作目录设为脚本所在目录的上一级（即 Project/Back）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

INDEX_DIR = os.path.join(PROJECT_DIR, "data", "cecsl_index")
META_PATH = os.path.join(INDEX_DIR, "metadata_dev.json")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings_dev.npy")


def main():
    if not os.path.exists(META_PATH):
        print(f"[错误] 元数据文件不存在: {META_PATH}")
        sys.exit(1)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    samples = meta.get("samples", [])
    model_name = meta.get("model_name", "BAAI/bge-small-zh-v1.5")

    sentences = [s["sentence"] for s in samples]
    print(f"[rebuild] 从 metadata 中读取到 {len(sentences)} 条句子")
    print(f"[rebuild] 句向量模型: {model_name}")

    # 加载模型
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[错误] 请先安装 sentence-transformers: pip install sentence-transformers")
        sys.exit(1)

    print("[rebuild] 正在加载模型（首次可能需下载 ~100MB）...")
    model = SentenceTransformer(model_name)

    print("[rebuild] 正在编码句子...")
    embeddings = model.encode(
        sentences,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"[rebuild] 向量形状: {embeddings.shape}")
    np.save(EMB_PATH, embeddings)
    print(f"[rebuild] ✅ 已保存到: {EMB_PATH}")

    # 验证
    loaded = np.load(EMB_PATH)
    assert loaded.shape == embeddings.shape, "保存验证失败！"
    print(f"[rebuild] ✅ 验证通过: {loaded.shape[0]} 条向量, 维度={loaded.shape[1]}")


if __name__ == "__main__":
    main()
