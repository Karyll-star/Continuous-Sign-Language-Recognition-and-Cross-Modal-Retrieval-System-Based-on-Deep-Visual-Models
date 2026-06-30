# 实时翻译 — LLM 后处理简化方案

> 版本：v2.0  
> 日期：2026-06-30  
> 前提：WebSocket 实时管道已运行，CTC 输出准确度不足  
> 目标：用云端 LLM 过滤纠错，提升最终结果质量  
> 原则：**新功能集中到 `enhancer/`，现有文件最小改动，保留扩展点**

---

## 目录

- [一、核心思路](#一核心思路)
- [二、文件布局](#二文件布局)
- [三、改动清单](#三改动清单)
- [四、enhancer/token_buffer.py — 词级上下文](#四enhancertoken_bufferpy--词级上下文)
- [五、enhancer/llm_prompts.py — Prompt 模板](#五enhancerllm_promptspy--prompt-模板)
- [六、enhancer/llm_gateway.py — LLM 调用层](#六enhancerllm_gatewaypy--llm-调用层)
- [七、enhancer/__init__.py — 对外入口](#七enhancer__init__py--对外入口)
- [八、main.py 集成](#八mainpy-集成)
- [九、前端改动](#九前端改动)
- [十、降级与容错](#十降级与容错)
- [十一、扩展路线图](#十一扩展路线图)
- [附录：环境变量](#附录环境变量)

---

## 一、核心思路

```
                        enhancer/ 目录 (所有新代码)
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  CTC 输出                     llm_gateway.py                  │
│  raw_token ──▶ token_buffer ──▶ LLMGateway.enhance()         │
│                   │                    │                       │
│                   │           ┌────────┴────────┐             │
│                   │           │ AliyunGateway   │  (当前)     │
│                   │           │ Qwen-Plus API   │             │
│                   │           ├─────────────────┤             │
│                   │           │ OllamaGateway   │  (预留)     │
│                   │           │ 本地 Qwen2.5    │             │
│                   │           └────────┬────────┘             │
│                   │                    │                       │
│              快通道 (不变)      慢通道 (sentence)              │
│              type:result       type:sentence                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘

现有文件改动：main.py 仅加 1 个 import + 1 行函数调用
```

---

## 二、文件布局

```
enhancer/                          ← 所有新代码放在这里
├── __init__.py                    ← 对外导出 enhance_result()
├── token_buffer.py                ← 词级上下文累积
├── llm_prompts.py                 ← Prompt 模板
└── llm_gateway.py                 ← LLM 调用层 (抽象 + 百炼实现 + 本地预留)

Project/Back/app/main.py           ← 仅 2 处修改 (import + 调用)
Project/Front/pages/recognize.vue   ← 处理新增 sentence 消息
```

---

## 三、改动清单

| 文件 | 改动 | 行数 |
|------|------|------|
| `enhancer/__init__.py` | **新建** — 对外入口 `enhance_result()` | ~15 行 |
| `enhancer/token_buffer.py` | **新建** — TokenBuffer 类 | ~40 行 |
| `enhancer/llm_prompts.py` | **新建** — Prompt 模板 | ~20 行 |
| `enhancer/llm_gateway.py` | **新建** — LLMGateway 抽象 + 百炼实现 | ~70 行 |
| `Project/Back/app/main.py` | **修改** — 1 个 import + 1 行调用 | +3 行 |
| `Project/Front/pages/recognize.vue` | **修改** — 处理 `sentence` 消息 | +10 行 |

**总计：约 155 行新代码分布在新目录 `enhancer/` 下，现有文件改动 < 15 行。**

---

## 四、enhancer/token_buffer.py — 词级上下文

```python
"""词级上下文缓冲区 — 滑动窗口 + 可替换触发策略"""

import time
from dataclasses import dataclass, field
from typing import List, Callable


@dataclass
class TokenEntry:
    text: str
    confidence: float   # 0-100
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 触发策略 (可替换)
# ============================================================

def count_trigger(every_n: int = 3) -> Callable:
    """简单计数策略：每 N 个新 token 触发一次 (当前默认)"""
    sent = 0

    def should_trigger(buffer: "TokenBuffer") -> bool:
        nonlocal sent
        total = len(buffer.tokens)
        if total < 2:
            return False
        if total - sent >= every_n:
            sent = total
            return True
        return False

    return should_trigger


# 预留: 后续可替换为状态机策略
# def state_machine_trigger(silence_sec=1.5, confidence_drop=40):
#     ...


# ============================================================
# TokenBuffer
# ============================================================

@dataclass
class TokenBuffer:
    """滑动窗口维护最近 N 个 CTC token，按策略触发 LLM"""

    max_size: int = 20
    tokens: List[TokenEntry] = field(default_factory=list)
    last_sentence: str = ""

    # 触发策略 — 替换这里即可切换策略
    _trigger: Callable = field(default_factory=lambda: count_trigger(3))

    def add(self, text: str, confidence: float) -> TokenEntry:
        entry = TokenEntry(text=text, confidence=confidence)
        self.tokens.append(entry)
        if len(self.tokens) > self.max_size:
            self.tokens = self.tokens[-self.max_size:]
        return entry

    def should_enhance(self) -> bool:
        return self._trigger(self)

    def get_context(self) -> str:
        return " ".join(t.text for t in self.tokens)

    def reset(self):
        self.tokens.clear()
        self.last_sentence = ""
        self._trigger = count_trigger(3)  # 重新创建闭包
```

**扩展点：** 替换 `_trigger` 即可从简单计数切换到状态机策略，不修改调用方。

---

## 五、enhancer/llm_prompts.py — Prompt 模板

```python
"""LLM Prompt 模板 — 集中管理，方便调优"""

SYSTEM_PROMPT = """\
你是手语翻译的后处理助手。你会收到一组手语识别系统输出的词语序列\
（可能包含识别错误、重复、乱序），请做以下处理：
1. 纠正明显错误的识别结果（如"尺饭"→"吃饭"、"窝"→"我"）
2. 将碎片词语组装成自然流畅的中文句子
3. 去除无意义的重复词

如果输入为空或全部是乱码/无法理解，直接返回空字符串。

只返回修正后的句子，不要解释、不要前缀。"""


def build_user_prompt(raw_tokens: str) -> str:
    return f"原始识别序列: {raw_tokens}\n请输出修正后的完整句子:"
```

---

## 六、enhancer/llm_gateway.py — LLM 调用层

```python
"""
LLM 调用层 — 抽象接口 + 阿里百炼实现 + 本地 Ollama 预留
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
import httpx

from .llm_prompts import SYSTEM_PROMPT, build_user_prompt


# ============================================================
# 配置 (全部通过环境变量)
# ============================================================

LLM_ENABLED = os.getenv("LLM_ENABLED", "1") == "1"
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "3.0"))


# ============================================================
# 抽象接口
# ============================================================

class LLMGateway(ABC):
    """LLM 后处理抽象网关 — 后续新增后端只需实现此接口"""

    @abstractmethod
    async def enhance(self, raw_tokens: str) -> str:
        """输入原始 CTC 序列，返回纠错成句后的文本"""
        ...


# ============================================================
# 阿里百炼 (Qwen-Plus) — 当前默认实现
# ============================================================

class AliyunGateway(LLMGateway):
    """
    阿里百炼平台 (DashScope) Qwen-Plus 模型

    环境变量:
        DASHSCOPE_API_KEY    API Key
        DASHSCOPE_MODEL      模型名 (默认 qwen-plus)
    """

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")

    async def enhance(self, raw_tokens: str) -> str:
        if not self.api_key:
            return ""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(raw_tokens)},
            ],
            "max_tokens": 100,
            "temperature": 0.1,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                resp = await client.post(self.BASE_URL, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[llm_gateway] API {resp.status_code}: {resp.text[:200]}")
                    return ""
        except Exception as e:
            print(f"[llm_gateway] 异常: {e}")
            return ""


# ============================================================
# 本地 Ollama — 预留实现 (后续启用)
# ============================================================

class OllamaGateway(LLMGateway):
    """
    本地 Ollama Qwen2.5-1.5B 模型 — 离线/低延迟场景

    环境变量:
        OLLAMA_BASE_URL    Ollama 地址 (默认 http://localhost:11434)
        OLLAMA_MODEL       模型名 (默认 qwen2.5:1.5b)
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    async def enhance(self, raw_tokens: str) -> str:
        # TODO: 后续实现
        return ""


# ============================================================
# 工厂函数 — 根据环境变量选择后端
# ============================================================

_gateway: Optional[LLMGateway] = None


def get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is not None:
        return _gateway

    backend = os.getenv("LLM_BACKEND", "aliyun").lower()

    if backend == "ollama":
        _gateway = OllamaGateway()
    else:
        _gateway = AliyunGateway()

    print(f"[llm_gateway] 后端: {_gateway.__class__.__name__}")
    return _gateway
```

**扩展点：**
- 新增后端只需实现 `LLMGateway` 抽象类
- 通过 `LLM_BACKEND=ollama` 环境变量切换
- `LLMGateway.enhance()` 可扩展为返回 `EnhanceResult`（含置信度、纠错详情等）

---

## 七、enhancer/__init__.py — 对外入口

```python
"""
enhancer — LLM 后处理模块

对外唯一入口: enhance_result()
main.py 只需 import 这个函数即可。
"""

from .token_buffer import TokenBuffer
from .llm_gateway import LLM_ENABLED, get_gateway


async def enhance_result(
    websocket,             # FastAPI WebSocket
    buffer: TokenBuffer,  # 词级上下文缓冲区
    text: str,            # CTC 最新输出
    confidence: float,    # CTC 置信度
):
    """LLM 后处理主流程 — 累积 token → 条件触发 → 异步推送句级结果"""

    if not LLM_ENABLED:
        return

    buffer.add(text, confidence)

    if not buffer.should_enhance():
        return

    raw_context = buffer.get_context()
    gateway = get_gateway()
    corrected = await gateway.enhance(raw_context)

    if corrected and corrected != buffer.last_sentence:
        buffer.last_sentence = corrected
        import time
        await websocket.send_json({
            "type": "sentence",
            "data": {
                "text": corrected,
                "isComplete": False,
            },
            "timestamp": int(time.time() * 1000),
        })
```

**设计意图：** main.py 只需知道 `enhance_result` 一个函数，所有 LLM 内部细节（TokenBuffer、Gateway、Prompts）全部隐藏在 `enhancer/` 下。

---

## 八、main.py 集成

**唯一修改点** — WebSocket handler 中 CTC 推理后：

```python
# ===== main.py 顶部新增 1 行 import =====
from enhancer import enhance_result, TokenBuffer

# ===== WebSocket handler 内部 =====
@app.websocket("/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()
    service = get_realtime_ctc_service()

    buffer = TokenBuffer()          # ← 新增: 每条连接独立缓冲区

    try:
        while True:
            # ... 现有帧处理代码不变 ...

            if result and result.get("text"):
                text = result["text"]
                confidence = result.get("confidence", 0)

                # === 词级快通道 (不变) ===
                info = lookup_text_info(text)
                await websocket.send_json({
                    "type": "result",
                    "data": {
                        "text": text,
                        "pinyin": info.get("pinyin", ""),
                        "meaning": info.get("meaning", ""),
                        "confidence": confidence,
                    },
                    "timestamp": int(time.time() * 1000),
                })

                # === LLM 后处理 (新增 1 行) ===
                await enhance_result(websocket, buffer, text, confidence)

    except WebSocketDisconnect:
        print("[ws] 断开")
```

**总结：main.py 改动 = 1 个 import + 1 个 TokenBuffer 声明 + 1 行 `await enhance_result()`。**

---

## 九、前端改动

在 `useWebSocket.ts` 的 `onmessage` 中处理新增的 `sentence` 消息类型：

```typescript
// useWebSocket.ts — onmessage 中新增:

ws.value.onmessage = (event) => {
    const data = JSON.parse(event.data)

    // ... 原有 result 处理不变 ...

    // === 新增: 句级结果 ===
    if (data.type === 'sentence') {
        // 存入 store 供 UI 展示
        recognitionStore.setSentenceResult({
            text: data.data.text,
            isComplete: data.data.isComplete,
            timestamp: data.timestamp,
        })
    }
}
```

`recognitionStore` 中新增状态：

```typescript
// stores/recognition.ts 新增:

const sentenceResult = ref<{ text: string; isComplete: boolean } | null>(null)

function setSentenceResult(result: { text: string; isComplete: boolean }) {
    sentenceResult.value = result
}
```

`recognize.vue` 模板中展示句级结果（在词级展示下方新增一行）：

```html
<!-- 句级 LLM 修正结果 -->
<div v-if="recognitionStore.sentenceResult" class="mt-3 p-3 bg-blue-50 rounded-xl">
    <span class="text-xs text-blue-500 mr-2">AI 修正</span>
    <span class="text-lg font-medium text-gray-900">
        {{ recognitionStore.sentenceResult.text }}
    </span>
</div>
```

---

## 十、降级与容错

| 场景 | 行为 |
|------|------|
| `LLM_ENABLED=0` | `enhance_result()` 第一行直接 return |
| `DASHSCOPE_API_KEY` 未设置 | `AliyunGateway.enhance()` 返回 `""` |
| API 超时 (3s) | 返回 `""`，词级正常 |
| API 返回非 200 | 返回 `""`，日志打印错误 |
| LLM 返回空字符串 | `if corrected` 拦截，不推送 |
| LLM 返回与上次相同 | `last_sentence` 去重，不推送 |
| 网络不通 | `httpx` 抛异常 → 返回 `""` |

**核心原则：LLM 层是"增强"而非"必需"。任何环节失败 → 静默跳过 → 词级快通道照常工作。**

---

## 十一、扩展路线图

本方案预留了三个清晰的扩展点，渐进式升级：

### 11.1 触发策略升级 (token_buffer.py)

```
当前: count_trigger(3)         ← 简单计数
  ↓ 替换 _trigger 即可
未来: state_machine_trigger()  ← 静默超时 + 置信度骤降
```

**改动范围：** 仅 `token_buffer.py` 一个文件，新增一个触发函数。

### 11.2 本地 LLM 接入 (llm_gateway.py)

```
当前: AliyunGateway → Qwen-Plus (云端)
  ↓ 设置 LLM_BACKEND=ollama
未来: OllamaGateway → Qwen2.5:1.5b (本地)
```

**改动范围：** `OllamaGateway` 类已预留框架，只需实现 `enhance()` 方法即可。

### 11.3 拼音/释义自动补全 (llm_gateway.py)

```
当前: lookup_text_info() 词典查表 (100 条)
  ↓ 扩展 LLMGateway 接口
未来: LLMGateway.enrich()  自动生成拼音 + 释义
```

**改动范围：** `LLMGateway` 抽象类加一个 `enrich()` 方法，各后端实现。

### 11.4 完整路线

```
Phase 1 (本方案)           Phase 2                 Phase 3
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ TokenBuffer     │    │ 状态机触发       │    │ 本地 Ollama     │
│ AliyunGateway   │───▶│ 静默超时检测     │───▶│ 离线可用        │
│ 每 3 token 触发 │    │ 置信度骤降检测   │    │ 低延迟 (<200ms) │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
     0.5-1 天               0.5 天                  1 天
```

---

## 附录：环境变量

```bash
# ===== 总开关 =====
LLM_ENABLED=1                     # 1=开启  0=关闭 (关闭时零影响)

# ===== LLM 后端选择 =====
LLM_BACKEND=aliyun                # aliyun | ollama

# ===== 阿里百炼 (DashScope) =====
DASHSCOPE_API_KEY=sk-xxxxxxxx     # 百炼 API Key
DASHSCOPE_MODEL=qwen-plus         # qwen-plus | qwen-max | qwen-turbo

# ===== 本地 Ollama (预留) =====
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:1.5b

# ===== 性能 =====
LLM_TIMEOUT=3.0                   # API 超时秒数
```
