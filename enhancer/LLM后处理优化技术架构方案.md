# 手语识别 LLM 后处理优化 — 技术架构方案

> 版本：v1.0  
> 日期：2026-06-30  
> 目标：在现有实时 CTC 识别管道上叠加 LLM 后处理层，提升输出质量与成句能力

---

## 一、现状分析与问题定位

### 1.1 当前管道

```
摄像头帧 (250ms/帧)
  → WebSocket → RealtimeCTCService (滑动窗口 64 帧, 每 4 帧推理)
  → CTC Greedy 解码 → 字符序列 "你好"
  → lookup_text_info() 词典查表 → 补充 pinyin/meaning
  → WebSocket → 前端展示
```

### 1.2 核心瓶颈

| 瓶颈 | 表现 | 根因 |
|------|------|------|
| **无语境累积** | 每次推理独立输出，前后结果无关联 | 帧缓冲区只维护视觉上下文，不维护语义上下文 |
| **无语法修正** | 输出为原始 token 拼接，如 `"我想尺饭"` 若模型误识别 `"吃"` 为 `"尺"` | CTC greedy 解码无语言模型约束 |
| **无成句能力** | 连续手语打出的多个词被割裂为独立推送，前端收到的是碎片 | 管道中没有句子边界检测和组装机制 |
| **词典局限** | 仅 100 条精确匹配，未覆盖则拼音/释义为空 | 查表无模糊匹配、无语义扩展 |

### 1.3 优化目标

1. **纠错**：修正 CTC 模型误识别（同音字、形近手势）
2. **成句**：将碎片化词语序列组装为自然流畅的中文句子
3. **补全**：为未登录词自动生成拼音和释义
4. **可降级**：LLM 不可用时自动回退到现有管道

---

## 二、总体架构

### 2.1 架构全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          用户浏览器 (Nuxt 3)                              │
│                                                                          │
│  摄像头 ──▶ useCamera ──▶ useWebSocket ──▶ WebSocket                    │
│                                               │                          │
│  结果面板 ◀── recognitionStore ◀── onmessage ◀─┘                         │
│  ┌─────────────────────┐                                                 │
│  │ 词级结果 (高频快显)  │  text / pinyin / meaning / confidence           │
│  │ 句级结果 (低频刷新)  │  sentence / words[] / isComplete               │
│  └─────────────────────┘                                                 │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │  ws://localhost:8000/recognize
┌────────────────────────────────────┼─────────────────────────────────────┐
│                        FastAPI 后端 (:8000)                               │
│                                     │                                     │
│  ┌──────────────────────────────────▼──────────────────────────────────┐ │
│  │                    WebSocket Handler (main.py)                       │ │
│  │                                                                      │ │
│  │  frame ──▶ service.add_frame(img) ──▶ raw_result ──┐                │ │
│  │                                                     │                │ │
│  │  ┌──────────────────────────────────────────────────┤                │ │
│  │  │                                                  │                │ │
│  │  │  ┌─────────────────────┐    ┌────────────────────▼──────────┐    │ │
│  │  │  │  快通道 (不变)       │    │  慢通道 (新增 LLM 后处理)      │    │ │
│  │  │  │                     │    │                               │    │ │
│  │  │  │ lookup_text_info()  │    │  SentenceAccumulator          │    │ │
│  │  │  │ → 词级结果           │    │   .add_token(text, conf)      │    │ │
│  │  │  │ → send_json(result) │    │   → 句子边界检测               │    │ │
│  │  │  │                     │    │   → LLMPipeline.process()     │    │ │
│  │  │  └─────────┬───────────┘    │   → 句级结果                   │    │ │
│  │  │            │                │   → send_json(sentence)       │    │ │
│  │  └────────────┼────────────────┼───────────────────────────────┘    │ │
│  │               │                │                                     │ │
│  │               ▼                ▼                                     │ │
│  │         前端收到 type:"result"  前端收到 type:"sentence"              │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                      LLM Pipeline (llm_pipeline.py)                   │ │
│  │                                                                       │ │
│  │  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐ │ │
│  │  │ Stage 1     │    │ Stage 2      │    │ Stage 3                  │ │ │
│  │  │ 词语纠错    │───▶│ 句子组装     │───▶│ 拼音/释义自动补全        │ │ │
│  │  │ (轻量 LLM)  │    │ (标准 LLM)   │    │ (词典 + LLM 混合)        │ │ │
│  │  └─────────────┘    └──────────────┘    └──────────────────────────┘ │ │
│  │                                                                       │ │
│  │  ┌──────────────────────────────────────────────────────────────┐    │ │
│  │  │                    LLM Gateway (抽象层)                        │    │ │
│  │  │                                                               │    │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐               │    │ │
│  │  │  │ Local    │  │ Cloud    │  │ Hybrid        │               │    │ │
│  │  │  │ Ollama   │  │ DeepSeek │  │ Local纠错     │               │    │ │
│  │  │  │ Qwen2.5  │  │ API      │  │ +Cloud成句    │               │    │ │
│  │  │  └──────────┘  └──────────┘  └───────────────┘               │    │ │
│  │  └──────────────────────────────────────────────────────────────┘    │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 双通道设计原理

| 维度 | 快通道（词级） | 慢通道（句级） |
|------|---------------|---------------|
| **触发频率** | 每次 CTC 推理 (约 1s/次) | 检测到句子边界时 (约 3-10s/次) |
| **延迟目标** | < 50ms（纯查表） | < 500ms-2000ms（取决于 LLM） |
| **输出内容** | `{ text, pinyin, meaning, confidence }` | `{ sentence, words[], isComplete }` |
| **LLM 参与** | 可选 Stage-1 快速纠错 | Stage-2 句子组装 |
| **降级行为** | 查表失败则留空 | LLM 不可用则退回原始拼接 |

---

## 三、核心组件设计

### 3.1 SentenceAccumulator（句子累积器）

**职责**：维护语义上下文，检测句子边界，触发 LLM 处理

**状态机**：

```
                    ┌─────────┐
     start/stop ──▶│  IDLE   │◀── reset()
                    └────┬────┘
                         │ add_token()
                    ┌────▼────┐
                    │ACTIVE   │ 累积 tokens + 计时
                    └────┬────┘
                         │ 触发条件满足
                    ┌────▼────┐
                    │FLUSHING │ 调用 LLM Pipeline
                    └────┬────┘
                         │ LLM 返回
                    ┌────▼────┐
                    │ACTIVE   │ 继续累积（保留最近 N 个词作为上下文窗口）
                    └─────────┘
```

**句子边界检测策略（三选一，可配置）：**

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **静默超时** | 连续 N 秒无新 token 输入 → 认为句子结束 | 简单可靠，适合慢速手语 |
| **置信度骤降** | 当前帧置信度 < 阈值且前一帧置信度高 → 句子边界 | 适合连续手语中自然停顿 |
| **LLM 判断** | 将累积序列送入轻量 LLM 询问"是否完整句子" | 最准确但增加延迟 |

**推荐组合**：静默超时（1.5s）+ 置信度骤降（< 40%）任一条触发。

**核心数据结构：**

```python
@dataclass
class AccumulatorState:
    tokens: List[TokenEntry]      # 累积的识别 token
    last_token_time: float        # 最后一个 token 的时间戳
    sentence_start_time: float    # 当前句子开始时间
    context_window: List[TokenEntry]  # 上下文窗口（最近 N 个 token）

@dataclass
class TokenEntry:
    text: str                     # CTC 原始输出 "你好"
    confidence: float             # 置信度 0-100
    pinyin: str                   # 查表拼音
    meaning: str                  # 查表释义
    timestamp: float              # 到达时间
```

### 3.2 LLM Gateway（LLM 抽象网关）

**职责**：统一抽象本地/云端 LLM，支持热切换和自动降级

```python
class LLMGateway(ABC):
    """LLM 抽象网关"""

    @abstractmethod
    async def correct_word(self, text: str, context: List[str]) -> CorrectResult:
        """Stage 1: 词语纠错（轻量）"""
        ...

    @abstractmethod
    async def form_sentence(self, tokens: List[TokenEntry]) -> SentenceResult:
        """Stage 2: 句子组装（核心）"""
        ...

    @abstractmethod
    async def enrich_word(self, text: str) -> EnrichResult:
        """Stage 3: 词语补全（拼音/释义）"""
        ...

class LocalLLMGateway(LLMGateway):
    """Ollama 本地模型实现"""
    # 使用 Qwen2.5-1.5B/7B via Ollama

class CloudLLMGateway(LLMGateway):
    """云端 API 实现"""
    # 使用 DeepSeek/OpenAI API

class HybridLLMGateway(LLMGateway):
    """混合模式：本地纠错 + 云端成句"""
    # 延迟敏感阶段用本地，质量敏感阶段用云端
```

### 3.3 LLM Pipeline（LLM 处理管道）

**职责**：编排多阶段处理流程

```
输入: List[TokenEntry]  例如: [("我", 92%), ("想", 88%), ("尺", 45%), ("饭", 90%)]
                                    ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: 词语纠错（可选，约 100-300ms）                       │
│                                                              │
│ 对每个低置信度 token 进行上下文感知纠错:                        │
│   "尺" (conf=45%, context=["我","想","?","饭"])                │
│   → LLM 判断 → "吃" (confidence 提升至 85%)                   │
│                                                              │
│ 实现: 仅对 confidence < 阈值(默认60%) 的 token 调用 LLM       │
│       高置信度 token 直接通过，减少 API 调用                     │
└──────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: 句子组装（核心，约 300-800ms）                       │
│                                                              │
│ 输入: ["我", "想", "吃", "饭"]                                │
│ 输出: {                                                      │
│   "sentence": "我想吃饭",                                     │
│   "isComplete": true,                                        │
│   "corrections": [{"original":"尺","corrected":"吃"}],       │
│   "confidence": 88.5                                         │
│ }                                                            │
│                                                              │
│ 同时完成:                                                     │
│   - 语法修正（"我吃饭想" → "我想吃饭"）                        │
│   - 标点插入（自动判断句号/问号）                              │
│   - 口语化润色（"我想去吃饭"）                                 │
└──────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: 拼音释义补全（约 50-100ms）                          │
│                                                              │
│ 对句子中每个词语:                                              │
│   词典命中 → 使用词典数据                                      │
│   词典未命中 → LLM 生成 pinyin + meaning                       │
│                                                              │
│ 输出: words[] 中每个词附带 pinyin, meaning                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 四、WebSocket 协议扩展

### 4.1 新增消息类型

在现有协议基础上增加两条下行消息：

```jsonc
// === 现有消息（不变） ===

// 服务端 → 客户端：词级结果（快通道，高频）
{
  "type": "result",
  "data": {
    "text": "你好",
    "pinyin": "nǐ hǎo",
    "meaning": "表示问候或打招呼",
    "confidence": 92.5
  },
  "timestamp": 1719734400000
}

// === 新增消息 ===

// 服务端 → 客户端：句级结果（慢通道，低频）
{
  "type": "sentence",
  "data": {
    "sentence": "你好，我想吃饭",          // 完整句子
    "isComplete": true,                    // 是否为完整句子 (false=实时碎片)
    "words": [                             // 每个词的详细信息
      { "text": "你好", "pinyin": "nǐ hǎo", "meaning": "hello", "confidence": 92.5 },
      { "text": "我",   "pinyin": "wǒ",    "meaning": "I/me",   "confidence": 95.0 },
      { "text": "想",   "pinyin": "xiǎng", "meaning": "want",   "confidence": 88.0 },
      { "text": "吃",   "pinyin": "chī",   "meaning": "eat",    "confidence": 85.0, "corrected": true, "original": "尺" },
      { "text": "饭",   "pinyin": "fàn",   "meaning": "meal",   "confidence": 90.0 }
    ],
    "corrections": [                       // 纠错记录
      { "position": 3, "original": "尺", "corrected": "吃", "reason": "上下文语义推断" }
    ],
    "confidence": 88.5,                    // 整句加权置信度
    "sessionId": "abc123"                  // 会话 ID，前端用于关联
  },
  "timestamp": 1719734415000
}

// 服务端 → 客户端：LLM 状态通知（可选）
{
  "type": "llm_status",
  "data": {
    "status": "processing",               // "idle" | "processing" | "error" | "degraded"
    "stage": "sentence_formation",        // 当前处理阶段
    "message": "正在生成完整句子..."
  },
  "timestamp": 1719734410000
}
```

### 4.2 时序示意

```
客户端                          服务端
  │                               │
  │──── start ───────────────────▶│ reset() → SentenceAccumulator 重置
  │                               │
  │──── frame ───────────────────▶│ CTC → "你"
  │◀── result {text:"你"} ──────│ (快通道，约 10ms)
  │                               │ Accumulator.add_token("你")
  │                               │
  │──── frame ───────────────────▶│ CTC → "好"
  │◀── result {text:"好"} ──────│ (快通道)
  │                               │ Accumulator.add_token("好")
  │                               │
  │──── frame ───────────────────▶│ CTC → "我"
  │◀── result {text:"我"} ──────│
  │                               │
  │──── frame ───────────────────▶│ CTC → "想"
  │◀── result {text:"想"} ──────│
  │                               │
  │       ... 用户停顿 1.5s ...    │
  │                               │ 静默超时触发！
  │                               │ Accumulator.flush()
  │                               │ LLMPipeline.process(["你","好","我","想"])
  │                               │ → "你好，我想……" (isComplete=false)
  │◀── sentence {...} ──────────│ (慢通道，约 500ms)
  │                               │
  │──── frame ───────────────────▶│ CTC → "吃"
  │◀── result {text:"吃"} ──────│
  │                               │
  │──── frame ───────────────────▶│ CTC → "饭"
  │◀── result {text:"饭"} ──────│
  │                               │
  │       ... 用户再次停顿 ...     │
  │                               │ LLMPipeline.process(["我","想","吃","饭"])
  │                               │ → "我想吃饭。" (isComplete=true)
  │◀── sentence {...} ──────────│
  │                               │
  │──── stop ────────────────────▶│ reset() → 清空 Accumulator
```

---

## 五、模型选型方案

### 5.1 推荐配置矩阵

| 部署模式 | 纠错模型 (Stage 1) | 成句模型 (Stage 2) | 适用场景 | 硬件要求 |
|---------|-------------------|-------------------|---------|---------|
| **纯云端** | DeepSeek-V3 (轻量 prompt) | DeepSeek-V3 (标准 prompt) | 快速上线、无 GPU | 无，需网络 |
| **纯本地** | Qwen2.5-1.5B (Ollama) | Qwen2.5-7B (Ollama) | 离线场景、隐私敏感 | ≥8GB VRAM |
| **混合推荐** | Qwen2.5-1.5B 本地 | DeepSeek API 云端 | 平衡延迟与质量 | ≥4GB VRAM |
| **极简模式** | 跳过纠错 | Qwen2.5-1.5B (成句+纠错合并) | 低资源设备 | ≥4GB RAM (CPU) |

### 5.2 模型对比

| 模型 | 参数量 | 中文能力 | 推理速度 (本地) | API 价格 (云端) | 推荐用途 |
|------|--------|---------|----------------|----------------|---------|
| **Qwen2.5-1.5B** | 1.5B | ★★★☆ | ~20 tokens/s (CPU) | — | 快速纠错 |
| **Qwen2.5-7B** | 7B | ★★★★☆ | ~30 tokens/s (GPU) | — | 本地成句 |
| **DeepSeek-V3** | 671B(MoE) | ★★★★★ | — | ¥1/1M tokens | 云端成句 |
| **DeepSeek-R1** | 671B(MoE) | ★★★★★ | — | ¥4/1M tokens | 复杂纠错(按需) |
| **GPT-4o-mini** | — | ★★★★ | — | $0.15/1M tokens | 备选云端 |

---

## 六、Prompt 工程设计

### 6.1 Stage 1 — 词语纠错 Prompt

```
系统: 你是一个中文手语识别纠错助手。给定一个可能识别错误的词和它的上下文，请判断是否需要纠错并给出正确结果。

已知词典（这些是正确的目标词）：
{所有 100 个词条列表}

用户输入格式: 
上下文: [{前一个词}, {当前词(待纠错)}, {后一个词}]
置信度: {0-100}

请输出 JSON:
{"needs_correction": true/false, "corrected": "修正后的词", "reason": "简要理由"}

示例:
上下文: ["我", "尺", "饭"]  置信度: 45
→ {"needs_correction": true, "corrected": "吃", "reason": "结合上下文'我X饭'，应为'吃'"}
```

### 6.2 Stage 2 — 句子组装 Prompt

```
系统: 你是一个中文手语翻译后处理助手。手语识别系统输出了一个词语序列，请将其组装成自然流畅的中文句子。

规则:
1. 按照中文语法调整词序（如需要）
2. 补充必要的虚词（的、了、吗、吧 等）
3. 根据语义添加标点符号（。！？）
4. 不要添加原始序列中没有的实词
5. 如果序列明显不完整，设置 isComplete 为 false
6. 标注你做了哪些修正

输出 JSON:
{
  "sentence": "完整句子",
  "isComplete": true/false,
  "corrections": [{"position": 索引, "original": "原词", "corrected": "修正"}],
  "explanation": "简要说明做了哪些处理"
}

词典参考: {100 个词条}
```

### 6.3 Stage 3 — 词语补全 Prompt

```
系统: 你是一个中文词典助手。给定一个中文词，请提供拼音和释义。

输出 JSON:
{"text": "吃饭", "pinyin": "chī fàn", "meaning": "进食，用餐"}

如果词汇不在常见词典中，请根据你的知识合理推断。
```

---

## 七、文件与代码组织

### 7.1 新增文件

```
Project/Back/app/
├── llm_gateway.py          # LLM 抽象网关 + 三种实现
├── llm_pipeline.py         # 三阶段处理管道编排
├── llm_prompts.py          # Prompt 模板管理
├── sentence_accumulator.py # 句子累积器 + 边界检测
└── config_llm.py           # LLM 配置（模型选择、端点、超时等）

Project/Back/
└── config/
    └── llm_config.yaml     # 用户可编辑的 LLM 配置文件
```

### 7.2 修改文件

| 文件 | 改动内容 |
|------|---------|
| `main.py` | WebSocket handler 中集成 SentenceAccumulator + LLM Pipeline，新增 `sentence` / `llm_status` 消息类型 |
| `recognition.ts` (前端 store) | 新增 `currentSentence`、`sentences[]` 状态，`isLLMProcessing` 状态 |
| `useWebSocket.ts` | 处理 `type: "sentence"` 和 `type: "llm_status"` 消息 |
| `recognize.vue` | 新增句子级展示区域（可折叠卡片，显示完整句子 + 纠错高亮） |

---

## 八、配置与部署

### 8.1 llm_config.yaml

```yaml
llm:
  # 模式: "cloud" | "local" | "hybrid" | "none"
  mode: "hybrid"

  # 云端配置
  cloud:
    provider: "deepseek"        # deepseek | openai | qwen
    api_key: "${DEEPSEEK_API_KEY}"
    base_url: "https://api.deepseek.com/v1"
    model_sentence: "deepseek-chat"     # 句子组装用
    model_correction: "deepseek-chat"   # 纠错用

  # 本地配置 (Ollama)
  local:
    base_url: "http://localhost:11434/v1"
    model_sentence: "qwen2.5:7b"        # 句子组装用
    model_correction: "qwen2.5:1.5b"    # 纠错用

  # 混合模式分配
  hybrid:
    correction: "local"          # 纠错用本地
    sentence: "cloud"            # 成句用云端
    enrichment: "local"          # 拼音释义用本地

# 管道参数
pipeline:
  correction_threshold: 60       # 置信度低于此值的 token 才纠错
  sentence_timeout_ms: 1500      # 静默超时 (ms)
  sentence_max_tokens: 20        # 最大累积 token 数（强制 flush）
  context_window_size: 10        # 上下文窗口大小

# 降级策略
fallback:
  llm_timeout_ms: 3000           # LLM 调用超时
  max_retries: 1                 # 最大重试次数
  degraded_mode: "passthrough"   # 降级行为: "passthrough" | "simple_concat"
```

### 8.2 环境变量

```bash
# 云端 API
DEEPSEEK_API_KEY=sk-xxxx
OPENAI_API_KEY=sk-xxxx

# 本地 Ollama
OLLAMA_HOST=http://localhost:11434

# LLM 开关（紧急回退）
LLM_ENABLED=true
LLM_MODE=hybrid
```

---

## 九、性能与延迟预算

### 9.1 端到端延迟分析（混合模式，推荐配置）

```
摄像头采集          250ms (前端节流)
  → WebSocket 传输    ~15ms (局域网)
  → CTC 推理         ~150ms (ResNet18 + BiLSTM, GPU)
  → 快通道查表        ~2ms  → 词级结果发回 (总延迟 ~420ms)  ← 用户可感知的实时反馈

  → Accumulator 累积  ~1500ms (等待句子边界)
  → Stage 1 纠错     ~100ms (本地 Qwen2.5-1.5B, GPU)
  → Stage 2 成句     ~500ms (云端 DeepSeek API)
  → Stage 3 补全     ~50ms (本地)
  → 句级结果发回       (总延迟 ~2250ms)  ← 句子完成后一次性展示
```

### 9.2 优化策略

- **并行处理**：Stage 1 完成后立即更新 UI 中的词级纠错，不等 Stage 2 完成
- **流式输出**：云端 LLM 使用 Streaming API，逐 token 推送到前端（类似 ChatGPT 打字效果）
- **预加载上下文**：Accumulator 维护的上下文窗口随帧持续更新，LLM 调用时直接使用
- **缓存词典补全**：对同一词汇的拼音/释义补全结果缓存到本地 LRU

---

## 十、降级与容错

### 10.1 降级层级

```
Level 0 (正常):  快通道 + 慢通道 (LLM 全功能)
Level 1 (部分降级): LLM 超时 → 跳过纠错，仅做简单拼接 ("".join(tokens))
Level 2 (云端降级): 云端不可用 → 全部使用本地模型 (质量略降)
Level 3 (完全降级): LLM 完全不可用 → 回退到现有管道 (v2.0 行为)
```

### 10.2 健康检查

```python
class LLMHealthChecker:
    """定期 ping 各 LLM 端点，自动切换模式"""
    # - 每 30s 检查一次
    # - 连续失败 3 次触发降级
    # - 恢复后自动升回正常模式
```

---

## 十一、前端适配要点

### 11.1 状态管理扩展 (recognition.ts)

```typescript
// 新增状态
const currentSentence = ref<SentenceResult | null>(null)
const sentences = ref<SentenceResult[]>([])
const isLLMProcessing = ref(false)
const llmStage = ref<string>('idle')  // 'idle' | 'correcting' | 'forming' | 'enriching'
```

### 11.2 识别页 UI 改动 (recognize.vue)

```
现有结果面板 (词级，顶部浮动):
┌─────────────────────────────────┐
│ 🔴 录制中  01:23  置信度 92%    │
│                                 │
│ 你好  nǐ hǎo  [播放] [复制]     │  ← 快通道，实时刷新
└─────────────────────────────────┘

新增句子面板 (句级，底部展开):
┌─────────────────────────────────┐
│ 📝 实时句子                      │
│ ┌─────────────────────────────┐ │
│ │ 你好，我想吃饭。              │ │  ← 慢通道，句子完成时刷新
│ │                             │ │
│ │ 词: 你好 / 我 / 想 / 吃 / 饭 │ │
│ │      ✏️ 尺→吃 (自动修正)      │ │  ← 纠错高亮
│ │                             │ │
│ │ [📢 朗读] [📋 复制] [⭐ 收藏] │ │
│ └─────────────────────────────┘ │
│                                 │
│ ◀ 上句  下句 ▶                  │  ← 历史句子导航
└─────────────────────────────────┘
```

### 11.3 LLM 状态指示器

在录制按钮旁增加一个小的 AI 图标指示器：
- 灰色 (idle)：LLM 空闲
- 蓝色旋转 (processing)：LLM 处理中
- 绿色 (done)：处理完成
- 红色 (error/degraded)：LLM 不可用

---

## 十二、实施路线图

| 阶段 | 内容 | 预计人天 | 交付物 |
|------|------|---------|--------|
| **Phase 1** | LLM Gateway 抽象层 + 云端 DeepSeek 实现 | 1-2 天 | `llm_gateway.py` + `config_llm.py` |
| **Phase 2** | SentenceAccumulator + 边界检测 | 1 天 | `sentence_accumulator.py` |
| **Phase 3** | LLM Pipeline 三阶段编排 + Prompt 模板 | 1-2 天 | `llm_pipeline.py` + `llm_prompts.py` |
| **Phase 4** | main.py WebSocket 集成 + 协议扩展 | 1 天 | 修改后的 `main.py` |
| **Phase 5** | 前端适配 (store + socket + UI) | 1-2 天 | 修改后的前端文件 |
| **Phase 6** | 本地 Ollama 实现 + 混合模式 | 1 天 | LocalLLMGateway + HybridLLMGateway |
| **Phase 7** | 降级/容错/健康检查 + 联调测试 | 1-2 天 | 完整可部署系统 |

总计：约 **7-11 人天**，可分阶段交付，Phase 1-4 即可上线云端版本。

---

## 十三、总结

这套方案的核心设计哲学：

1. **双通道异步** — 词级快反馈 + 句级慢润色，不阻塞实时体验
2. **可拔插 LLM** — Gateway 抽象层使本地/云端/混合模式可热切换
3. **优雅降级** — LLM 不可用时自动回退现有管道，不影响基本功能
4. **渐进增强** — 每个阶段独立可工作，可按需逐步实现
