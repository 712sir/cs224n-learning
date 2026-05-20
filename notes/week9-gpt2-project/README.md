# Week 9: GPT-2 Default Project（上）

> 时间：第 9 周 | 阶段：五（项目输出） | 状态：🔴 未开始

---

## 本周目标

- [ ] 从零实现 GPT-2 核心架构（Multi-Head Attention + FFN + LayerNorm）
- [ ] 加载 HuggingFace 预训练权重并验证正确性
- [ ] 实现文本生成（Temperature Sampling, Top-K, Top-P, Beam Search）
- [ ] 准备 3 个下游任务的数据集和评估管线

---

## 1. GPT-2 架构实现

### 与标准 Transformer 的关键差异

| 组件 | 标准 Transformer (Vaswani) | GPT-2 |
|------|---------------------------|-------|
| Layer Norm 位置 | Post-LN（Attention 后） | Pre-LN（Attention 前） |
| Activation | ReLU | GELU |
| Position Encoding | 固定 Sinusoidal | Learned |
| Weight Tying | 否 | 是（input embedding = output projection） |

### GPT-2 Small (124M) 参数表

| 层 | 维度 | 参数 |
|-----|------|------|
| Token Embedding | vocab_size × d_model | 50257 × 768 |
| Position Embedding | max_len × d_model | 1024 × 768 |
| 每层 Attention | — | 4 × 768 × 768 |
| 每层 FFN | — | 768 × 3072 × 2 |
| Layer Norm | — | 2 × 768 × 12 |
| 总计 | — | ~124M |

### Architecture Checklist

- [ ] Block (Pre-LN + Attention + Residual + Pre-LN + FFN + Residual)
- [ ] CausalSelfAttention (QKV projection → split heads → scored → mask → softmax → weighted sum → output projection)
- [ ] MLP (Linear → GELU → Linear)
- [ ] LayerNorm
- [ ] GPT2Model (Token Embed → Position Embed → Dropout → N × Block → Final LayerNorm)
- [ ] GPT2LMHeadModel (GPT2Model + LM Head with weight tying)

---

## 2. 加载预训练权重

```python
from transformers import GPT2LMHeadModel
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")

# 将 HuggingFace 权重映射到自己的模型
my_model.load_state_dict(convert_hf_weights(model_hf.state_dict()))
```

### 验证正确性

- [ ] 输入相同 prompt，对比 HF 和自己模型的 logits 差异
- [ ] 预期：差异 < 1e-5

---

## 3. 文本生成

### 解码策略

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| Greedy | 每步选最高概率 token | 确定性强 | 重复、单调 |
| Temperature | softmax 除以 T 再采样 | 控制多样性 | T 难调 |
| Top-K | 仅从 Top-K 选 | 避免低概率噪声 | 不同步 K 不合适 |
| Top-P (Nucleus) | 累积概率 ≥ P 的集合中选 | 动态调整候选集 | — |
| Beam Search | 保留 B 条最优路径 | 全局更优 | 多样性不足 |

---

## 4. 下游任务准备

> 待确认具体任务，暂定：

1. **SST-2**（情感分类）：二分类任务
2. **CNN/DailyMail**（摘要生成）：文本生成任务
3. **SQuAD**（阅读理解）：抽取式 QA

### 评估管线

```python
# 每项任务：
#   1. 数据加载 → 2. Prompt 构造 → 3. 模型推理 → 4. 指标计算 → 5. 结果记录
```

---

## 本周记录

> 待补充：实现细节、遇到的 bug 和解决方案

---

## 参考资料

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [minGPT (Karpathy)](https://github.com/karpathy/minGPT)
- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT) （已在 llm.c-learning 中学过）
- [CS224N DFP Handout](http://web.stanford.edu/class/cs224n/project/DFP_Instructions.pdf)
