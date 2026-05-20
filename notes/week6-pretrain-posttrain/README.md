# Week 6: 预训练架构 + 项目开题

> 时间：第 6 周 | 阶段：三（Transformer 核心） | 状态：🔴 未开始

---

## 本周目标

- [ ] 理解 BERT 的预训练目标（MLM + NSP）
- [ ] 理解 GPT 的自回归生成与 causal attention mask
- [ ] 对比 Encoder-only / Decoder-only / Encoder-Decoder 三类架构
- [ ] 完成 Assignment 3 并提交
- [ ] 确定 Final Project 方向（Default: Build GPT-2）

---

## 1. BERT (Encoder-only)

### 预训练目标

| 任务 | 描述 | 掩码策略 |
|------|------|---------|
| Masked Language Model (MLM) | 预测被 [MASK] 的词 | 15% 的词：80% [MASK], 10% random, 10% 不变 |
| Next Sentence Prediction (NSP) | 判断 B 是否是 A 的下一句 | 50% 正例, 50% 负例 |

### BERT 输入表示

```
[CLS] Token1 Token2 ... [SEP] TokenA TokenB ... [SEP]
  ↑                              ↑
 用于分类                    用于 NSP
```

### 关键设计选择

> 待分析：为什么 BERT 用 Layer Norm 在 Attention 之后，而 GPT-2 改成了 Attention 之前（Pre-LN）？

---

## 2. GPT (Decoder-only)

### 自回归语言模型

$$
P(x) = \prod_{i=1}^{n} P(x_i | x_1, ..., x_{i-1})
$$

### Causal Attention Mask

```
Attention Matrix (上三角掩码):
[ 1, 0, 0, 0 ]
[ 1, 1, 0, 0 ]
[ 1, 1, 1, 0 ]
[ 1, 1, 1, 1 ]
```

### GPT 系列演进

| 模型 | 年份 | 参数量 | 关键创新 |
|------|------|--------|---------|
| GPT-1 | 2018 | 117M | 无监督预训练 + 有监督微调 |
| GPT-2 | 2019 | 1.5B | Zero-shot Transfer |
| GPT-3 | 2020 | 175B | In-context Learning |
| GPT-4 | 2023 | ~1.8T? | Multimodal |

---

## 3. 三大架构对比

| 维度 | BERT (Encoder) | GPT (Decoder) | T5 (Encoder-Decoder) |
|------|---------------|---------------|---------------------|
| 注意力方向 | 双向 | 单向（causal） | 双向 Enc + 单向 Dec |
| 预训练目标 | MLM + NSP | LM (next token) | Span Corruption |
| 最擅长 | 理解任务（分类、NER） | 生成任务 | 翻译/摘要 |
| 典型模型 | BERT, RoBERTa | GPT-2/3/4, LLaMA | T5, BART |

---

## 4. Default Final Project: Build GPT-2

### 项目概述

> 来源：CS224N Winter 2026 Default Final Project

| 阶段 | 内容 | 产��� |
|------|------|------|
| Part 1 | 实现 GPT-2 核心架构 | 可运行的 GPT-2 模型 |
| Part 2 | 加载预训练权重 | HuggingFace → 自定义模型 |
| Part 3 | 文本生成实现 | Sampling / Beam Search |
| Part 4 | 下游任务微调 | 3 个任务的评估结果 |
| Part 5 | 分析与报告 | ACL 格式论文 |

### 三个下游任务（待确认）

1. 文本分类（如 SST-2 情感分析）
2. 摘要生成（如 CNN/DailyMail）
3. 阅读理解（如 SQuAD）

---

## 5. Assignment 3 完成记录

> 待补充

---

## 本周小结

> 待完成

---

## 参考资料

- [BERT Paper](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)
- [GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019)
- [Harnessing the Power of LLMs in Practice](https://arxiv.org/abs/2304.13712) — 架构选择指南
