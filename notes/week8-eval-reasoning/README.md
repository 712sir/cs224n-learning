# Week 8: RAG + Agent + LLM 评估

> 时间：第 8 周 | 阶段：四（LLM 全栈） | 状态：🔴 未开始

---

## 本周目标

- [ ] 理解 RAG 的检索-生成流程及其设计空间
- [ ] 掌握 Agent 的 ReAct / Tool Use 模式
- [ ] 理解 LLM 评估的核心指标与陷阱
- [ ] 了解 LLM-as-a-Judge 范式
- [ ] 完成 Assignment 4：LLM 基准测试与评估

---

## 1. RAG (Retrieval-Augmented Generation)

### 架构

```
Query → Retriever → Top-K Documents → Generator → Answer
```

### 设计空间

| 维度 | 选项 | 权衡 |
|------|------|------|
| 检索粒度 | Document / Passage / Sentence | 上下文窗口 vs 相关性 |
| 检索模型 | Sparse (BM25) / Dense (DPR) / Hybrid | 精确匹配 vs 语义匹配 |
| 融合方式 | 拼接 / 交叉注意力 / 摘要 | 信息密度 vs 计算成本 |
| 索引更新 | 静态 / 增量 / 实时 | 新鲜度 vs 复杂度 |

### RAG vs 长上下文

> 待分析：10K context window 普及后，RAG 是否还有必要？

---

## 2. Agent

### ReAct 模式

```
Thought → Action → Observation → Thought → Action → ...
```

### 关键组件

```
Agent = LLM + Tool Use + Memory + Planning

  LLM: 推理引擎
  Tool Use: 搜索引擎/计算器/代码执行/API
  Memory: Short-term（上下文窗口）/ Long-term（向量数据库）
  Planning: 任务分解 / Self-reflection
```

### Tool Use 实现方案

| 方案 | 描述 |
|------|------|
| Function Calling | 原生 API，结构化输出 |
| JSON Mode | 强制输出 JSON → 解析调用 |
| Code Generation | 生成代码 → 执行 → 获取结果 |

---

## 3. LLM 评估

### 核心指标

| 指标 | 计算方式 | 适用场景 | 局限 |
|------|---------|---------|------|
| Perplexity | $\exp(-\frac{1}{n}\sum \log P(w_i\|w_{<i}))$ | 语言建模 | 不与下游任务相关 |
| BLEU | n-gram precision + brevity penalty | 翻译 | 不考虑语义，不敏感于重排序 |
| ROUGE | n-gram recall | 摘要 | 同 BLEU |
| BERTScore | BERT embedding 相似度 | 通用 | 计算量大 |

### LLM-as-a-Judge

```
Prompt: "请给以下回复的准确性打分（1-5）：
问题：{question}
参考答案：{reference}
模型回答：{candidate}"
```

### 评估陷阱

- [ ] Data contamination（训练集泄露）
- [ ] Prompt sensitivity（同一模型，不同 prompt 结果差异大）
- [ ] Metric gaming（为高分而优化，而非真正变好）

---

## 4. Assignment 4 记录

> 待补充

---

## 本周小结

> 待完成

---

## 参考资料

- [RAG Paper](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [ReAct Paper](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- [Challenges and Opportunities in NLP Benchmarking](https://www.ruder.io/nlp-benchmarking/) (Sebastian Ruder)
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023)
