# Week 10: GPT-2 Default Project（下）+ 最终报告

> 时间：第 10 周 | 阶段：五（项目输出） | 状态：🔴 未开始

---

## 本周目标

- [ ] 完成 3 个下游任务的微调与评估
- [ ] 分析不同参数规模的性能差异
- [ ] 撰写最终项目报告（ACL 格式）
- [ ] 整理全部笔记、代码、实验结果
- [ ] 发布 GitHub 仓库完整版

---

## 1. 下游任务评估

### 任务 1: 文本分类（SST-2）

| 项目 | 内容 |
|------|------|
| 评估指标 | Accuracy |
| Few-shot 策略 | 5-shot / 10-shot |
| 结果 | 待填入 |

### 任务 2: 摘要生成（CNN/DailyMail）

| 项目 | 内容 |
|------|------|
| 评估指标 | ROUGE-1/2/L |
| 策略 | Zero-shot Prompt |
| 结果 | 待填入 |

### 任务 3: 阅读理解（SQuAD）

| 项目 | 内容 |
|------|------|
| 评估指标 | Exact Match / F1 |
| 策略 | Zero-shot Prompt |
| 结果 | 待填入 |

---

## 2. 规模对比分析

| 模型 | 参数量 | SST-2 Acc | ROUGE-L | SQuAD F1 |
|------|--------|-----------|---------|----------|
| GPT-2 Small | 124M | 待测 | 待测 | 待测 |
| GPT-2 Medium | 355M | 待测 | 待测 | 待测 |

### 分析框架

1. **性能随规模递增吗？** → 是/否，分析原因
2. **什么样的任务收益最大？** → 分类 vs 生成 vs 理解
3. **瓶颈在哪里？** → 上下文窗口？知识密度？

---

## 3. 最终报告模板

### 报告结构（ACL 风格）

```
1.  Abstract（200 词总结）
2.  Introduction（为什么复现 GPT-2 有意义）
3.  Model Architecture（架构实现 + 加载预训练权重）
4.  Generation Strategy（解码策略选择与对比）
5.  Downstream Task Evaluation（3 个任务的结果与分析）
6.  Ablation Studies（如有）
7.  Related Work
8.  Conclusion
9.  References
```

### 图表清单

- [ ] 模型架构图（手工或 draw.io）
- [ ] 三个任务的评估结果对比柱状图
- [ ] 不同解码策略的生成样例对比
- [ ] 不同模型规模的性能对比

---

## 4. 全局回顾

### 10 周知识体系自查

| 知识域 | 关键概念 | 掌握程度 |
|--------|---------|---------|
| 词向量 | Skip-gram / Negative Sampling / GloVe | ⬜ |
| 反向传播 | Matrix Calculus / BPTT | ⬜ |
| RNN/LSTM | 门控机制 / 梯度消失 | ⬜ |
| Attention | QKV / Multi-Head / Scaled Dot-Product | ⬜ |
| Transformer | Encoder-Decoder / Pre-LN vs Post-LN | ⬜ |
| 预训练 | BERT MLM / GPT LM / Scaling Law | ⬜ |
| 对齐 | SFT / RLHF / DPO | ⬜ |
| PEFT | LoRA / Adapter / QLoRA | ⬜ |
| RAG & Agent | Retriever / Tool Use / ReAct | ⬜ |
| 评估 | PPL / BLEU / BERTScore / LLM-as-Judge | ⬜ |

---

## 5. GitHub 发布清单

- [ ] README.md — 完整的路线图
- [ ] notes/ — 10 周笔记，每周内容充实
- [ ] assignments/ — 4 次作业完整解答
- [ ] experiments/ — 3 个实验（word2vec / transformer / gpt2）
- [ ] papers/ — 论文精读笔记 ≥15 篇
- [ ] diagrams/ — 架构图 ≥5 张
- [ ] reports/ — 5 个阶段总结 + 最终项目报告
- [ ] .github/ — 可选的 CI 配置

---

## 本周记录

> 待补充

---

## 参考

- [ACL Paper Formatting Guidelines](https://acl-org.github.io/ACLPUB/formatting.html)
- [CS224N Project Report Instructions](http://web.stanford.edu/class/cs224n/project/Project_Report_Instructions.pdf)
