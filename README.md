# CS224N 学习笔记 —— 从词向量到 GPT-2

> 核心原则：以"同一概念，两层对照"的方式交替进行。学了 Transformer 的理论推导，立刻去看 PyTorch 实现和论文源码，建立"数学公式→代码实现→论文原意"的三层映射。

---

## 我做了什么

在 10 周内，跟随 [Stanford CS224N (Winter 2026)](http://web.stanford.edu/class/cs224n/index.html)，从词向量基础出发，逐层深入到 Transformer 架构和 LLM 全栈：

- 完成全部 4 次官方作业（词向量 → 神经网络 → Transformer → LLM 评估）
- 逐行推导反向传播、Self-Attention、Transformer 的数学公式
- 从零复现 word2vec、Transformer、GPT-2 三个核心模型
- 完成 Default Final Project：实现 GPT-2 + 3 个下游任务评估

---

## 学习路线图

| 阶段 | 周次 | 核心目标 | 笔记 |
|------|------|---------|------|
| 一：词向量与基础 | 第 1-2 周 | 环境搭建、词向量原理、完成 A1 | [笔记 →](notes/week1-wordvec/) |
| 二：神经网络与 RNN | 第 3-4 周 | 反向传播推导、RNN/LSTM、依存解析、完成 A2 | [笔记 →](notes/week3-rnn-lm/) |
| 三：Transformer 核心 | 第 5-6 周 | Self-Attention 推导、Transformer 复现、完成 A3 | [笔记 →](notes/week5-transformer/) |
| 四：LLM 全栈 | 第 7-8 周 | 预训练/PEFT/RLHF/RAG/Agents、LLM 评估、完成 A4 | [笔记 →](notes/week7-peft-rag-agents/) |
| 五：项目输出 | 第 9-10 周 | 完成 GPT-2 Default Project、最终报告 | [笔记 →](notes/week9-gpt2-project/) |

---

## 详细 10 周计划

### 阶段一：词向量与基础

#### 第 1 周：NLP 历史 + 词向量（Word2Vec）

**视频**：[CS224N 2024 Lecture 1](https://www.youtube.com/watch?v=HlK2z1-eL6c&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) + [Lecture 2](https://www.youtube.com/watch?v=rmVRLeJRkl4)

**目标**：
- [ ] 搭建 PyTorch 开发环境
- [ ] 理解词向量的直觉：分布语义学（Distributional Semantics）
- [ ] 推导 Word2Vec 的 Skip-gram 和 CBOW 目标函数
- [ ] 手推 Negative Sampling 的梯度
- [ ] 理解 GloVe 的共现矩阵分解思路
- [ ] 完成 Assignment 1：实现 word2vec（Python + NumPy）

**核心论文**：
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (Mikolov et al., 2013)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) (Pennington et al., 2014)

**实验**：用 NumPy 从头实现 Skip-gram + Negative Sampling

---

#### 第 2 周：反向传播 + 神经网络基础

**视频**：[CS224N 2024 Lecture 3](https://www.youtube.com/watch?v=8CWyBNf1k2s&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D)

**目标**：
- [ ] 手推 softmax + cross-entropy 的梯度
- [ ] 掌握矩阵微积分（Matrix Calculus）技巧
- [ ] 理解反向传播的计算图视角
- [ ] 掌握 PyTorch 的 autograd 机制
- [ ] 完成 Assignment 2（第一部分）：手算 tensor 导数

**核心资料**：
- [CS224N 2019 Notes: Neural Networks](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes03-neuralnets.pdf)
- [Matrix Calculus Notes](http://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf)
- [CS231N: Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/handouts/derivatives.pdf)

**实验**：纯 NumPy 实现两层 MLP + 手动反向传播

---

### 阶段二：神经网络与 RNN

#### 第 3 周：RNN + 语言模型

**视频**：[CS224N 2024 Lecture 4](https://www.youtube.com/watch?v=QEwPSI4FLTQ&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) + [Lecture 5](https://www.youtube.com/watch?v=PLryWeHPcBs)

**目标**：
- [ ] 理解 N-gram 语言模型及其平滑方法
- [ ] 推导 RNN 的 BPTT（Backpropagation Through Time）
- [ ] 理解梯度消失/爆炸问题的数学本质
- [ ] 掌握 LSTM 和 GRU 的门控机制
- [ ] 完成 Assignment 2（第二部分）：实现依存解析器

**核心论文**：
- [Learning Representations by Backpropagating Errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) (Rumelhart et al., 1986)

**对照阅读**：
- CS224N 2019 Notes: [Language Models and RNNs](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)

**实验**：用 PyTorch 实现字符级 LSTM 语言模型

---

#### 第 4 周：依存解析 + 回顾巩固

**视频**：[CS224N 2024 Lecture 5（dependency parsing 部分）](https://www.youtube.com/watch?v=PLryWeHPcBs)

**目标**：
- [ ] 理解依存语法的基本概念
- [ ] 掌握 Transition-based Dependency Parsing
- [ ] 理解 Arc-Standard 和 Arc-Eager 转换系统
- [ ] 完成 Assignment 2 全部内容并提交
- [ ] 回顾前 3 周内容，整理知识图谱

**实验**：完成 Assignment 2 的依存解析器部分

---

### 阶段三：Transformer 核心

#### 第 5 周：Self-Attention + Transformer 架构

**视频**：[CS224N 2024 Lecture 6](https://www.youtube.com/watch?v=ptgX3-6zLpA&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D)

**目标**：
- [ ] 从 Attention 的直觉出发，推导 Scaled Dot-Product Attention
- [ ] 逐行推导 Multi-Head Attention 的完整计算过程
- [ ] 理解 Positional Encoding 的数学动机（正弦编码 vs 可学习编码）
- [ ] 掌握 Layer Normalization 和 Residual Connection
- [ ] 完成 Assignment 3：实现 Transformer 组件

**核心论文**：
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) —— **必读，逐节精读**

**对照阅读**：
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) (Harvard NLP)

**实验**：从零用 PyTorch 复现 Transformer Encoder + Decoder

---

#### 第 6 周：Transformer 深入 + 项目开题

**视频**：[CS224N 2024 Lecture 7](https://www.youtube.com/watch?v=KZX4vX0SqIc)（Final Project 指导）

**目标**：
- [ ] 深入理解 Transformer 的变体（Transformer-XL、Relative Position）
- [ ] 理解 BERT 的预训练目标（MLM + NSP）
- [ ] 理解 GPT 的自回归生成方式
- [ ] 完成 Assignment 3 全部内容并提交
- [ ] 确定 Final Project 方向（推荐 Default Project: Build GPT-2）

**核心论文**：
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018) —— GPT-1

**实验**：对比 BERT 和 GPT 的 Attention Mask 差异

---

### 阶段四：LLM 全栈

#### 第 7 周：预训练 + 后训练（SFT/RLHF）

**视频**：[CS224N 2024 Lecture 8](https://www.youtube.com/watch?v=KZX4vX0SqIc) + [Lecture 9](https://www.youtube.com/watch?v=z8sQeqoKYIA)

**目标**：
- [ ] 理解 GPT 预训练的数据准备、Scaling Law
- [ ] 掌握 SFT（Supervised Fine-Tuning）的流程
- [ ] 理解 RLHF 的三阶段：SFT → Reward Model → PPO
- [ ] 了解 DPO 作为 RLHF 的替代方案
- [ ] PEFT（LoRA/Adapter）的原理与实践

**核心论文**：
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
- [Training language models to follow instructions](https://arxiv.org/abs/2203.02155) (InstructGPT)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)

**实验**：用 HuggingFace PEFT 对 GPT-2 做 LoRA 微调

---

#### 第 8 周：RAG + Agent + 评估

**视频**：[CS224N 2024 Lecture 10](https://www.youtube.com/watch?v=5vh0mDW2FUo) + [Lecture 11](https://www.youtube.com/watch?v=vn6cDHDpkSQ)

**目标**：
- [ ] 理解 RAG 的检索-生成流程
- [ ] 掌握 Agent 的 ReAct / Tool Use 模式
- [ ] 理解 LLM 评估的核心指标（Perplexity, BLEU, ROUGE, BERTScore）
- [ ] 了解 LLM-as-a-Judge 范式
- [ ] 完成 Assignment 4：LLM 基准测试与评估

**核心论文**：
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)

**实验**：搭建一个简单的 RAG 问答系统（LangChain + ChromaDB）

---

### 阶段五：项目输出

#### 第 9 周：GPT-2 Default Project（上）

**目标**：
- [ ] 实现 GPT-2 的核心架构（Multi-Head Attention + FFN + LayerNorm）
- [ ] 加载预训练权重（HuggingFace → 自定义模型）
- [ ] 实现文本生成（Temperature Sampling, Top-K, Top-P）
- [ ] 准备 3 个下游任务的数据集

**参考资料**：
- [minGPT](https://github.com/karpathy/minGPT) (Karpathy)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2 paper)
- [Default Final Project Handout](http://web.stanford.edu/class/cs224n/project/DFP_Instructions.pdf)

---

#### 第 10 周：GPT-2 Default Project（下）+ 最终报告

**目标**：
- [ ] 完成 3 个下游任务的微调与评估
- [ ] 分析不同参数规模（GPT-2 Small/Medium）的性能差异
- [ ] 撰写最终项目报告（类似 ACL 论文格式）
- [ ] 整理全部笔记、代码、实验结果
- [ ] 发布 GitHub 仓库完整版

---

## 仓库导航

```
cs224n-learning/
├── notes/              # 学习笔记（每周一个目录）
│   ├── week1-wordvec/       # 词向量与 Word2Vec
│   ├── week2-backprop/      # 反向传播与神经网络
│   ├── week3-rnn-lm/        # RNN 与语言模型
│   ├── week4-dependency/    # 依存解析
│   ├── week5-transformer/   # Self-Attention 与 Transformer
│   ├── week6-pretrain-posttrain/  # BERT/GPT + 预训练-后训练
│   ├── week7-peft-rag-agents/     # LoRA + RAG + Agent
│   ├── week8-eval-reasoning/      # LLM 评估 + 推理
│   ├── week9-gpt2-project/  # GPT-2 Default Project
│   └── week10-final-report/ # 最终报告
├── assignments/        # 4 次官方作业完整解答
│   ├── a1/  # 词向量
│   ├── a2/  # 神经网络 + 依存解析
│   ├── a3/  # Transformer
│   └── a4/  # LLM 评估
├── experiments/        # 关键模型复现
│   ├── word2vec/              # Word2Vec 从零实现
│   ├── transformer-from-scratch/  # Transformer 完整复现
│   └── gpt2/                  # GPT-2 复现（Default Project）
├── papers/             # 论文精读笔记
├── diagrams/           # 手绘图（架构图、注意力可视化）
├── reports/            # 阶段总结 + 最终报告
└── .github/            # CI workflows
```

---

## 资源索引

### 视频

| 来源 | 链接 | 说明 |
|------|------|------|
| B 站 2025 最新 | [BV1vQMBz6EvP](https://www.bilibili.com/video/BV1vQMBz6EvP) | 中英字幕，基于 2024 版 |
| YouTube 2024 官方 | [播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) | 官方公开课 |
| YouTube 2023 官方 | [播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4) | 早期版本 |

### 课件与作业

| 资源 | 链接 |
|------|------|
| 课程官网 (Winter 2026) | [CS224N Home](http://web.stanford.edu/class/cs224n/index.html) |
| 官方作业 | 见官网 Assignments 区（每年更新，勿用旧版） |
| PKUFlyingPig 参考仓库 | [GitHub](https://github.com/PKUFlyingPig/CS224n) |

### 参考教材

| 教材 | 作者 | 说明 |
|------|------|------|
| [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) | Jurafsky & Martin | NLP 圣经，2024 新版 |
| [NLP with Transformers](https://transformersbook.com/) | Tunstall et al. | HuggingFace 实战 |
| [Deep Learning](http://www.deeplearningbook.org/) | Goodfellow et al. | 深度学习理论 |

---

## 关键技术栈

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/712sir/cs224n-learning.git

# 安装环境
conda create -n cs224n python=3.9
conda activate cs224n
pip install torch numpy matplotlib jupyter transformers datasets

# 浏览笔记
cd cs224n-learning/notes/
ls

# 运行实验
cd experiments/word2vec/
python skipgram.py
```

---

## 学习日志

| 时间 | 进度 | 备注 |
|------|------|------|
| Week 1 | 仓库初始化 | 建立学习计划 |
| ... | ... | ... |
