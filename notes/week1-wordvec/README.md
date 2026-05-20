# Week 1: 词向量与 Word2Vec

> 时间：第 1 周 | 阶段：一（词向量与基础） | 状态：🔴 未开始

---

## 本周目标

- [ ] 搭建 PyTorch 开发环境（conda + PyTorch + Jupyter）
- [ ] 理解 NLP 的四个历史阶段及关键突破
- [ ] 理解分布语义学直觉："You shall know a word by the company it keeps" (J.R. Firth, 1957)
- [ ] 推导 Word2Vec Skip-gram 的目标函数
- [ ] 手推 Negative Sampling 的梯度更新公式
- [ ] 理解 GloVe 的共现矩阵分解思想
- [ ] 完成 Assignment 1：Exploring Word Vectors

---

## 1. NLP 历史与关键突破

### 四个阶段

| 阶段 | 时间 | 代表方法 | 核心思想 |
|------|------|---------|---------|
| 规则时代 | 1950s-1980s | 手写规则、正则 | 人工编码语言规则 |
| 统计时代 | 1980s-2010 | N-gram、HMM、CRF | 概率模型 + 特征工程 |
| 深度学习时代 | 2010-2020 | RNN/LSTM、Seq2Seq | 端到端神经网络 |
| LLM 时代 | 2020-至今 | GPT、BERT、LLaMA | 大规模预训练 + 涌现 |

### 关键思考题

1. 为什么统计方法在 NLP 中停留了这么久？（答：……）
2. 深度学习给 NLP 带来的最大变化是什么？（答：……）
3. 现在为什么是 LLM 的时代？（答：……）

---

## 2. 词向量基础

### 核心直觉

```
"The meaning of a word is its use in the language." — Wittgenstein
```

### Word2Vec: Skip-gram 模型

#### 目标函数推导

$$
J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} | w_t)
$$

> 待补充：完整推导过程

#### Softmax 的计算瓶颈

- 词汇表大小 $|V|$ 通常 $\geq 10^5$
- 每次计算 softmax 需要 $O(|V|)$ 时间
- 解决方案：Negative Sampling / Hierarchical Softmax

#### Negative Sampling

> 待手推梯度

### 思考题

1. Skip-gram 和 CBOW 各有什么优劣？什么时候用哪个？
2. 为什么 Negative Sampling 采样概率取 $P(w)^{3/4}$？
3. 两个词向量矩阵 $U$（input）和 $V$（output）分别起什么作用？为什么最后可以只用 $U$？

---

## 3. GloVe

### 核心思想

> 待补充：共现矩阵 → 加权最小二乘 的推导

### 与 Word2Vec 的对比

| 维度 | Word2Vec | GloVe |
|------|---------|-------|
| 信息来源 | 局部窗口 | 全局共现 |
| 训练方式 | 随机梯度下降 | 矩阵分解 |
| 统计利用 | 隐式 | 显式 |

---

## 4. Assignment 1 记录

### 环境搭建

```bash
conda create -n cs224n python=3.9
conda activate cs224n
pip install torch numpy matplotlib jupyter
```

> 记录踩坑过程：

### 作业关键点

- [ ] Part 1: Count-Based Word Vectors（SVD 分解共现矩阵）
- [ ] Part 2: Word2Vec in NumPy（朴素 softmax）
- [ ] Part 3: Word2Vec with Negative Sampling
- [ ] Part 4: 语义类比与可视化

---

## 5. 核心论文精读

### Mikolov et al., 2013 — Efficient Estimation of Word Representations in Vector Space

| 项目 | 内容 |
|------|------|
| **核心贡献** | 提出 CBOW 和 Skip-gram，大幅降低训练复杂度 |
| **关键技巧** | Hierarchical Softmax |
| **遗留问题** | HS 仍然较慢，负采样更优 |

> 待补充完整 paper notes

### Mikolov et al., 2013 — Distributed Representations of Words and Phrases

| 项目 | 内容 |
|------|------|
| **核心贡献** | 提出 Negative Sampling + Subsampling |
| **关键公式** | $P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$ (subsampling) |
| **惊艳结果** | King - Man + Woman = Queen |

> 待补充完整 paper notes

---

## 本周小结

> 待完成

---

## 参考资料

- [CS224N 2024 Lecture 1 (YouTube)](https://www.youtube.com/watch?v=HlK2z1-eL6c)
- [CS224N 2024 Lecture 2 (YouTube)](https://www.youtube.com/watch?v=rmVRLeJRkl4)
- [CS224N Notes: Word Vectors I](http://web.stanford.edu/class/cs224n/readings/cs224n_winter2023_lecture1_notes_draft.pdf)
- [CS224N Notes: Word Vectors II](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)
- [Speech and Language Processing — Chapter 6: Vector Semantics](https://web.stanford.edu/~jurafsky/slp3/6.pdf)
