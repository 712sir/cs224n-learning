# CS224N Lec1-2：NLP 导论 + Word2Vec — 面经级详解

> 课程：Stanford CS224N: NLP with Deep Learning (2024)
> 状态：🟡 进行中 | 关联：Transformer 的前身——理解 Word2Vec 才能理解 Embedding
> 面试权重：⭐⭐（NLP 基础，面试常问 "Word2Vec 原理" "负采样是什么"）

---

## Lec 1：NLP 导论 + 词表示

### 1.1 NLP 的三次范式转移

```
规则时代 (1950s-1980s)
  ├── 手写语法规则：if word ends with "ing" → verb
  └── 问题：语言无限复杂，规则永远写不完

统计时代 (1980s-2010s)
  ├── N-gram 语言模型：P(w_i | w_{i-2}, w_{i-1}) = count / count
  └── 问题：稀疏性——大部分 n-gram 从未出现过，概率为 0

深度学习时代 (2013-now)
  ├── 2013: Word2Vec — 用神经网络学稠密词向量
  ├── 2017: Transformer — "Attention is All You Need"
  ├── 2018: BERT/GPT — 预训练大模型
  └── 现在: GPT-4/Claude/Llama — 通用智能
```

### 1.2 为什么需要分布式表示？

**One-hot 编码的问题**：

```python
# One-hot: 每个词一个独热向量
vocab = {"cat": 0, "dog": 1, "run": 2, "walk": 3, ...}  # 50K 词
"cat"  = [1, 0, 0, 0, 0, ...]    # 50,000 维，只有一个 1
"dog"  = [0, 1, 0, 0, 0, ...]
# 问题 1: 维度灾难——50K 维，太稀疏
# 问题 2: "cat" 和 "dog" 的相似度 = 0（余弦相似度为 0），但语义上它们很相似！
# 问题 3: 无法泛化——没见过 "catdog" 就无法处理
```

**分布式表示 (Word Embedding)**：

```python
# 稠密向量: 每个词一个低维、稠密的浮点向量
"cat"  = [ 0.2, -0.5,  0.8,  0.3, ...]   # 通常 100-300 维
"dog"  = [ 0.3, -0.4,  0.7,  0.2, ...]   # "cat" 和 "dog" 很接近！
"run"  = [-0.1,  0.6, -0.2, -0.8, ...]   # 动词在另一个方向
```

**核心洞察**：*You shall know a word by the company it keeps* (J.R. Firth, 1957)
→ 出现在相似上下文中的词，语法和语义也相似。

---

## Lec 2：Word2Vec 详解

### 2.1 Skip-gram 模型

**目标**：给定中心词，预测周围的上下文词。

```
Window size = 2, 中心词 = "learning"

" I  enjoy  [learning]  about  NLP "
              ↑
           中心词
    ┌────────┼────────┐
    │        │        │
  enjoy    about     NLP      ← 上下文词（目标）
```

**Skip-gram 的数学**：

```
对于每个 (中心词, 上下文词) 对 (w, c)：

P(c | w) = exp(v_c · v_w) / Σ_{i ∈ V} exp(v_i · v_w)
           \________________/   \____________________/
              正样本得分             所有词的得分之和（归一化）

v_w = 中心词的向量（输入向量）
v_c = 上下文词的向量（输出向量）
两个是不同的矩阵！每个词有两套向量。
```

### 2.2 训练目标

```python
# 伪代码：Skip-gram 的损失函数
# 对每个窗口位置：
#   center = 中心词
#   contexts = 窗口内的其他词
#
# 损失 = -Σ_{上下文词} log P(上下文词 | 中心词)

# 例："I enjoy learning about NLP"
# 中心词 = "learning", 上下文 = ["enjoy", "about"]
#
# loss = -log P(enjoy | learning) - log P(about | learning)
#      = -log[exp(v_enjoy·v_learning) / Z] - log[exp(v_about·v_learning) / Z]
```

### 2.3 🔥 负采样 (Negative Sampling) — 面试必问

**问题**：Softmax 分母需要对 50K 词汇表求和 → 太慢！

```python
# 原始 softmax 计算量：
# P(c|w) = exp(v_c · v_w) / Σ 50000 个 exp(v_i · v_w)
#                           ↑ 每步训练都要算 5 万次内积
```

**负采样解决方案**：

```
原始：对 50K 个词做 softmax → 5 万次计算
负采样：随机抽 K 个"负样本"（噪声词），只对 (正样本 + K 个负样本) 计算

新的目标：对每个 (w, c)：
  正样本：最大化 P(是上下文 | w, c)
  负样本：最小化 P(是上下文 | w, c_neg)

损失函数（每个样本）：
  loss = -log σ(v_c · v_w) - Σ_{i=1}^{K} log σ(-v_{neg_i} · v_w)
          \_______________/   \______________________________/
             正样本（希望大）          K 个负样本（希望小）
```

| | 原始 Softmax | 负采样 |
|------|:--:|:--:|
| 计算量 | O(|V|) = 50K | O(K+1) = 6-21 |
| 推荐 K | — | 小语料 K=5-20，大语料 K=2-5 |
| 负样本分布 | — | 词频的 3/4 次方效果好 |
| 谁在用 | 早期 | word2vec, BERT (MLM用全部vocab但可以并行) |

**Q: 为什么负采样用 \`P(w)^(3/4)\` 而不是均匀采样？**

> 均匀采样：高频词和低频词被抽到的概率一样 → 高频词浪费太多负样本名额
> 频率采样 P(w)：the/a/an 等停用词几乎每次都被抽到 → 模型学不到有区分度的负样本
> P(w)^(3/4)：折中——提高低频词的采样概率，抑制超高频词
>
> 例：`is` vs `bombastic`
> - P(is) / P(bombastic) ≈ 10000
> - P(is)^0.75 / P(bombastic)^0.75 ≈ 1000
> → 高频词的相对优势被削弱了

### 2.4 CBOW (Continuous Bag of Words)

**与 Skip-gram 对称**：

```
Skip-gram: 中心词 → 预测上下文（一对多）
CBOW:      上下文 → 预测中心词（多对一）

" I  enjoy  [learning]  about  NLP "
     ↓        ↑         ↓
   上下文 —→ predict → 中心词
```

| | Skip-gram | CBOW |
|------|:--:|:--:|
| 方向 | 中心→上下文 | 上下文→中心 |
| 速度 | 慢（多目标） | 快（单一目标） |
| 低频词效果 | **好**（每个词有机会做中心词） | 差（低频词易被上下文平均掉） |
| 高频词效果 | 一般 | 好 |
| 实际用谁 | word2vec 默认 skip-gram | 小语料用 CBOW |

### 2.5 训练后的向量怎么用？

```python
# 每个词学了两套向量：
# v_w: 输入向量（做中心词时用）
# u_w: 输出向量（做上下文时用）

# 最终词向量通常用 v_w（输入向量），或 v_w + u_w

"king"   = [0.50, 0.32, -0.18, ...]
"man"    = [0.45, 0.28, -0.15, ...]
"woman"  = [0.40, 0.25, -0.12, ...]
"queen"  = [0.48, 0.30, -0.17, ...]

# 最有名的类比：
# king - man + woman ≈ queen
# → 词向量编码了语义关系！
```

### 2.6 🔥 评估方式

| 方式 | 任务 | 例子 |
|------|------|------|
| **Intrinsic** | 词类比 | `king - man + woman = ?` (期望 queen) |
| | 词相似度 | `cat` 和 `dog` 的余弦相似度 |
| **Extrinsic** | 下游任务 | 用学到的 embedding 做情感分类/NER/翻译 |

---

## 与 AI Infra 的连接

```
Word2Vec (2013)
  ↓ 核心思想：用向量表示离散符号
GloVe (2014) — 全局共现统计
  ↓
Transformer (2017) — Self-Attention 替代 RNN
  ↓
BERT/GPT (2018) — 预训练 + 微调
  ↓
现在 — vLLM 推理引擎、FlashAttention

你要做的推理引擎，输入 token ID 的第一步就是查 embedding 表！
→ wte: token_id → embedding vector (vocab_size=50257, n_embd=768)
→ 这就是 Word2Vec 的后代——可学习的 token embedding
```

---

## 面试拷问

**Q1: Word2Vec 的两个模型是什么？区别？**

> Skip-gram：中心词预测上下文，低频词效果好，慢
> CBOW：上下文预测中心词，快，低频词效果差

**Q2: 什么是负采样？为什么需要？**

> 不需要对全部 |V| 个词算 softmax，只算 K 个负样本（K=5-20）。训练速度从 O(|V|) 降到 O(K)。

**Q3: 为什么负样本按 P(w)^(3/4) 采样？**

> 纯频率采样：高频 stop words 占了太多负样本，浪费。3/4 次方削弱高频词优势。

**Q4: 每个词有几个向量？为什么？**

> 两个：输入向量 v_w（做中心词）和输出向量 u_w（做上下文）。最终通常用 v_w 或 v_w + u_w。

**Q5: Word2Vec 的局限性？**

> 1. 静态——一个词只有一种向量，"bank"（银行/河岸）无法区分
> 2. 上下文窗口有限——不能捕捉长距离依赖
> 3. 后来被 BERT（上下文相关 embedding）和 GPT 取代
> 4. 不能处理 OOV（未登录词）

**Q6: "king - man + woman ≈ queen" 为什么能 work？**

> 词向量中编码了语义关系。"king" 和 "queen" 的差异 ≈ "man" 和 "woman" 的差异（性别维度）。向量加减 ≈ 语义运算。

---

## 自测题

1. [ ] One-hot 的两个致命缺点是什么？
2. [ ] Skip-gram 的目标函数写出来（不用背公式，能描述就行）
3. [ ] 负采样从 50000 个词里抽几个？为什么这个数？
4. [ ] P(w)^(3/4) 和均匀采样有什么区别？
5. [ ] word2vec 和 GPT 的 embedding 层有什么联系？
