# Week 5: Self-Attention 与 Transformer

> 时间：第 5 周 | 阶段：三（Transformer 核心） | 状态：🔴 未开始

---

## 本周目标

- [ ] 从 Attention 直觉出发，完整推导 Scaled Dot-Product Attention
- [ ] 逐行推导 Multi-Head Attention 的每个矩阵操作
- [ ] 理解 Positional Encoding 的数学动机
- [ ] 掌握 Layer Normalization vs Batch Normalization
- [ ] 完成 Assignment 3：Self-attention and Transformers

---

## 1. Attention 的直觉与数学

### 从 Seq2Seq 瓶颈到 Attention

```
传统 Seq2Seq: Encoder 压缩成单个向量 → 信息瓶颈
Attention: Decoder 每一步"回顾" Encoder 所有隐藏状态 → 动态加权
```

### Scaled Dot-Product Attention 推导

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

#### 为什么是 scaled？

> 待推导：当 $d_k$ 很大时，点积的方差增长 → softmax 趋于 one-hot → 梯度消失

#### 逐元素计算展开

> 待补充：将 Q、K、V 显式写出矩阵维度，逐步推导

---

## 2. Multi-Head Attention

### 完整计算流程

```
输入: X ∈ R^{seq_len × d_model}

对每个 head i (共 h 个):
  Q_i = X · W^Q_i    (d_model → d_k)
  K_i = X · W^K_i    (d_model → d_k)
  V_i = X · W^V_i    (d_model → d_v)

  head_i = Attention(Q_i, K_i, V_i)   # [seq_len, d_v]

MultiHead = Concat(head_1, ..., head_h) · W^O   # [seq_len, d_model]
```

### 为什么 Multi-Head？

- 不同 head 关注不同位置（语法 / 语义 / 位置关系）
- 类似 CNN 的多通道，增强表达能力

---

## 3. Transformer 完整架构

### 架构图

```
                    Output Probabilities
                           ↑
                      Softmax
                           ↑
                       Linear
                           ↑
                    Add & Norm ←──┐
                    ↑             │
                 Feed Forward     │
                    ↑             │
                 Add & Norm       │
                    ↑             │
              Multi-Head Attn ────┘
                    ↑
              Add & Norm ←── Positional Encoding
                    ↑             │
           Multi-Head Attn ←──────┘
                    ↑
               Input Embedding
                    ↑
               Input Tokens
```

### 三大支柱

| 组件 | 作用 | 公式 |
|------|------|------|
| Self-Attention | 序列内全局交互 | $softmax(QK^T / \sqrt{d_k})V$ |
| Layer Norm | 稳定训练 | $y = \gamma \cdot \frac{x-\mu}{\sigma} + \beta$ |
| Residual Connection | 梯度高速公路 | $y = F(x) + x$ |

### Batch Norm vs Layer Norm

| 维度 | Batch Norm | Layer Norm |
|------|-----------|------------|
| 归一化轴 | Batch 维度 | Feature 维度 |
| batch_size 敏感 | 是（小 batch 不稳定） | 否 |
| 序列模型适配 | 差（变长序列不好处理） | 好 |
| NLP 首选 | 否 | 是 |

---

## 4. Positional Encoding

### Sinusoidal Encoding

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

### 为什么用正弦/余弦？

> 待推导：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数 → 模型可学习相对位置

### 其他位置编码方案

| 方案 | 核心思想 | 代表模型 |
|------|---------|---------|
| Sinusoidal | 固定正弦函数 | Transformer, BERT |
| Learned | 可学习参数 | GPT, ViT |
| Relative | 相对距离偏置 | Transformer-XL |
| RoPE | 旋转位置编码 | LLaMA, Qwen |

---

## 5. Assignment 3 记录

### 关键实现

- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention
- [ ] Transformer Encoder Layer
- [ ] Positional Encoding
- [ ] 完整 Transformer 模型

> 待记录实现细节、训练日志、实验结果

---

## 6. "Attention Is All You Need" 论文精读

| 维度 | 内容 |
|------|------|
| **核心贡献** | 纯 Attention 架构，无循环无卷积 |
| **关键创新** | Multi-Head Attention + Positional Encoding + Layer Norm |
| **实验结果** | BLEU 28.4 (WMT 2014 EN→DE)，训练时间比 RNN 少很多 |
| **遗留问题** | 位置编码不完美，$O(n^2)$ 复杂度 |

> 待补充逐节 paper notes

---

## 本周小结

> 待完成

---

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [CS224N Notes: Self-Attention and Transformers](http://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)
