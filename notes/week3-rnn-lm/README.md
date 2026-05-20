# Week 3: RNN 与语言模型

> 时间：第 3 周 | 阶段：二（神经网络与 RNN） | 状态：🔴 未开始

---

## 本周目标

- [ ] 理解 N-gram 语言模型及 smoothing 技术
- [ ] 推导 RNN 的 BPTT（Backpropagation Through Time）
- [ ] 理解梯度消失/爆炸的数学本质
- [ ] 掌握 LSTM 和 GRU 的门控机制
- [ ] 完成 Assignment 2（第二部分）：依存解析器

---

## 1. 语言模型基础

### N-gram 模型

$$
P(w_1, w_2, ..., w_n) \approx \prod_{i=1}^{n} P(w_i | w_{i-k+1}, ..., w_{i-1})
$$

### Smoothing 技术

| 方法 | 核心思想 | 公式 |
|------|---------|------|
| Add-1 (Laplace) | 所有计数 +1 | $P(w_i|w_{i-1}) = \frac{C(w_{i-1},w_i)+1}{C(w_{i-1})+V}$ |
| Add-K | 加一个小 K | 同上，K < 1 |
| Backoff | 回退到低阶 | …… |
| Interpolation | 线性插值各阶 | …… |
| Kneser-Ney | 考虑词的多样性 | …… |

---

## 2. RNN 核心推导

### 前向传播

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$
$$
\hat{y}_t = \text{softmax}(W_y h_t + b_y)
$$

### BPTT (Backpropagation Through Time)

> 待补充：完整的 BPTT 推导，展开 K 步的计算图 + 梯度流

### 梯度消失的数学分析

> 待补充：为什么 $\| \frac{\partial h_T}{\partial h_t} \| \approx \prod_{i=t}^{T-1} \| W_h \| \cdot \| \text{diag}(\tanh'(h_i)) \|$ 会衰减？

---

## 3. LSTM / GRU

### LSTM

| 门 | 公式 | 作用 |
|----|------|------|
| Forget gate | $f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$ | 决定遗忘哪些旧信息 |
| Input gate | $i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$ | 决定写入哪些新信息 |
| Output gate | $o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$ | 决定输出什么 |
| Cell update | $\tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$ | 候选新记忆 |
| New cell | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | 更新记忆 |
| Hidden | $h_t = o_t \odot \tanh(C_t)$ | 输出 |

### 为什么 LSTM 能缓解梯度消失？

> 待分析：cell state 的加法结构使得梯度可以"无损"传播

### LSTM vs GRU

| 维度 | LSTM | GRU |
|------|------|-----|
| 门数 | 3（forget + input + output） | 2（reset + update） |
| 参数 | 较多 | 较少（约 LSTM 的 3/4） |
| 适用场景 | 大数据集 | 小数据集可能更好 |

---

## 4. 依存解析 (Dependency Parsing)

### Transition-Based Parsing

#### Arc-Standard 转换系统

| Transition | Stack before | Stack after | Buffer before | Buffer after | Arc added |
|-----------|-------------|-------------|---------------|-------------|-----------|
| LEFT-ARC | $s_1, s_2, ...$ | $..., s_2$ | ... | ... | $s_2 \rightarrow s_1$ |
| RIGHT-ARC | $s_1, s_2, ...$ | $..., s_1$ | ... | ... | $s_1 \rightarrow s_2$ |
| SHIFT | ... | $..., s_1, b_1$ | $b_1, ...$ | ... | 无 |

### Neural Dependency Parser

> 待补充：Chen & Manning (2014) 的架构图与公式

---

## 5. Assignment 2 Part 2 记录

### 关键实现

- [ ] parser_transitions.py
- [ ] parser_model.py（MLP 打分）
- [ ] 训练与评估

> 待记录具体实现细节

---

## 本周小结

> 待完成

---

## 参考资料

- [CS224N Notes: Language Models and RNNs](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Karpathy, 2015)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Christopher Olah)
- [A Fast and Accurate Dependency Parser using Neural Networks](https://aclanthology.org/D14-1082.pdf) (Chen & Manning, 2014)
