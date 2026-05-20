# Week 2: 反向传播与神经网络基础

> 时间：第 2 周 | 阶段：一（词向量与基础） | 状态：🔴 未开始

---

## 本周目标

- [ ] 手推 softmax + cross-entropy 的完整梯度
- [ ] 掌握矩阵微积分技巧（分母布局 vs 分子布局）
- [ ] 理解计算图视角下的反向传播
- [ ] 掌握 PyTorch autograd 的原理
- [ ] 完成 Assignment 2（第一部分）：手算 tensor 导数

---

## 1. 反向传播的数学推导

### 1.1 Softmax 梯度推导

> 待补充完整推导：已知 $\hat{y} = \text{softmax}(\theta)$，求 $\frac{\partial \hat{y}}{\partial \theta}$

### 1.2 Cross-Entropy Loss 梯度

> 待补充：为什么 softmax + cross-entropy 的组合梯度异常简洁？

### 1.3 矩阵微积分速查

| 规则 | 公式 | 示例 |
|------|------|------|
| 线性 | $\frac{\partial Ax}{\partial x} = A^T$ | …… |
| 二次型 | $\frac{\partial x^T A x}{\partial x} = (A + A^T)x$ | …… |
| 链式法则 | $\frac{\partial f(g(x))}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$ | …… |

---

## 2. 计算图与自动微分

### 关键概念

- **前向传播**：沿计算图正向计算每个节点的值
- **反向传播**：沿计算图反向传播梯度
- **局部梯度**：每个节点只需知道自己的局部导数

### 手动示例

> 待补充：一个完整的前向+反向传播数值例子

---

## 3. PyTorch Autograd

### 核心机制

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2*2 + 3 = 7
```

### 思考题

1. `torch.no_grad()` 和 `detach()` 的区别？
2. `backward()` 的 `retain_graph=True` 什么场景需要？
3. 为什么 PyTorch 默认不保留中间梯度？

---

## 4. 神经网络基础

### 从逻辑回归到 MLP

| 模型 | 决策边界 | 激活函数 |
|------|---------|---------|
| Logistic Regression | 线性 | sigmoid / softmax |
| 1-Hidden MLP | 凸多边形 | sigmoid / tanh |
| Deep MLP | 任意复杂 | ReLU / GELU |

### 常见激活函数

| 函数 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | 平滑、可解释 | 梯度消失、非零中心 |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 零中心 | 梯度消失 |
| ReLU | $f(x) = \max(0, x)$ | 计算快、缓解梯度消失 | 神经元死亡 |
| GELU | $f(x) = x \cdot \Phi(x)$ | 平滑、适合 Transformer | 计算较慢 |

---

## 5. Assignment 2 第一部分 记录

### 手算 tensor 导数题

> 待记录解题过程

---

## 本周小结

> 待完成

---

## 参考资料

- [CS224N Notes: Neural Networks](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes03-neuralnets.pdf)
- [Matrix Calculus Notes](http://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf)
- [CS231N: Derivatives and Backpropagation](http://cs231n.stanford.edu/handouts/derivatives.pdf)
