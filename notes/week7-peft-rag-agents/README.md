# Week 7: 预训练 + 后训练 (SFT/RLHF/PEFT)

> 时间：第 7 周 | 阶段：四（LLM 全栈） | 状态：🔴 未开始

---

## 本周目标

- [ ] 理解 GPT 预训练的数据管线与 Scaling Law
- [ ] 掌握 SFT（Supervised Fine-Tuning）的流程
- [ ] 理解 RLHF 的三阶段：SFT → Reward Model → PPO
- [ ] 理解 DPO（Direct Preference Optimization）的简化思路
- [ ] 掌握 LoRA 原理并用 PEFT 库实践

---

## 1. 预训练（Pretraining）

### Scaling Law (Kaplan et al., 2020)

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

- $N$ = 参数量, $D$ = 数据量
- 参数量和数据量应等比扩大

### Chinchilla Scaling Law (Hoffmann et al., 2022)

> 待补充：与 Kaplan 的关键分歧——数据量和参数量应该同等重要

### 预训练数据管线

```
Raw Text → 清洗 → 去重 → 分词 (BPE/SentencePiece) → 混配 → Batching → 训练
```

---

## 2. 监督微调 (SFT)

### Concept

- 预训练学到的是"语言能力"，SFT 学到的是"遵循指令"
- 数据格式：(instruction, response) pair

### 关键技巧

- [ ] 多任务混合训练
- [ ] Chat template 的重要性
- [ ] 过拟合风险（SFT 数据通常很少，1K-10K）

---

## 3. RLHF (InstructGPT / ChatGPT)

### 三阶段流程

```
Phase 1: SFT
  Base Model + 高质量 instruction data → SFT Model

Phase 2: Reward Model (RM)
  SFT Model 生成多个 response → 人类排序 → 训练 RM

Phase 3: PPO
  SFT Model + RM 反馈 → PPO 强化学习 → 最终模型
```

### PPO 目标函数

$$
R(x, y) = r_{\phi}(x,y) - \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)}
$$

> 待补充：KL 散度惩罚项的意义

---

## 4. DPO (Direct Preference Optimization)

### 核心思想：把 RL 训练转化为分类任务

$$
\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

### RLHF vs DPO

| 维度 | RLHF | DPO |
|------|------|-----|
| 需要 Reward Model | 是（独立训练） | 否 |
| 训练稳定性 | 低（PPO 难调） | 高（直接优化） |
| 效果 | 更灵活 | 相近 |

---

## 5. PEFT: LoRA

### LoRA 原理

$$
W_{new} = W_{pretrained} + \Delta W = W_{pretrained} + B \cdot A
$$

- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$
- 只训练 $A, B$，参数量大幅减少

### 为什么 LoRA 有效？

> 待分析：微调时的权重更新 $\Delta W$ 是 low-rank 的假设

### PEFT 实践

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
```

### 其他 PEFT 方法

| 方法 | 核心思想 | 训练量 |
|------|---------|--------|
| Adapter | 插入小 FFN | 3-5% |
| Prefix Tuning | 学可训练 prefix | <1% |
| LoRA | 低秩分解 | <1% |
| QLoRA | LoRA + 4-bit 量化 | <1% |

---

## 本周小结

> 待完成

---

## 参考资料

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022) — Chinchilla
- [InstructGPT](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
