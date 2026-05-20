# Experiments

> 关键模型的从零复现实验。

| 实验 | 目录 | 对应周次 | 状态 |
|------|------|---------|------|
| Word2Vec from Scratch | [word2vec/](word2vec/) | Week 1 | 🔴 |
| Transformer from Scratch | [transformer-from-scratch/](transformer-from-scratch/) | Week 5 | 🔴 |
| GPT-2 (Default Project) | [gpt2/](gpt2/) | Week 9-10 | 🔴 |

## 实验目录结构

```
experiments/
├── word2vec/
│   ├── skipgram.py        # Skip-gram + Negative Sampling
│   ├── cbow.py            # CBOW 实现
│   ├── utils.py           # 数据预处理、词表构建
│   └── results/           # 词类比结果 + 可视化
│
├── transformer-from-scratch/
│   ├── attention.py       # Scaled Dot-Product + Multi-Head Attention
│   ├── layers.py          # FFN, LayerNorm, Residual
│   ├── transformer.py     # 完整 Transformer Encoder-Decoder
│   ├── train.py           # 训练脚本（copy task / 小翻译）
│   └── results/           # 训练曲线 + 注意力可视化
│
└── gpt2/
    ├── model.py           # GPT-2 完整架构
    ├── generation.py      # 解码策略实现
    ├── load_weights.py    # HuggingFace 权重转换
    ├── evaluate.py        # 下游任务评估
    └── results/           # 评估结果 + 生成样例
```

## 实验目标

每个实验的目标是**用最少的外部依赖（仅 NumPy + PyTorch）从零实现**，不使用 HuggingFace 的高级 API（模型加载和评估除外）。
