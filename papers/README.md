# Papers

> 论文精读笔记索引。每篇论文一份 Markdown，包含核心贡献、关键方法、公式推导、实验结果和我的思考。

## 必读论文（15 篇）

### 词向量

| # | 论文 | 年份 | 笔记 | 状态 |
|---|------|------|------|------|
| 1 | Efficient Estimation of Word Representations in Vector Space (Mikolov et al.) | 2013 | [→](word2vec-2013.md) | ⬜ |
| 2 | Distributed Representations of Words and Phrases (Mikolov et al.) | 2013 | [→](word2vec-neg-2013.md) | ⬜ |
| 3 | GloVe: Global Vectors for Word Representation (Pennington et al.) | 2014 | [→](glove-2014.md) | ⬜ |

### RNN 与语言模型

| # | 论文 | 年份 | 笔记 | 状态 |
|---|------|------|------|------|
| 4 | Learning Representations by Backpropagating Errors (Rumelhart et al.) | 1986 | [→](backprop-1986.md) | ⬜ |
| 5 | A Fast and Accurate Dependency Parser using Neural Networks (Chen & Manning) | 2014 | [→](chen-manning-2014.md) | ⬜ |

### Transformer

| # | 论文 | 年份 | 笔记 | 状态 |
|---|------|------|------|------|
| 6 | Attention Is All You Need (Vaswani et al.) | 2017 | [→](attention-2017.md) | ⬜ |
| 7 | BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al.) | 2019 | [→](bert-2019.md) | ⬜ |
| 8 | Improving Language Understanding by Generative Pre-Training (Radford et al.) | 2018 | [→](gpt1-2018.md) | ⬜ |
| 9 | Language Models are Unsupervised Multitask Learners (Radford et al.) | 2019 | [→](gpt2-2019.md) | ⬜ |

### LLM 训练与对齐

| # | 论文 | 年份 | 笔记 | 状态 |
|---|------|------|------|------|
| 10 | Scaling Laws for Neural Language Models (Kaplan et al.) | 2020 | [→](scaling-laws-2020.md) | ⬜ |
| 11 | Training Language Models to Follow Instructions (InstructGPT) (Ouyang et al.) | 2022 | [→](instructgpt-2022.md) | ⬜ |
| 12 | LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.) | 2021 | [→](lora-2021.md) | ⬜ |
| 13 | Direct Preference Optimization (Rafailov et al.) | 2023 | [→](dpo-2023.md) | ⬜ |

### RAG 与 Agent

| # | 论文 | 年份 | 笔记 | 状态 |
|---|------|------|------|------|
| 14 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al.) | 2020 | [→](rag-2020.md) | ⬜ |
| 15 | ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al.) | 2022 | [→](react-2022.md) | ⬜ |

## 选读论文

| # | 论文 | 笔记 | 状态 |
|---|------|------|------|
| 16 | Training Compute-Optimal Large Language Models (Chinchilla) | [→](chinchilla-2022.md) | ⬜ |
| 17 | QLoRA: Efficient Finetuning of Quantized LLMs | [→](qlora-2023.md) | ⬜ |
| 18 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | [→](llm-judge-2023.md) | ⬜ |

## 笔记模板

每篇 paper note 包含：

```
1. Motivation（为什么需要这篇工作）
2. Method（核心方法，带公式推导）
3. Experiments（关键实验结果与分析）
4. Limitations（作者承认的局限 + 我看到的局限）
5. Impact（对后续工作的影响）
6. My Thoughts（批判性思考）
```
