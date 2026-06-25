<div align="center">

# DeepLearningWithRNN

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-13.2-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-IMDB-FF6F00?style=flat-square&logo=imdb&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-Shakespeare-FF6F00?style=flat-square&logo=bookstack&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p>基于 PyTorch 的 RNN / LSTM 文本任务实战系列</p>
<p>以 <b>IMDB 情感分类 → Shakespeare 字符级生成 → 子词级 LSTM</b> 为主线，循序渐进覆盖文本分类与语言建模</p>
<p>每个 Notebook 均配有详细中文注释，适合自然语言处理与循环神经网络学习</p>

</div>

---

## 📋 目录

- [项目概览](#-项目概览)
- [目录结构](#-目录结构)
- [Notebook 介绍](#-notebook-介绍)
- [数据集](#-数据集)
- [学习路径](#-学习路径)
- [环境依赖](#️-环境依赖)
- [快速开始](#-快速开始)

---

## 🗺 项目概览

| #   | Notebook | 核心技术 | 亮点 |
| --- | --- | --- | --- |
| 1   | IMDB 词级 Tokenization + Pooling | 词级分词 · Embedding · Global Pooling | 最小可用文本分类基线 |
| 2   | IMDB 双向 RNN（1 层） | BiRNN · Padding/Mask · Text Classification | 理解双向循环结构 |
| 3   | IMDB 双向 RNN（2 层） | Deep BiRNN · 多层堆叠 · 正则化 | 深层 RNN 表达能力提升 |
| 4   | Shakespeare 字符级 RNN 生成 | Char Tokenization · Language Modeling · 温度采样 | 从零实现文本生成流程 |
| 5   | IMDB 双向 LSTM（2 层） | BiLSTM · 门控机制 · Early Stopping | 对比 RNN 与 LSTM 效果 |
| 6   | Shakespeare 字符级 Tokenization + LSTM | Char Tokenization · LSTM · 自回归推理 | 更稳定的长依赖生成 |
| 7   | IMDB 子词级 Tokenization + LSTM | BPE 分词 · Subword Tokenization · BiLSTM | OOV 处理与子词建模 |

---

## 📁 目录结构

```
DeepLearningWithRNN/
├── 📓 1.IMDBText_classification_WordLevelTokenization_pooling.ipynb
├── 📓 2.IMDBText_classification_WordLevelTokenization_RNN1Layer2Direction.ipynb
├── 📓 3.IMDBText_classification_WordLevelTokenization_RNN2Layer2Direction.ipynb
├── 📓 4.ShakespeareText_generation_CharLevelTokenization_rnn.ipynb
├── 📓 5.IMDBText_classification_WordLevelTokenization_LSTM2Layer2Direction.ipynb
├── 📓 6.ShakespeareText_generation_CharLevelTokenization_lstm.ipynb
├── 📓 7.IMDBText_classification_SubwordLevelTokenization_LSTM.ipynb
│
├── 📂 data/
│   ├── imdb/                       # HuggingFace IMDB 本地缓存
│   ├── imdb_train.txt              # IMDB 训练文本（清洗后）
│   ├── imdb_test.txt               # IMDB 测试文本（清洗后）
│   ├── imdb_bpe_code               # BPE 学习到的合并规则
│   ├── imdb_bpe_vocab              # BPE 词表统计文件
│   ├── imdb_train_bpe.txt          # BPE 处理后的训练文本
│   ├── imdb_test_bpe.txt           # BPE 处理后的测试文本
│   ├── imdb_subwords.csv           # 子词级样本与标签整合文件
│   └── shakespeare.txt             # 莎士比亚语料
│
├── 📂 model_checkpoints/           # 各实验最优模型权重（.ckpt）
│   ├── 1_model/1_model_best.ckpt
│   ├── 2_model/2_model_best.ckpt
│   ├── 3_model/3_model_best.ckpt
│   ├── 4_model/4_model_best.ckpt
│   ├── 5_model/5_model_best.ckpt
│   ├── 6_model/6_model_best.ckpt
│   └── 7_model/7_model_best.ckpt
│
├── 📄 README.md
└── 📄 LICENSE
```

---

## 📚 Notebook 介绍

### 1.IMDB 词级 Tokenization + Pooling 分类基线

> `1.IMDBText_classification_WordLevelTokenization_pooling.ipynb`

以 IMDB 影评数据集为例，完成从文本清洗、词表构建、序列化到 Embedding + Pooling 分类器训练的完整流程，建立文本二分类起点模型。

| 章节 | 内容 |
| --- | --- |
| 数据准备 | 加载 IMDB · 提取文本与标签 · 清洗与分词 |
| 特征工程 | 构建 `word2idx/idx2word` · 长度分布分析 · Padding |
| 模型构建 | Embedding + 池化层 + 全连接输出 |
| 模型训练 | TensorBoard 可视化 · Save Best · Early Stop |
| 推理评估 | 测试集推理 · 指标统计 · 训练曲线 |

---

### 2.IMDB 双向 RNN（1 层）文本分类

> `2.IMDBText_classification_WordLevelTokenization_RNN1Layer2Direction.ipynb`

在词级输入基础上使用单层双向 RNN 进行序列建模，展示双向上下文融合对情感分类任务的增益。

| 章节 | 内容 |
| --- | --- |
| 数据准备 | IMDB 加载 · 文本清洗 · 词表构建 |
| 数据管道 | 自定义 Dataset · DataLoader · 序列对齐 |
| 模型定义 | 单层双向 RNN · 参数量对比 · 前向测试 |
| 模型训练 | 评估函数 · TensorBoard · 早停与最优权重保存 |
| 模型评估 | 加载最优模型 · 测试集准确率 |

---

### 3.IMDB 双向 RNN（2 层）文本分类

> `3.IMDBText_classification_WordLevelTokenization_RNN2Layer2Direction.ipynb`

将网络扩展为双层双向 RNN，比较深层循环网络在表示能力与收敛稳定性上的变化。

| 章节 | 内容 |
| --- | --- |
| 数据准备 | IMDB 数据解析 · 清洗分词 · 词典构建 |
| 数据管道 | Dataset/DataLoader 构建 · 长度分布分析 |
| 模型定义 | 两层双向 RNN 结构设计 · 参数量分析 |
| 模型训练 | 评估函数 · TensorBoard · Save Best · Early Stop |
| 结果评估 | 学习曲线可视化 · 测试集评估 |

---

### 4.Shakespeare 字符级 RNN 文本生成

> `4.ShakespeareText_generation_CharLevelTokenization_rnn.ipynb`

基于字符级建模训练语言模型，使用 RNN 学习 Shakespeare 语料分布，并通过温度采样与多项分布采样生成连续文本。

| 章节 | 内容 |
| --- | --- |
| 数据准备 | 字符词表构建 · 文本整数化编码 · 滑窗采样 |
| 模型构建 | CharRNN 定义 · 前向传播测试 |
| 模型训练 | 检查点回调 · 训练执行 · 损失曲线 |
| 文本生成 | 温度采样原理 · 多项分布采样 · 生成示例 |

---

### 5.IMDB 双向 LSTM（2 层）文本分类

> `5.IMDBText_classification_WordLevelTokenization_LSTM2Layer2Direction.ipynb`

将分类骨干替换为双向双层 LSTM，通过门控机制增强长依赖建模能力，并保持完整训练监控与回调体系。

| 章节 | 内容 |
| --- | --- |
| 数据准备 | IMDB 加载 · 文本清洗 · 词表构建 |
| 数据管道 | Tokenizer · Dataset/DataLoader |
| 模型定义 | BiLSTM 分类模型 |
| 模型训练 | TensorBoard · Save Best · Early Stop · 主循环 |
| 结果分析 | 可视化训练曲线 · 指标评估 |

---

### 6.Shakespeare 字符级 Tokenization + LSTM 生成

> `6.ShakespeareText_generation_CharLevelTokenization_lstm.ipynb`

在字符级生成任务中引入 Embedding + LSTM 架构，提升训练稳定性与生成文本连贯性，适合理解语言模型的标准实现路径。

| 章节 | 内容 |
| --- | --- |
| 环境配置 | 依赖导入 · 设备检测 · 版本记录 |
| 数据准备 | 语料读取 · 字符映射构建 · 数据集封装 |
| 模型定义 | CharLSTM 结构 · 参数量公式验证 |
| 模型训练 | Checkpoint 回调 · 训练执行 · 曲线可视化 |
| 推理生成 | 带温度采样的自回归文本生成 |

---

### 7.IMDB 子词级 Tokenization + LSTM 文本分类

> `7.IMDBText_classification_SubwordLevelTokenization_LSTM.ipynb`

引入 BPE 子词分词（subword-nmt）构建子词级输入，缓解 OOV 问题，并通过 BiLSTM 完成 IMDB 二分类训练与评估。

| 章节 | 内容 |
| --- | --- |
| 数据准备 | IMDB 下载 · 清洗 · 文本落盘 |
| BPE 预处理 | 学习 BPE 规则 · 分词应用 · 结果检查 |
| 数据管道 | 子词词表构建 · Tokenizer · Dataset/DataLoader |
| 模型定义 | 子词级 Embedding + BiLSTM 分类器 |
| 模型训练 | TensorBoard · Save Best · Early Stop · 主循环 |
| 推理评估 | 学习曲线 · 测试集指标 |

> **BPE 原理** · 通过高频字符对迭代合并得到子词单元，使模型在词级语义与字符级泛化之间取得平衡，显著改善低频词与未登录词处理能力。  
> 参考论文：*Neural Machine Translation of Rare Words with Subword Units*（Sennrich et al., 2016）

---

## 🗃 数据集

**IMDB Reviews** · 来自 [HuggingFace Datasets / stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)

| 属性 | 详情 |
| --- | --- |
| 任务类型 | 二分类情感分析（正面 / 负面） |
| 训练集 | 25,000 条 |
| 测试集 | 25,000 条 |
| 文本形式 | 英文影评长文本 |
| 使用方式 | 通过 `datasets.load_dataset("stanfordnlp/imdb")` 自动下载并缓存到 `data/imdb/` |

---

**Shakespeare Corpus** · 字符级文本生成语料

| 属性 | 详情 |
| --- | --- |
| 任务类型 | 字符级语言建模 / 文本生成 |
| 数据文件 | `data/shakespeare.txt` |
| 文本规模 | 约百万字符级别（视语料版本） |
| 典型输入 | 固定长度字符窗口 |
| 典型输出 | 下一字符预测 |

---

## 🛤 学习路径

```
Notebook 1          Notebook 2          Notebook 3
IMDB 词级基线    →   IMDB BiRNN 1层   →   IMDB BiRNN 2层
Pooling 分类          序列建模入门          深层循环建模
      │
      ▼
Notebook 5          Notebook 7
IMDB BiLSTM      →   IMDB BPE + BiLSTM
门控网络建模          子词级建模与 OOV 处理

并行分支（文本生成）：
Notebook 4          Notebook 6
Char-RNN         →   Char-Tokenization + LSTM
字符级生成入门        更稳定的字符级生成
```

建议先按 `1 → 2 → 3 → 5 → 7` 完成文本分类主线，再学习 `4 → 6` 文本生成支线。

---

## ⚙️ 环境依赖

| 包名 | 版本 | 用途 |
| --- | --- | --- |
| `torch` | 2.12.0+cu132 | 深度学习框架核心（CUDA 13.2） |
| `torchvision` | 0.27.0+cu132 | 张量与视觉工具（部分 Notebook 环境统一依赖） |
| `tensorboard` | 2.20.0 | 训练过程实时可视化 |
| `matplotlib` | 3.10.9 | 损失曲线与统计图绘制 |
| `numpy` | 2.4.6 | 数值计算 |
| `pandas` | 3.0.3 | 文本与标签表格处理 |
| `scikit-learn` | 1.9.0 | 准确率等评估指标 |
| `Pillow` | 12.2.0 | 图像工具依赖（环境通用） |
| `tqdm` | 4.68.2 | 训练进度条显示 |
| `datasets` | 5.0.0 | HuggingFace 数据集加载 |
| `subword-nmt` | 0.3.8 | BPE 子词规则学习与分词 |

---

## 🚀 快速开始

**1. 克隆仓库**

```bash
git clone https://github.com/your-username/DeepLearningWithRNN.git
cd DeepLearningWithRNN
```

**2. 安装依赖**

```bash
pip install torch torchvision tensorboard matplotlib numpy pandas scikit-learn pillow tqdm datasets subword-nmt
```

> ⚠️ `torch` 与 `torchvision` 带有 `+cu132`（CUDA 13.2）后缀，如需 CPU 版本或其他 CUDA 版本，请参考 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 替换。

**3. 准备数据**

- **IMDB**：首次运行相关 Notebook 自动下载，缓存位于 `data/imdb/`。
- **Shakespeare**：确保 `data/shakespeare.txt` 可用。
- **BPE 中间文件**：运行 `7` 号 Notebook 后将自动生成到 `data/`。

**4. 启动 Jupyter**

```bash
jupyter notebook
```

按学习路径打开 Notebook 即可开始实验。

---

## 📄 License

[MIT](LICENSE) © 2026
