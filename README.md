# Surface Science Assistant

一个基于深度学习和大型语言模型的表面科学文献处理与智能问答系统，专门用于处理 PubMed 导出的文献数据，并构建针对表面科学（Surface Science）和表面合成（On-surface synthesis）领域的智能助手。

## 🌟 项目概览

本项目实现了从 PubMed 文献数据到智能问答系统的完整工作流程，包括数据预处理、模型训练、内容提取、知识库构建和 RAG（检索增强生成）问答系统。系统能够自动筛选、处理和分析表面科学相关文献，并提供专业的科学问答服务。

## 📂 项目结构

```
SurfaceScienceAssitant/
├── T5/                          # T5 模型：TXT 到 BibTeX 转换
│   ├── txt_bib_train.py        # T5 模型训练
│   └── txt_bib_predict.py      # T5 模型预测
├── Classification/             # 文献分类模块
│   ├── classification_class_2_bib_train.py  # 二分类训练（Surface Science 相关性）
│   ├── classification_class_3_bib_train.py  # 三分类训练（PPT/FT/Review）
│   ├── predict_bibtex_class_2.py           # 二分类预测
│   └── predict_bibtex_class_3.py           # 三分类预测
├── PDF2TXT/                    # PDF 文档处理
│   ├── PDF2md.py              # PDF 转 Markdown
│   └── md2txt.py              # Markdown 转纯文本
├── JSON/                       # JSON 结构化提取
│   └── prediction_json.py     # 使用 Claude 提取结构化数据
├── QA/                         # 问答数据生成
│   └── claude_qa_batch.py     # 使用 Claude 批量生成 QA 数据
├── PEFT/                       # 参数高效微调
│   ├── Lora_CoT.py           # LoRA 微调训练
│   └── Inference.py          # 微调后模型推理
└── RAG/                        # 检索增强生成
    └── RAG_router.py         # RAG 路由问答系统
```

## 🚀 主要功能模块

### 1. T5 文本转换模块 (`T5/`)

**功能**: 将 PubMed 导出的 TXT 格式文献元数据转换为标准 BibTeX 格式

**核心模型**: Google T5 ([Hugging Face 链接](https://huggingface.co/t5-base))

**特点**:
- 支持超参数优化（Optuna）
- 使用 ROUGE 指标评估生成质量
- 支持批量预测和增量处理
- 自动 DOI 匹配和验证

**使用方法**:
```bash
# 训练 T5 模型
python T5/txt_bib_train.py

# 使用训练好的模型进行预测
python T5/txt_bib_predict.py
```

### 2. 文献分类模块 (`Classification/`)

**功能**: 筛选与 Surface Science 和 On-surface synthesis 相关的文献

**核心模型**: SciBERT ([Hugging Face 链接](https://huggingface.co/allenai/scibert_scivocab_uncased))

**分类类型**:
- **二分类**: 判断文献是否与表面科学相关（YES/NO）
- **三分类**: 区分文献类型（PPT 实验/FT 理论/Review 综述）

**特点**:
- 基于 SciBERT 的领域专用预训练模型
- 支持 Focal Loss 处理类别不平衡
- 使用 k-fold 交叉验证
- 支持超参数优化

**使用方法**:
```bash
# 训练二分类模型
python Classification/classification_class_2_bib_train.py

# 训练三分类模型
python Classification/classification_class_3_bib_train.py

# 预测分类
python Classification/predict_bibtex_class_2.py --model_path path/to/model
python Classification/predict_bibtex_class_3.py --model_path path/to/model
```

### 3. PDF 处理模块 (`PDF2TXT/`)

**功能**: 将 PDF 文件转换为可处理的文本格式

**工具链**: Magic-PDF → Markdown → 纯文本

**特点**:
- 支持批量 PDF 处理
- 自动超时控制和错误处理
- 保留科学文献格式
- 清理 Markdown 标记，保留核心内容

**使用方法**:
```bash
# PDF 转 Markdown
python PDF2TXT/PDF2md.py

# Markdown 转纯文本
python PDF2TXT/md2txt.py
```

### 4. JSON 结构化提取模块 (`JSON/`)

**功能**: 使用 Claude 模型从文献文本中提取结构化信息

**核心模型**: Claude Sonnet 4.0 (Anthropic)

**特点**:
- 基于预定义 JSON 模板提取
- 支持批量处理和错误恢复
- 自动重试和超时控制
- 结构化验证和清理

**使用方法**:
```bash
python JSON/prediction_json.py
```

### 5. QA 数据生成模块 (`QA/`)

**功能**: 生成高质量的问答对用于模型训练

**核心模型**: Claude Sonnet 4.0 (Anthropic)

**特点**:
- 支持实时 API 和批处理 API（成本更低）
- 生成包含推理过程的 CoT 问答对
- 自动 JSONL 格式验证
- 支持增量处理和断点续传

**使用方法**:
```bash
# 实时模式
python QA/claude_qa_batch.py

# 批处理模式（推荐）
python QA/claude_qa_batch.py
# 检查批处理状态
python QA/claude_qa_batch.py check_batch <batch_id>
# 获取批处理结果
python QA/claude_qa_batch.py get_results <batch_id>
```

### 6. PEFT 微调模块 (`PEFT/`)

**功能**: 使用 LoRA 技术微调大语言模型

**核心模型**: Meta Llama 3.1 8B ([Hugging Face 链接](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B))

**技术特点**:
- 使用 LoRA (Low-Rank Adaptation) 进行参数高效微调
- 支持 4-bit 量化节省显存
- 集成 SwanLab 进行训练监控
- 支持思维链 (Chain-of-Thought) 训练

**训练配置**:
- LoRA rank: 8
- 量化: 4-bit NF4
- 序列长度: 4096
- 批量大小: 1（梯度累积 4 步）

**使用方法**:
```bash
# LoRA 微调训练
python PEFT/Lora_CoT.py

# 模型推理
python PEFT/Inference.py
```

### 7. RAG 问答系统 (`RAG/`)

**功能**: 构建检索增强生成的智能问答系统

**核心组件**:
- **LLM**: GPT-4.1 (OpenAI)
- **嵌入模型**: all-MiniLM-L6-v2
- **重排序**: BGE-reranker-large
- **向量数据库**: FAISS
- **检索器**: Hybrid (FAISS + BM25)

**智能路由**:
- **Role 1 (合成专家)**: 处理化学反应、合成方法相关问题（使用 JSON 数据库）
- **Role 2 (表征专家)**: 处理表面结构、电子态相关问题（使用 TXT 数据库）

**特点**:
- 自动问题分类路由
- 对话历史感知检索
- 混合检索 + 重排序
- 分片存储支持大规模数据

**使用方法**:
```bash
python RAG/RAG_router.py
```

## 📋 环境要求

### 基础依赖
```
python >= 3.8
torch >= 1.12.0
transformers >= 4.20.0
datasets >= 2.0.0
```

### 主要依赖包
```bash
pip install torch transformers datasets
pip install optuna scikit-learn rouge-score
pip install anthropic openai langchain
pip install faiss-cpu sentence-transformers
pip install bibtexparser magic-pdf
pip install peft trl swanlab
```

## 🛠️ 安装和使用

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥
在相应的配置文件中设置：
- Anthropic API Key (Claude)
- OpenAI API Key (GPT-4)

### 3. 准备数据
- 将 PubMed 导出的 TXT 文件放入指定目录
- 准备对应的 BibTeX 标注文件
- 配置各模块的数据路径

### 4. 运行完整工作流程
```bash
# 1. T5 模型训练和转换
python T5/txt_bib_train.py

# 2. 文献分类
python Classification/classification_class_2_bib_train.py

# 3. PDF 处理
python PDF2TXT/PDF2md.py
python PDF2TXT/md2txt.py

# 4. JSON 提取
python JSON/prediction_json.py

# 5. QA 生成
python QA/claude_qa_batch.py

# 6. 模型微调
python PEFT/Lora_CoT.py

# 7. RAG 系统
python RAG/RAG_router.py
```

## 📝 数据格式

### 输入格式
- **TXT**: PubMed 标准导出格式
- **BibTeX**: 标准学术文献格式
- **PDF**: 学术论文 PDF 文件

### 输出格式
- **JSON**: 结构化文献信息
- **JSONL**: 问答对数据
- **BibTeX**: 标准文献引用格式


## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情


**注意**: 本项目专为表面科学研究领域设计，使用前请确保已正确配置所有必需的 API 密钥和模型文件。