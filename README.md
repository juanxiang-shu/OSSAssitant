# Surface Science Assistant

A deep learning and large language model-based literature processing and intelligent Q&A system for surface science, specifically designed to handle PubMed-exported literature data and build an intelligent assistant for the fields of Surface Science and On-surface synthesis.

## 🌟 Project Overview

This project implements a complete workflow from PubMed literature data to an intelligent Q&A system, including data preprocessing, model training, content extraction, knowledge base construction, and a RAG (Retrieval-Augmented Generation) question-answering system. The system can automatically filter, process, and analyze surface science-related literature and provide professional scientific Q&A services.

## 📂 Project Structure

```
SurfaceScienceAssitant/
├── T5/                          # T5 model: TXT to BibTeX conversion
│   ├── txt_bib_train.py         # T5 model training
│   └── txt_bib_predict.py       # T5 model prediction
├── Classification/              # Literature classification module
│   ├── classification_class_2_bib_train.py # Binary classification training (Surface Science relevance)
│   ├── classification_class_3_bib_train.py # Ternary classification training (PPT/FT/Review)
│   ├── predict_bibtex_class_2.py           # Binary classification prediction
│   └── predict_bibtex_class_3.py           # Ternary classification prediction
├── PDF2TXT/                     # PDF document processing
│   ├── PDF2md.py                # PDF to Markdown
│   └── md2txt.py                # Markdown to plain text
├── JSON/                        # JSON structured extraction
│   └── prediction_json.py       # Use Claude to extract structured data
│   └── template.json            # template for JSON extracted
├── QA/                          # Q&A data generation
│   └── claude_qa_batch.py       # Use Claude to batch-generate QA data
├── PEFT/                        # Parameter-efficient fine-tuning
│   ├── Lora_CoT.py              # LoRA fine-tuning training
│   └── Inference.py             # Inference with fine-tuned model
└── RAG/                         # Retrieval-Augmented Generation
└── RAG_router.py            # RAG routing Q&A system
```

## 🚀 Main Functional Modules

### 1. T5 Text Conversion Module (`T5/`)

**Function**: Convert PubMed-exported TXT format metadata into standard BibTeX format

**Core Model**: Google T5 (Hugging Face link: https://huggingface.co/t5-base)

**Features**:
- Supports hyperparameter optimization (Optuna)
- Evaluates generation quality using ROUGE metrics
- Supports batch prediction and incremental processing
- Automatic DOI matching and validation

**Usage**:
```bash
# Train T5 model
python T5/txt_bib_train.py

# Predict with trained model
python T5/txt_bib_predict.py
```

### 2. Literature Classification Module (`Classification/`)

**Function**: Filter literature related to Surface Science and On-surface synthesis

**Core Model**: SciBERT ([Hugging Face link](https://huggingface.co/allenai/scibert_scivocab_uncased))

**Classification Types**:
- **Binary**: Determine whether the paper is related to surface science（YES/NO）
- **Ternary**: Distinguish paper type（surface chemistry/on-surface synthesis/Review）

**Features**:
- Domain-specific pre-trained model based on SciBERT
- Supports Focal Loss to handle class imbalance
- Uses k-fold cross-validation
- Supports hyperparameter optimization

**Usage**:
```bash
# Train binary classification model
python Classification/classification_class_2_bib_train.py

# Train ternary classification model
python Classification/classification_class_3_bib_train.py

# Predict classification
python Classification/predict_bibtex_class_2.py --model_path path/to/model
python Classification/predict_bibtex_class_3.py --model_path path/to/model
```

### 3. PDF Processing Module (`PDF2TXT/`)

**Function**: Convert PDF files into processable text format

**Toolchain**: Magic-PDF → Markdown → Plain text

**Features**:
- Supports batch PDF processing
- Automatic timeout control and error handling
- Preserves scientific literature formatting
- Cleans Markdown markup while retaining core content

**Usage**:
```bash
# PDF to Markdown
python PDF2TXT/PDF2md.py

# Markdown to plain text
python PDF2TXT/md2txt.py
```

### 4. JSON Structured Extraction Module (`JSON/`)

**Function**: Extract structured information from literature text using Claude model

**Core Model**: Claude Sonnet 4.0 (Anthropic)

**Features**:
- Extraction based on predefined JSON template
- Supports batch processing and error recovery
- Automatic retry and timeout control
- Structured validation and cleanup

**Usage**:
```bash
python JSON/prediction_json.py
```

### 5. QA Data Generation Module (`QA/`)

**Function**: Generate high-quality question-answer pairs for model training

**Core Model**: Claude Sonnet 4.0 (Anthropic)

**Features**:
- Supports real-time API and batch API (lower cost)
- Generates Chain-of-Thought (CoT) QA pairs with reasoning
- Automatic JSONL format validation
- Supports incremental processing and resume from breakpoint

**Usage**:
```bash
# Real-time mode
python QA/claude_qa_batch.py

# Batch mode (recommended)
python QA/claude_qa_batch.py
# Check batch status
python QA/claude_qa_batch.py check_batch <batch_id>
# Retrieve batch results
python QA/claude_qa_batch.py get_results <batch_id>
```

### 6. PEFT Fine-tuning Module (`PEFT/`)

**Function**: Fine-tune large language models using LoRA technology

**Core Model**: Meta Llama 3.1 8B ([Hugging Face link](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B))

**Technical Features**:
- Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Supports 4-bit quantization to save VRAM
- Integrates SwanLab for training monitoring
- Supports Chain-of-Thought (CoT) training

**Training Configuration**:
- LoRA rank: 8
- Quantization: 4-bit NF4
- Sequence length: 4096
- Batch size: 1 (gradient accumulation: 4 steps)

**Usage**:
```bash
# LoRA fine-tuning training
python PEFT/Lora_CoT.py

# Model inference
python PEFT/Inference.py
```

### 7. RAG Q&A System (`RAG/`)

**Function**: Build a retrieval-augmented generation intelligent Q&A system

**Core Components**:
- **LLM**: GPT-4.1 (OpenAI)
- **Embedding model**: all-MiniLM-L6-v2
- **Reranker**: BGE-reranker-large
- **Vector database**: FAISS
- **Retriever**: Hybrid (FAISS + BM25)

**Intelligent Routing**:
- **Role 1 (Synthesis Expert)**: Handles questions about chemical reactions and synthesis methods (uses JSON database)
- **Role 2 (Characterization Expert)**: Handles questions about surface structure and electronic states (uses TXT database)

**Features**:
- Automatic question classification and routing
- Conversation history-aware retrieval
- Hybrid retrieval + reranking
- Chunked storage supporting large-scale data

**Usage**:
```bash
python RAG/RAG_router.py
```

## 📋 Environment Requirements

### Basic Dependencies
```
python >= 3.8
torch >= 1.12.0
transformers >= 4.20.0
datasets >= 2.0.0
```

### Main Packages
```bash
pip install torch transformers datasets
pip install optuna scikit-learn rouge-score
pip install anthropic openai langchain
pip install faiss-cpu sentence-transformers
pip install bibtexparser magic-pdf
pip install peft trl swanlab
```

## 🛠️ Installation and Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
Set in the corresponding configuration files：
- Anthropic API Key (Claude)
- OpenAI API Key (GPT-4)

### 3. Prepare data
- Place PubMed-exported TXT files in the specified directory
- Prepare corresponding annotated BibTeX files
- Configure data paths for each module

### 4. Run the full workflow
```bash
# 1. T5 model training and conversion
python T5/txt_bib_train.py

# 2. Literature classification
python Classification/classification_class_2_bib_train.py

# 3. PDF processing
python PDF2TXT/PDF2md.py
python PDF2TXT/md2txt.py

# 4. JSON extraction
python JSON/prediction_json.py

# 5. QA generation
python QA/claude_qa_batch.py

# 6. Model fine-tuning
python PEFT/Lora_CoT.py

# 7. RAG system
python RAG/RAG_router.py
```

## 📝 Data Formats

### Input Formats
- **TXT**: Standard PubMed export format
- **BibTeX**: Standard academic citation format
- **PDF**: Academic paper PDF files

### Output Formats
- **JSON**: Structured literature information
- **JSONL**: Question-answer pair data
- **BibTeX**: Standard citation format


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

**Note**: This project is specifically designed for surface science research. Please ensure all required API keys and model files are properly configured before use.