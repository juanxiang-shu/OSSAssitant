# Surface Science Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> A comprehensive deep learning and large language model-based literature processing and intelligent Q&A system for surface science, specifically designed for on-surface synthesis research.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Datasets and Model Weights](#datasets-and-model-weights)
- [Quick Start](#quick-start)
- [License](#license)

---

## 🌟 Overview

This project implements a complete end-to-end workflow for processing scientific literature from PubMed and building an intelligent question-answering system for **Surface Science** and **On-surface Synthesis** research. The system integrates state-of-the-art NLP models including T5, SciBERT, LLaMA, and GPT-4 to enable:

- Automated literature filtering and classification
- Structured information extraction from scientific papers
- High-quality QA pair generation for model training
- Domain-specific fine-tuned language models
- Retrieval-Augmented Generation (RAG) for general and synthesis Q&A

**Target Domain**: Surface chemistry, on-surface synthesis, scanning probe microscopy (STM/AFM), 2D materials synthesis

---

## ✨ Key Features

- **🔄 Complete Pipeline**: From raw PubMed data to production RAG system
- **🎯 Domain-Specific**: Tailored for surface science literature
- **🤖 Multi-Model Integration**: T5, SciBERT, LLaMA 3.1, Claude Sonnet 4 and GPT-4
- **💡 Intelligent QA**: Chain-of-Thought reasoning in generated answers
- **🔍 Hybrid Retrieval**: FAISS + BM25 with BGE reranking
- **⚡ Efficient Training**: LoRA-based parameter-efficient fine-tuning

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: PubMed Literature                     │
│                     (TXT format metadata)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 1: T5 Text Conversion                                   │
│  • Convert TXT → BibTeX format                                  │
│  • Model: Google T5-base                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 2: Literature Classification                            │
│  • Binary: Surface Science relevance (Yes/No)                   │
│  • Model: SciBERT                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 3: PDF Processing                                       │
│  • PDF → Markdown → Plain Text                                  │
│  • Tool: Magic-PDF                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 4: Structured Extraction                                │
│  • Extract: Precursors, Reactions, Conditions                   │
│  • Model: Claude Sonnet 4                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 5: QA Generation                                        │
│  • Generate Chain-of-Thought QA pairs                           │
│  • Model: Claude Sonnet 4 (Batch API)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 6: Model Fine-tuning                                    │
│  • LoRA fine-tuning on domain QA data                           │
│  • Base Model: LLaMA 3.1 8B                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 7: RAG Q&A System                                       │
│  • Hybrid Retrieval (FAISS + BM25)                              │
│  • Intelligent Routing (Synthesis/Characterization)             │
│  • LLM: GPT-4 Turbo                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 11.7+ (for GPU acceleration, recommended)
- **RAM**: 32GB+ recommended
- **GPU**: NVIDIA GPU with 16GB+ VRAM for Inference and NVIDIA H800 GPU for fine-tuning

### Environment Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/juanxiang-shu/OSSAssitant.git
cd OSSAssitant
```

#### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n oss-assistant python=3.10
conda activate oss-assistant

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.30.0` - Hugging Face transformers
- `peft>=0.4.0` - Parameter-Efficient Fine-Tuning
- `bitsandbytes` - 4-bit quantization support
- `langchain>=0.1.0` - RAG framework
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `anthropic` - Claude API client
- `openai` - OpenAI API client
- `magic-pdf` - PDF processing
- `bibtexparser` - BibTeX format handling
- `optuna` - Hyperparameter optimization

#### 4. Download Required Models

```bash
# T5 base model (for text conversion)
mkdir -p models
cd models
# Download from: https://huggingface.co/t5-base

# SciBERT (for classification)
# Download from: https://huggingface.co/allenai/scibert_scivocab_uncased

# LLaMA 3.1 8B (for fine-tuning)
# Download from: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
# Note: Requires accepting Meta's license agreement

# Embedding model (for RAG)
# Download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Reranker model (for RAG)
# Download from: https://huggingface.co/BAAI/bge-reranker-large
```

#### 5. Configure API Keys

Create a `.env` file in the project root:

```bash
# Claude API (for QA generation and JSON extraction)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI API (for RAG system)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: HuggingFace token (for gated models)
HUGGINGFACE_TOKEN=your_hf_token_here
```

Or export as environment variables:

```bash
export ANTHROPIC_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
```

---

## 📦 Datasets and Model Weights

### Pre-trained Models

| Model | Purpose | Download Link | Size |
|-------|---------|---------------|------|
| T5-base | TXT→BibTeX conversion | [HuggingFace](https://huggingface.co/t5-base) | 4.15 MB |
| SciBERT | Literature classification | [HuggingFace](https://huggingface.co/allenai/scibert_scivocab_uncased) | 841 MB |
| LLaMA 3.1 8B | Base model for fine-tuning | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) | 29.9 GB |
| all-MiniLM-L6-v2 | Embedding model | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 931 MB |
| BGE-reranker-large | Reranking model | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-large) | 3.27 GB |

### Fine-tuned Model Weights

**🔗 Download Links**: [SurfaceScienceAssistant on HuggingFace](https://huggingface.co/JuanXiang-SHU/SurfaceScienceAssistant)

---

## 🎯 Quick Start

This section provides a complete, step-by-step example to reproduce the entire pipeline from scratch.

### Prerequisites

```bash
# Ensure environment is set up
conda activate oss-assistant
export ANTHROPIC_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
```

### Step 1: Data Preparation

```bash
# Create directory structure
mkdir -p data/{classified,json_extraction,qa_generation,rag,t5}

# Download sample data
wget https://raw.githubusercontent.com/juanxiang-shu/OSSAssitant/main/examples/t5/pubmed_sample.txt -O data/raw/pubmed.txt
wget https://raw.githubusercontent.com/juanxiang-shu/OSSAssitant/main/examples/qa_generation/sample_article.txt -O data/texts/sample_article.txt
wget https://raw.githubusercontent.com/juanxiang-shu/OSSAssitant/main/examples/qa_generation/sample_article.json -O data/structured/sample_article.json
wget "https://raw.githubusercontent.com/juanxiang-shu/OSSAssitant/main/examples/rag/json/sonogashira%20cross-coupling%20and%20homocoupling%20on%20a%20silver%20surface%20chlorobenzene%20and%20phenylacetylene%20on%20Ag100.json" -O data/knowledge_base/example_surface_reaction.json
wget "https://raw.githubusercontent.com/juanxiang-shu/OSSAssitant/main/examples/rag/txt/sonogashira%20cross-coupling%20and%20homocoupling%20on%20a%20silver%20surface%20chlorobenzene%20and%20phenylacetylene%20on%20Ag100.txt" -O data/knowledge_base/example_surface_reaction.txt
```

### Step 2: Convert PubMed Data

```bash
# Edit the configuration variables (T5_MODEL_PATH, TOTEL_LABELED_TXT, etc.)
# at the top of T5/txt_bib_train.py before running.
# Train T5 model (if needed) or use pre-trained weights
python T5/txt_bib_train.py

# Edit the configuration variables (INPUT_TXT_PATH, OUTPUT_BIB_PATH, T5_MODEL_PATH, etc.)
# at the top of T5/txt_bib_predict.py before running.
# Convert new data
python T5/txt_bib_predict.py
```

### Step 3: Classify Literature

```bash
# Edit the yes_path, no_path and model_path variables inside
# Classification/classification_class_2_bib_train.py before running.
# Train classifiers (if needed) or use pre-trained weights
python Classification/classification_class_2_bib_train.py

# Classify papers
python Classification/predict_bibtex_class_2.py \
  --model_path models/scibert_class2_best \
  --pretrained_model models/scibert \
  --test_data data/converted/converted.bib \
  --output_dir data/classified
```

### Step 4: Process PDFs

```bash
# Edit input_root and output_root variables at the top of PDF2TXT/PDF2md.py before running.
# Convert PDFs to Markdown
python PDF2TXT/PDF2md.py

# Edit input_directory and output_directory variables in the main() function
# of PDF2TXT/md2txt.py before running.
# Convert Markdown to plain text
python PDF2TXT/md2txt.py
```

### Step 5: Extract Structured Data

```bash
# Edit TXT_PATH and SAVE_PREDICTIONS in the CONFIG dict
# at the top of JSON/prediction_json.py before running.
# Extract structured information using Claude
python JSON/prediction_json.py
```

### Step 6: Generate QA Pairs

```bash
# Edit TXT_PATH, JSON_PATH, SAVE_QA_DATA, and USE_BATCH_API in the CONFIG dict
# at the top of QA/claude_qa_batch.py before running.
# Generate QA pairs (batch API is cost-effective when USE_BATCH_API is True)
python QA/claude_qa_batch.py
```

### Step 7: Fine-tune Model

```bash
# Edit local_model_path and other path variables at the top of PEFT/Lora_CoT.py
# before running.
# Fine-tune LLaMA 3.1 8B with LoRA
python PEFT/Lora_CoT.py

# Monitor training (in another terminal)
# SwanLab: http://localhost:5092

# Edit BASE_MODEL_PATH and LORA_CHECKPOINT_PATH at the top of PEFT/Inference.py
# before running.
# Test inference
python PEFT/Inference.py
```

### Step 8: Build RAG System

```bash
# Prepare knowledge bases
cp data/texts/*.txt data/knowledge_base/txt/
cp data/structured/*.json data/knowledge_base/json/

# Edit KNOWLEDGE_DIR and SYNTHESIS_KNOWLEDGE_DIR at the top of RAG/RAG_router.py
# before running.
# Build FAISS indices and run system
python RAG/RAG_router.py

# Test queries
# Q: What is the typical temperature for Ullmann coupling on Au(111)?
# Q: How does nc-AFM provide bond-order contrast?
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
