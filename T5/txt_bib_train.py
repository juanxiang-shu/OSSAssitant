# -*- coding: utf-8 -*-
import os
import json
import time
import random
import zipfile
import re
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5TokenizerFast as T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EvalPrediction
)
from sklearn.model_selection import train_test_split
import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from rouge_score import rouge_scorer
import optuna

# ========== Configuration ==========
T5_ZIP_PATH = "./models/t5.zip"
T5_MODEL_PATH = "./models/t5"
SAVE_MODEL_PATH = "./models/t5_txt2bib_best"
TOTEL_LABELED_TXT = "/OSS_Database/PubMed/totel.txt"
TOTEL_LABELED_BIB = "/OSS_Database/PubMed/totel.bib"
TOTEL_PREDICT_TXT = "/OSS_Database/PubMed/predict.txt"
BEST_PARAMS_PATH = "./best_params.json"

NUM_LABELED = 525
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 1024
SEED = 42
NUM_TRIALS = 3

INPUT_TEMPLATE = "Convert the following PubMed article to BibTeX format: {text}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def entry_to_json(entry: Dict) -> str:
    data = {
        "ENTRYTYPE": entry.get("ENTRYTYPE", "article"),
        "ID": entry.get("ID", "UNKNOWN"),
    }
    for fld in ["title", "author", "journal", "year", "volume", "number", "pages", "doi", "abstract"]:
        if fld in entry and entry[fld].strip():
            data[fld] = entry[fld].strip()
    return json.dumps(data, ensure_ascii=False)


def json_to_bib(json_str):
    """
    Enhanced JSON-to-BibTeX converter, specifically handling trailing quote issues
    at the end of the abstract field.
    """
    print(f"Original JSON length: {len(json_str)}")
    print(f"Last 100 characters of JSON: {repr(json_str[-100:])}")

    # 1. First, clean all extra quotes at the end of the string
    # This is the key step: remove all consecutive quotes at the end of the JSON string
    json_str = re.sub(r'"+\s*$', '', json_str.strip())

    print(f"JSON length after cleaning: {len(json_str)}")
    print(f"Last 50 characters after cleaning: {repr(json_str[-50:])}")

    # 2. Ensure the JSON ends with '}', add it if missing
    if not json_str.strip().endswith('}'):
        json_str += '}'

    # 3. Handle possible escaping issues inside the abstract field
    # Unescape any \" to "
    json_str = json_str.replace('\\"', '"')

    # 4. Try to parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Print detailed error info for debugging
        error_pos = e.pos
        start = max(0, error_pos - 50)
        end = min(len(json_str), error_pos + 50)
        error_context = json_str[start:end]

        print("JSON parsing error details:")
        print(f"Error position: {error_pos}")
        print(f"Error message: {e.msg}")
        print(f"Error context: {repr(error_context)}")
        print(f"Error character: {repr(json_str[error_pos] if error_pos < len(json_str) else 'EOF')}")

        # Attempt to fix common JSON formatting problems
        fixed_json = json_str

        try:
            # If still failing, fall back to manual parsing for key fields
            data = manual_parse_json(json_str)
        except Exception:
            raise ValueError(f"JSON parsing failed: {e}\nContext near error: {error_context}")

    # 5. Extract core fields and generate BibTeX
    entry_type = data.get('ENTRYTYPE')
    entry_id = data.get('ID')

    if not entry_type or not entry_id:
        raise ValueError("JSON must contain ENTRYTYPE and ID fields")

    bib_lines = [f"@{entry_type}{{{entry_id},"]

    # Preferred field order
    field_order = ['title', 'author', 'journal', 'year', 'volume', 'number', 'pages', 'doi', 'abstract']

    # Collect all fields and sort them
    all_fields = set(field_order + [k for k in data.keys() if k not in ['ENTRYTYPE', 'ID']])
    fields_to_include = sorted(all_fields, key=lambda x: field_order.index(x) if x in field_order else len(field_order))

    for field in fields_to_include:
        if field in data and str(data[field]).strip():
            value = str(data[field])
            # Escape BibTeX special characters
            value = value.replace('{', '{{').replace('}', '}}').replace('&', '\\&')
            bib_lines.append(f"  {field} = {{{value}}},")

    # Remove trailing comma from the last field
    if len(bib_lines) > 1:
        bib_lines[-1] = bib_lines[-1].rstrip(',')

    bib_lines.append('}')

    return '\n'.join(bib_lines)


def manual_parse_json(json_str):
    """
    Manually parse a problematic JSON-like string to salvage fields.
    """
    data = {}

    patterns = {
        'ENTRYTYPE': r'"ENTRYTYPE":\s*"([^"]*)"',
        'ID': r'"ID":\s*"([^"]*)"',
        'title': r'"title":\s*"([^"]*)"',
        'author': r'"author":\s*"([^"]*)"',
        'journal': r'"journal":\s*"([^"]*)"',
        'year': r'"year":\s*"([^"]*)"',
        'volume': r'"volume":\s*"([^"]*)"',
        'number': r'"number":\s*"([^"]*)"',
        'pages': r'"pages":\s*"([^"]*)"',
        'doi': r'"doi":\s*"([^"]*)"',
        'abstract': r'"abstract":\s*"(.*?)(?="[^"]*$|$)',
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            data[field] = match.group(1)

    return data


class PubmedToBibDataProcessor:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def extract_t5_model(self):
        """Unzip the T5 model archive."""
        # Ensure the model directory exists
        os.makedirs(os.path.dirname(T5_MODEL_PATH), exist_ok=True)

        if os.path.exists(T5_MODEL_PATH):
            print(f"✅ T5 model already exists: {T5_MODEL_PATH}")
            return

        if not os.path.exists(T5_ZIP_PATH):
            print(f"❌ T5 zip archive not found: {T5_ZIP_PATH}")
            print("Please make sure the T5 model file exists!")
            raise FileNotFoundError(f"T5 model archive does not exist: {T5_ZIP_PATH}")

        print(f"📦 Extracting T5 model from: {T5_ZIP_PATH}")
        with zipfile.ZipFile(T5_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall('./models/')
        print(f"✅ T5 model extraction completed: {T5_MODEL_PATH}")

    def load_t5_model(self):
        """Load the T5 model and tokenizer."""
        self.extract_t5_model()

        print("🔄 Loading T5 tokenizer and model...")
        self.tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH, use_fast=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)
        print("✅ T5 model loaded successfully")

    def split_articles_by_blank(self, content: str) -> List[str]:
        return [rec.strip() for rec in re.split(r'\n{3,}', content) if rec.strip()]

    def extract_doi_from_txt(self, record: str) -> str:
        """
        Extract DOI from a TXT record:
        - Supports both "doi: 10.xxx..." and "doi:\n10.xxx..." styles
        - Returns lowercase DOI (with trailing punctuation removed), or None if not found
        """
        m = re.search(
            r'doi[:\s]*\r?\n*\s*(10\.\d{4,9}/\S+)',
            record,
            flags=re.IGNORECASE
        )
        if not m:
            return None
        doi = m.group(1).rstrip('.,;')
        return doi.lower()

    def extract_doi_from_bib_entry(self, entry: Dict) -> str:
        doi = entry.get('doi', '').strip()
        return doi.lower() if doi else None

    def load_txt_records(self, path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            return self.split_articles_by_blank(f.read())

    def load_bib_entries(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            db = bibtexparser.load(f)
        return db.entries

    def bib_entry_to_string(self, entry: Dict) -> str:
        """Convert a BibTeX entry dict to a standard BibTeX-formatted string."""
        entry_type = entry.get('ENTRYTYPE', 'article')
        entry_id = entry.get('ID', 'unknown')

        bib_lines = [f"@{entry_type}{{{entry_id},"]

        field_order = ['title', 'author', 'journal', 'year', 'volume', 'number', 'pages', 'doi', 'abstract']

        for field in field_order:
            if field in entry and entry[field]:
                value = entry[field].strip()
                bib_lines.append(f"  {field} = {{{value}}},")

        for key, value in entry.items():
            if key not in field_order + ['ENTRYTYPE', 'ID'] and value:
                value = str(value).strip()
                bib_lines.append(f"  {key} = {{{value}}},")

        if len(bib_lines) > 1:
            bib_lines[-1] = bib_lines[-1].rstrip(',')
        bib_lines.append("}")

        return '\n'.join(bib_lines)

    def create_training_pairs(self) -> List[Dict[str, str]]:
        txt_recs = self.load_txt_records(TOTEL_LABELED_TXT)
        bib_entries = self.load_bib_entries(TOTEL_LABELED_BIB)

        bib_by_doi = {}
        for ent in bib_entries:
            doi = self.extract_doi_from_bib_entry(ent)
            if doi:
                bib_by_doi[doi] = ent

        pairs = []
        missing = []
        for rec in txt_recs:
            doi = self.extract_doi_from_txt(rec)
            if doi and doi in bib_by_doi:
                input_text = INPUT_TEMPLATE.format(text=rec)
                output_text = entry_to_json(bib_by_doi[doi])
                # output_text = self.bib_entry_to_string(bib_by_doi[doi])
                pairs.append({"input": input_text, "output": output_text})
            else:
                missing.append((doi, rec[:80]))

        print(f"✅ Successfully matched {len(pairs)} TXT–BibTeX pairs")
        if missing:
            print(f"⚠️ There are {len(missing)} TXT records without a matching DOI in the BibTeX file:")
            for doi, snippet in missing[:5]:
                print(f"   DOI={doi} → {snippet} ...")

        return pairs

    @staticmethod
    def load_prediction_texts(path: str) -> List[str]:
        """Load text records to be predicted."""
        with open(path, 'r', encoding='utf-8') as f:
            return [rec.strip() for rec in re.split(r'\n{3,}', f.read()) if rec.strip()]


class T5BibDataset(Dataset):
    """Dataset class for T5 training on TXT→Bib JSON conversion."""
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_encoding = self.tokenizer(
            item["input"],
            truncation=True,
            padding="max_length",
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            item["output"],
            truncation=True,
            padding="max_length",
            max_length=MAX_OUTPUT_LENGTH,
            return_tensors="pt"
        )
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels
        }


tokenizer_global = None


def compute_metrics(eval_pred: EvalPrediction):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    preds = np.where(preds < 0, tokenizer_global.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer_global.pad_token_id)

    # Decode and compute ROUGE
    pred_list = preds.tolist()
    label_list = labels.tolist()
    decoded_preds = tokenizer_global.batch_decode(pred_list, skip_special_tokens=True)
    decoded_labels = tokenizer_global.batch_decode(label_list, skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    for p, l in zip(decoded_preds, decoded_labels):
        scores = scorer.score(l, p)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)

    return {
        "rouge1": np.mean(rouge1),
        "rouge2": np.mean(rouge2),
        "rougeL": np.mean(rougeL),
    }


def train_model_with_optuna(train_dataset, val_dataset, trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.1, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)
    generation_num_beams = trial.suggest_categorical("generation_num_beams", [2, 4])
    generation_max_length = trial.suggest_int("generation_max_length", 512, 1024)

    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f'./results/trial_{trial.number}',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        report_to="none",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='rougeL',
        greater_is_better=True,
        seed=SEED,
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        generation_num_beams=generation_num_beams,
        dataloader_num_workers=2,
        fp16=True,
        max_grad_norm=1.0,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # save_only_model=True,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    best_metric = trainer.state.best_metric

    eval_results = trainer.evaluate()
    with open("optuna_trial_metrics.log", 'a', encoding='utf-8') as f:
        log_entry = {
            "trial_number": trial.number,
            "metrics": eval_results,
            "params": trial.params,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_metric


def main():
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./final_results', exist_ok=True)
    os.makedirs('./final_logs', exist_ok=True)
    global tokenizer_global
    print("🚀 Starting T5 training for TXT→Bib conversion")
    print("=" * 60)

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")

    log_file = "training_metrics.log"
    # Initialize log file if it does not exist
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    # Create data processor and load tokenizer/model
    processor = PubmedToBibDataProcessor()
    processor.load_t5_model()
    tokenizer_global = processor.tokenizer

    # Prepare training data
    print("📊 Preparing training data...")
    training_pairs = processor.create_training_pairs()

    # Sanity check for training data
    if len(training_pairs) == 0:
        print("❌ No valid training pairs found. Please check the data files and DOI matching.")
        return

    # Split data (8:1:1)
    train_data, temp_data = train_test_split(training_pairs, test_size=0.2, random_state=SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

    print("📈 Dataset split:")
    print(f"   Train set: {len(train_data)} samples")
    print(f"   Validation set: {len(val_data)} samples")
    print(f"   Test set: {len(test_data)} samples")

    # Create datasets
    train_dataset = T5BibDataset(train_data, processor.tokenizer)
    val_dataset = T5BibDataset(val_data, processor.tokenizer)
    test_dataset = T5BibDataset(test_data, processor.tokenizer)

    # Load best hyperparameters if available
    best_params = None
    if os.path.exists(BEST_PARAMS_PATH):
        print("\n🔍 Found best-parameters file; will use them directly for training.")
        with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
            best_params = json.load(f)

        print("🏆 Loaded best hyperparameters:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
    else:
        print("\n🔍 No best-parameters file found; starting hyperparameter optimization with Optuna...")
        # Run Optuna hyperparameter search
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: train_model_with_optuna(train_dataset, val_dataset, trial),
            n_trials=NUM_TRIALS
        )

        print("🏆 Best hyperparameters found by Optuna:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")

        best_trial_number = study.best_trial.number
        with open("./best_trial_number.txt", 'w') as f:
            f.write(str(best_trial_number))

        # Save best parameters for reuse
        best_params = study.best_params
        with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        print(f"💾 Best hyperparameters saved to: {BEST_PARAMS_PATH}")

    best_trial_number = None
    if os.path.exists("./best_trial_number.txt"):
        with open("./best_trial_number.txt", 'r') as f:
            best_trial_number = int(f.read().strip())

    # Load the best model checkpoint from the best trial
    if best_trial_number is not None:
        trial_dir = f"./results/trial_{best_trial_number}"
        if os.path.exists(trial_dir):
            checkpoints = [d for d in os.listdir(trial_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                # Use the latest checkpoint
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                best_model_path = os.path.join(trial_dir, latest_checkpoint)
                print(f"🔄 Loading model from best trial checkpoint: {best_model_path}")
                model = T5ForConditionalGeneration.from_pretrained(best_model_path)
            else:
                print("❌ No checkpoint found for the best trial")
                return
        else:
            print("❌ Best trial directory not found")
            return
    else:
        print("❌ Best trial number not found")
        return

    model.config.pad_token_id = processor.tokenizer.eos_token_id
    model.to(device)

    # Ensure save directory exists
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    def manual_evaluate_model(model, test_dataset, tokenizer, device):
        """Manually evaluate model performance on the test set."""
        model.eval()
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        print(f"📊 Starting evaluation on {len(test_dataset)} test samples...")

        with torch.no_grad():
            for i, sample in enumerate(test_dataset):
                if i % 20 == 0:
                    print(f"Evaluation progress: {i}/{len(test_dataset)}")

                # Get inputs and labels
                input_ids = sample['input_ids'].unsqueeze(0).to(device)
                attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
                labels = sample['labels'].unsqueeze(0).to(device)

                # Generate predictions
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_OUTPUT_LENGTH,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # Decode predictions and labels
                pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                labels_copy = labels.clone()
                labels_copy[labels_copy == -100] = tokenizer.pad_token_id
                label_text = tokenizer.decode(labels_copy[0], skip_special_tokens=True)

                # Compute ROUGE scores
                scores = scorer.score(label_text, pred_text)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

        # Average scores
        avg_scores = {
            'eval_rouge1': np.mean(rouge_scores['rouge1']),
            'eval_rouge2': np.mean(rouge_scores['rouge2']),
            'eval_rougeL': np.mean(rouge_scores['rougeL']),
        }

        return avg_scores

    def log_metrics(metrics, phase="eval"):
        """Append metrics to the JSON log file."""
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        # Add timestamp and phase (train/eval/test)
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metrics["phase"] = phase

        logs.append(metrics)

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    print("\n📊 Evaluating model on the test set...")
    test_results = manual_evaluate_model(model, test_dataset, processor.tokenizer, device)
    log_metrics(test_results, phase="test")

    print(f"💾 Model (weights and config) will be saved to: {SAVE_MODEL_PATH}")
    # If needed, you can uncomment the following lines to save:
    # model.save_pretrained(SAVE_MODEL_PATH)
    # processor.tokenizer.save_pretrained(SAVE_MODEL_PATH)

    print("🎯 Test set performance:")
    print(f"   ROUGE-1: {test_results['eval_rouge1']:.4f}")
    print(f"   ROUGE-2: {test_results['eval_rouge2']:.4f}")
    print(f"   ROUGE-L: {test_results['eval_rougeL']:.4f}")

    # Demonstration: generate a few examples from the test set
    print("\n🔮 Generation examples from the test set:")
    model.eval()
    for i in range(min(3, len(test_data))):
        sample = test_data[i]
        enc = processor.tokenizer(
            sample["input"],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer_global.eos_token_id,
            )
        raw = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        raw = raw.strip()
        if not raw.startswith("{"):
            raw = "{" + raw
        if not raw.endswith("}"):
            raw = raw + "}"

        try:
            result = json_to_bib(raw)
            print("Conversion succeeded!")
            print("=" * 50)
            print(result)
        except Exception as e:
            print(f"Conversion failed: {e}")

    # Load prediction data (if needed)
    if os.path.exists(TOTEL_PREDICT_TXT):
        predict_recs = PubmedToBibDataProcessor.load_prediction_texts(TOTEL_PREDICT_TXT)
        print(f"▶️ Loaded {len(predict_recs)} records for prediction")
    else:
        print(f"⚠️ Prediction TXT file not found: {TOTEL_PREDICT_TXT}")


if __name__ == "__main__":
    main()