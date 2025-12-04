#!/usr/bin/env python3
# -*- coding: utf-8 -*
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import traceback
import json
from tqdm import tqdm

# Model-related paths
MODEL_PATH = r"path to best model\log\best_scibert_class_3.pt"  # Trained model weights (.pt file)
BEST_PARAMS_PATH = r"path to best model\log\best_params_class_3_scibert.json"  # Best hyperparameter config (.json file)
SCIBERT_PATH = r"path to best scibert"  # SciBERT pretrained model path

# Data paths
INPUT_DIR = r"path to evaluated data"  # Directory containing .md files to be classified
OUTPUT_DIR = r"path to output"  # Directory to save prediction results

# Optional configuration
BATCH_SIZE = 2  # Batch size; decrease if GPU memory is insufficient
MAX_FILES_PER_BATCH = 1000  # Max number of files per batch when processing large datasets
ENABLE_GPU = True  # Whether to use GPU if available

# Set random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Hierarchical SciBERT model (same as in training)
class HierarchicalSciBERT(nn.Module):
    def __init__(self, model_path, num_labels=3, aggregator="mean"):
        super().__init__()
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_path)
        hidden = self.bert.config.hidden_size
        assert aggregator in ("mean", "max"), "Only 'mean' and 'max' are supported"
        self.aggregator = aggregator
        # Classifier
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (B, W, L)
        attention_mask: (B, W, L)
        """
        B, W, L = input_ids.size()
        device = input_ids.device
        doc_embeddings = []

        # Iterate over documents in the batch
        for i in range(B):
            # Only process windows that contain at least one token
            mask_sum = attention_mask[i].sum(dim=1)  # (W,)
            valid_windows = (mask_sum > 0).nonzero(as_tuple=True)[0]
            cls_list = []

            # Run BERT on each valid window and collect [CLS] embeddings
            for w in valid_windows:
                ids = input_ids[i, w].unsqueeze(0)        # (1, L)
                mask = attention_mask[i, w].unsqueeze(0)  # (1, L)
                outputs = self.bert(input_ids=ids, attention_mask=mask)
                cls_emb = outputs.last_hidden_state[:, 0]  # (1, hidden)
                cls_list.append(cls_emb)

            # Stack to shape (W_valid, hidden)
            cls_stack = torch.cat(cls_list, dim=0)
            # Document-level aggregation
            if self.aggregator == "mean":
                doc_emb = cls_stack.mean(dim=0)  # (hidden,)
            else:
                doc_emb, _ = cls_stack.max(dim=0)

            doc_embeddings.append(doc_emb)

        # Final batch-level document embeddings (B, hidden)
        doc_embeddings = torch.stack(doc_embeddings, dim=0)
        # Classifier output (B, num_labels)
        logits = self.classifier(doc_embeddings)
        return logits

# Sliding-window tokenization (same as in training)
def tokenize_sliding_windows(
    text: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    stride: int
):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    windows = []

    for start in range(0, len(tokens), max_length - stride):
        chunk = tokens[start:start + max_length]
        mask = [1] * len(chunk)
        # Pad to max_length
        if len(chunk) < max_length:
            pad_len = max_length - len(chunk)
            chunk = chunk + [tokenizer.pad_token_id] * pad_len
            mask = mask + [0] * pad_len
        windows.append((chunk, mask))
        if start + max_length >= len(tokens):
            break
    return windows

# Dataset for prediction
class PredictDataset(Dataset):
    def __init__(self, windows_list, file_paths):
        self.data = windows_list
        self.file_paths = file_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        w = self.data[idx]
        ids = torch.tensor([c for c, _ in w], dtype=torch.long)
        masks = torch.tensor([m for _, m in w], dtype=torch.long)
        return ids, masks, self.file_paths[idx]

# Custom collate function to pad the number of windows per document
def collate_fn(batch):
    all_ids, all_masks, all_paths = zip(*batch)
    bs = len(all_ids)
    max_w = max(x.size(0) for x in all_ids)
    seq_len = all_ids[0].size(1)
    padded_ids = torch.zeros(bs, max_w, seq_len, dtype=torch.long)
    padded_masks = torch.zeros(bs, max_w, seq_len, dtype=torch.long)

    for i, (ids, masks) in enumerate(zip(all_ids, all_masks)):
        w = ids.size(0)
        padded_ids[i, :w] = ids
        padded_masks[i, :w] = masks

    return padded_ids, padded_masks, list(all_paths)

def load_md_files_for_prediction(input_dir, tokenizer, max_length=512, stride=384, max_windows=8):
    """
    Load all .md files from the input directory for prediction.
    """
    texts = []
    file_paths = []

    if not os.path.isdir(input_dir):
        print(f"Error: input directory {input_dir} does not exist")
        return texts, file_paths

    # Collect all .md files
    md_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.md'):
                md_files.append(os.path.join(root, file))

    print(f"Found {len(md_files)} .md files")

    processed_files = 0
    for fpath in tqdm(md_files, desc="Processing files"):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            print(f"Error reading file {fpath}: {e}")
            continue

        if not content:
            print(f"File {fpath} is empty, skipped")
            continue

        # Rough length check to avoid extremely long texts
        if len(content) > max_length * 20:
            content = content[:max_length * 20]
            print(f"File {fpath} is too long, truncated")

        try:
            windows = tokenize_sliding_windows(
                content, tokenizer, max_length, stride
            )
            # Limit the number of windows to avoid memory issues
            windows = windows[:max_windows]
            if windows:
                texts.append(windows)
                file_paths.append(fpath)
                processed_files += 1
        except Exception as e:
            print(f"Error applying sliding windows to {fpath}: {e}")
            continue

    print(f"Successfully processed {processed_files} files for prediction")
    return texts, file_paths

def setup_logging(log_dir):
    """Set up logging."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"prediction_log_{current_time}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return current_time

def check_configuration():
    """Check the validity of configuration."""
    print("=" * 60)
    print("           Configuration Check")
    print("=" * 60)

    errors = []
    warnings = []

    # Check model file
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file does not exist: {MODEL_PATH}")
    else:
        print(f"✓ Model file: {MODEL_PATH}")

    # Check hyperparameter file
    if not os.path.exists(BEST_PARAMS_PATH):
        errors.append(f"Hyperparameter file does not exist: {BEST_PARAMS_PATH}")
    else:
        print(f"✓ Hyperparameter file: {BEST_PARAMS_PATH}")
        try:
            with open(BEST_PARAMS_PATH, 'r') as f:
                params = json.load(f)
            print(f"  - batch_size: {params.get('batch_size', 'N/A')}")
            print(f"  - max_length: {params.get('max_length', 'N/A')}")
            print(f"  - aggregator: {params.get('aggregator', 'N/A')}")
        except Exception as e:
            warnings.append(f"Failed to read hyperparameter file: {e}")

    # Check SciBERT path
    if not os.path.exists(SCIBERT_PATH):
        errors.append(f"SciBERT path does not exist: {SCIBERT_PATH}")
    else:
        print(f"✓ SciBERT path: {SCIBERT_PATH}")

    # Check input directory
    if not os.path.exists(INPUT_DIR):
        errors.append(f"Input directory does not exist: {INPUT_DIR}")
    else:
        # Count number of .md files
        md_count = 0
        for root, dirs, files in os.walk(INPUT_DIR):
            for file in files:
                if file.lower().endswith('.md'):
                    md_count += 1
        print(f"✓ Input directory: {INPUT_DIR} (contains {md_count} .md files)")

    # Check output directory
    if not os.path.exists(OUTPUT_DIR):
        print(f"○ Output directory does not exist, will be created automatically: {OUTPUT_DIR}")
    else:
        print(f"✓ Output directory: {OUTPUT_DIR}")

    # Check GPU
    if ENABLE_GPU and torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    elif ENABLE_GPU:
        warnings.append("GPU is not enabled or not available, CPU will be used")
    else:
        print("○ Configured to use CPU")

    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Max files per batch: {MAX_FILES_PER_BATCH}")

    # Show warnings
    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    # Show errors
    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease modify the configuration section at the top of the script to fix these issues.")
        return False

    print("\n✅ Configuration check passed!")
    return True

def predict_and_organize_files():
    """
    Main prediction function.
    """
    print("=" * 60)
    print("        SciBERT MD File Classification Tool")
    print("=" * 60)

    # Check configuration
    if not check_configuration():
        print("\n❌ Configuration check failed. Please fix the configuration and retry.")
        return

    set_seed(42)
    device = torch.device("cuda:0" if (ENABLE_GPU and torch.cuda.is_available()) else 'cpu')
    print(f"\nUsing device: {device}")

    # Set up output directory for logs
    log_dir = os.path.join(OUTPUT_DIR, "log")
    current_time = setup_logging(log_dir)

    # Label mapping
    label_to_name = {0: "PPT", 1: "FT", 2: "Review"}

    # Create category output directories
    for label_name in label_to_name.values():
        output_dir = os.path.join(OUTPUT_DIR, label_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

    try:
        # Load best parameters
        with open(BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)

        logging.info("Loaded best parameters:")
        for key, value in best_params.items():
            logging.info(f"  {key}: {value}")

        # Extract configuration from parameters
        max_length = best_params.get("max_length", 512)
        stride = best_params.get("stride", 384)
        max_windows = best_params.get("max_windows", 8)
        aggregator = best_params.get("aggregator", "mean")
        # Use BATCH_SIZE from the top configuration; if not set, fall back to param file
        batch_size = BATCH_SIZE if BATCH_SIZE else best_params.get("batch_size", 4)

        print(f"\nPrediction parameters:")
        print(f"  - max_length: {max_length}")
        print(f"  - stride: {stride}")
        print(f"  - max_windows: {max_windows}")
        print(f"  - aggregator: {aggregator}")
        print(f"  - batch_size: {batch_size}")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(SCIBERT_PATH)
        logging.info(f"Tokenizer loaded: {SCIBERT_PATH}")

        # Load data
        print(f"\nStart loading files to be predicted from {INPUT_DIR} ...")
        texts, file_paths = load_md_files_for_prediction(
            INPUT_DIR, tokenizer, max_length, stride, max_windows
        )

        if len(texts) == 0:
            logging.error("No files found to process")
            print("❌ No files found to process")
            return

        print(f"✅ Loaded {len(texts)} files for prediction")
        logging.info(f"Loaded {len(texts)} files for prediction")

        # Split into batches for large datasets
        total_files = len(texts)
        batches = []
        for i in range(0, total_files, MAX_FILES_PER_BATCH):
            end_idx = min(i + MAX_FILES_PER_BATCH, total_files)
            batches.append((texts[i:end_idx], file_paths[i:end_idx]))

        print(f"Files will be processed in {len(batches)} batch(es)")

        # Load model
        model = HierarchicalSciBERT(
            SCIBERT_PATH,
            num_labels=3,
            aggregator=aggregator
        ).to(device)

        # Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        logging.info(f"Model loaded: {MODEL_PATH}")
        print("✅ Model loaded successfully")

        # Process all batches
        all_predictions = []
        all_predicted_files = []
        all_confidences = []

        for batch_idx, (batch_texts, batch_file_paths) in enumerate(batches):
            print(f"\nProcessing batch {batch_idx + 1}/{len(batches)} ({len(batch_texts)} files)...")

            # Create dataset and dataloader
            predict_dataset = PredictDataset(batch_texts, batch_file_paths)
            predict_dataloader = DataLoader(
                predict_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

            # Run prediction
            batch_predictions = []
            batch_predicted_files = []
            batch_confidences = []

            with torch.no_grad():
                for input_ids, attention_mask, batch_file_paths_inner in tqdm(predict_dataloader, desc="Predicting"):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                    outputs = model(input_ids, attention_mask)
                    probs = torch.softmax(outputs, dim=1)
                    confidence_scores, predicted_labels = torch.max(probs, 1)

                    batch_predictions.extend(predicted_labels.cpu().numpy())
                    batch_predicted_files.extend(batch_file_paths_inner)
                    batch_confidences.extend(confidence_scores.cpu().numpy())

            # Append batch results to global lists
            all_predictions.extend(batch_predictions)
            all_predicted_files.extend(batch_predicted_files)
            all_confidences.extend(batch_confidences)

        # Count prediction results
        prediction_counts = {label_name: 0 for label_name in label_to_name.values()}

        # Copy files into corresponding directories and record results
        results = []

        print(f"\nOrganizing files into output directories...")
        for i, (file_path, pred_label, confidence) in enumerate(
            tqdm(zip(all_predicted_files, all_predictions, all_confidences), desc="Organizing files")
        ):
            pred_name = label_to_name[pred_label]
            prediction_counts[pred_name] += 1

            # Build target file path
            filename = os.path.basename(file_path)
            target_dir = os.path.join(OUTPUT_DIR, pred_name)
            target_path = os.path.join(target_dir, filename)

            # Handle file name conflicts
            counter = 1
            original_target_path = target_path
            while os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                counter += 1

            try:
                # Copy file
                shutil.copy2(file_path, target_path)

                # Record result
                results.append({
                    'original_path': file_path,
                    'target_path': target_path,
                    'predicted_label': pred_name,
                    'confidence': confidence,
                    'success': True
                })

                # Log confidence for all files
                logging.info(
                    f"File {filename} predicted as {pred_name} (confidence: {confidence:.4f}) -> {target_path}"
                )

            except Exception as e:
                logging.error(f"Failed to copy file {file_path} to {target_path}: {e}")
                results.append({
                    'original_path': file_path,
                    'target_path': target_path,
                    'predicted_label': pred_name,
                    'confidence': confidence,
                    'success': False,
                    'error': str(e)
                })

        # Log statistics
        print("\n" + "=" * 60)
        print("           Prediction Statistics")
        print("=" * 60)

        logging.info("\n" + "=" * 50)
        logging.info("Prediction statistics:")
        total_files = len(all_predicted_files)
        for label_name, count in prediction_counts.items():
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            logging.info(f"{label_name}: {count} files ({percentage:.2f}%)")
            print(f"{label_name}: {count} files ({percentage:.2f}%)")

        logging.info(f"Total processed: {total_files} files")
        print(f"Total processed: {total_files} files")

        # Compute average confidence
        avg_confidence = np.mean(all_confidences)
        logging.info(f"Average prediction confidence: {avg_confidence:.4f}")
        print(f"Average prediction confidence: {avg_confidence:.4f}")

        # Save detailed results to CSV
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(log_dir, f"prediction_results_{current_time}.csv")
        results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Detailed results saved to: {results_csv_path}")

        # Save summary statistics
        summary_data = {
            'total_files': total_files,
            'prediction_time': current_time,
            'model_path': MODEL_PATH,
            'input_directory': INPUT_DIR,
            'output_directory': OUTPUT_DIR,
            'predictions': prediction_counts,
            'average_confidence': float(avg_confidence),
            'parameters_used': best_params,
            'configuration': {
                'batch_size': batch_size,
                'max_files_per_batch': MAX_FILES_PER_BATCH,
                'enable_gpu': ENABLE_GPU,
                'device_used': str(device)
            }
        }

        summary_path = os.path.join(log_dir, f"prediction_summary_{current_time}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Summary statistics saved to: {summary_path}")

        print("\n" + "=" * 60)
        print("           Prediction Completed!")
        print("=" * 60)
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print(f"📄 Detailed logs: {log_dir}")
        print(f"📊 Per-class file counts are recorded in the log files")

    except Exception as e:
        logging.error(f"Error occurred during prediction: {e}")
        logging.error(traceback.format_exc())
        print(f"❌ Prediction failed: {e}")
        print("Please check the log files for detailed error information")
        raise

if __name__ == "__main__":
    print("SciBERT MD File Classification Tool")
    print("Please ensure that the paths in the configuration section at the top of this script are correctly set.")
    print()

    # Simple confirmation prompt
    print("Current configuration:")
    print(f"  Model file: {MODEL_PATH}")
    print(f"  Hyperparameter file: {BEST_PARAMS_PATH}")
    print(f"  Input directory: {INPUT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()

    try:
        input("Press Enter to start prediction, or Ctrl+C to cancel...")
        predict_and_organize_files()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nProgram encountered an error: {e}")
        input("Press Enter to exit...")