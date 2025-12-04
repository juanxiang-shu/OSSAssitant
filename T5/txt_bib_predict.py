# -*- coding: utf-8 -*-
import os
import json
import re
import time
from typing import List, Dict
import torch
from transformers import T5TokenizerFast as T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# ========== Configuration ==========
BEST_TRIAL_NUMBER_FILE = "/data/best_trial_number.txt"
RESULTS_DIR = "/data/results"
T5_MODEL_PATH = "/data/models/t5"

INPUT_TXT_PATH = "/data/totel.txt"
OUTPUT_BIB_PATH = "/data/totel.bib"
FAILED_TXT_PATH = "/data/failed.txt"
PROGRESS_FILE = "/data/progress.txt"

MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 1024
BATCH_SIZE = 10
NUM_BEAMS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_TEMPLATE = "Convert the following PubMed article to BibTeX format: {text}"

# ---------- Helper Functions ----------


def json_to_bib(json_str):
    """
    More robust converter: given a raw JSON string, try to parse it
    (with a fallback to manual_parse_json) and generate a BibTeX entry.
    """
    # Clean trailing quotes at the end
    json_str = re.sub(r'"+\s*$', '', json_str.strip())
    if not json_str.strip().endswith('}'):
        json_str += '}'
    json_str = json_str.replace('\\"', '"')

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = manual_parse_json(json_str)

    # Completing required fields and ID generation are handled by the caller
    return dict_to_bib(data)


def dict_to_bib(data: Dict) -> str:
    """
    Take a parsed dict (parsed_data) and generate a BibTeX entry string.
    Tolerates missing ENTRYTYPE/ID, but the caller should ensure they are filled.
    """
    entry_type = data.get('ENTRYTYPE', 'article')
    entry_id = data.get('ID') or data.get('id')  # Sometimes ID might be lowercase
    if not entry_id:
        raise ValueError("An ID field is required to generate a BibTeX entry")

    bib_lines = [f"@{entry_type}{{{entry_id},"]

    field_order = ['title', 'author', 'journal', 'year', 'volume', 'number', 'pages', 'doi', 'abstract']
    all_fields = set(field_order + [k for k in data.keys() if k not in ['ENTRYTYPE', 'ID']])
    fields_to_include = sorted(all_fields, key=lambda x: field_order.index(x) if x in field_order else len(field_order))

    for field in fields_to_include:
        if field in data and str(data[field]).strip():
            value = str(data[field])
            # Escape BibTeX special characters
            value = value.replace('{', '{{').replace('}', '}}').replace('&', '\\&')
            bib_lines.append(f"  {field} = {{{value}}},")

    if len(bib_lines) > 1:
        bib_lines[-1] = bib_lines[-1].rstrip(',')  # Remove trailing comma from the last field
    bib_lines.append('}')
    return '\n'.join(bib_lines)


def manual_parse_json(json_str):
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


def split_articles_by_blank(content: str) -> List[str]:
    return [rec.strip() for rec in re.split(r'\n{3,}', content) if rec.strip()]


def extract_doi_from_txt(record: str) -> str:
    m = re.search(r'doi[:\s]*\r?\n*\s*(10\.\d{4,9}/\S+)', record, flags=re.IGNORECASE)
    if not m:
        return None
    doi = m.group(1).rstrip('.,;')
    return doi.lower()


def generate_unique_id(text: str, index: int) -> str:
    doi = extract_doi_from_txt(text)
    if doi:
        doi_parts = doi.split('/')
        if len(doi_parts) >= 2:
            return f"{doi_parts[0].replace('.', '_')}_{doi_parts[1].split('.')[0]}"
    return f"article_{index + 1}"


def load_txt_records(path: str) -> List[str]:
    print(f"📖 Loading TXT file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        records = split_articles_by_blank(f.read())
    print(f"✅ Successfully loaded {len(records)} records")
    return records


def load_best_model_and_tokenizer():
    print("🔍 Searching for the best trained model...")
    best_trial_number = None
    if os.path.exists(BEST_TRIAL_NUMBER_FILE):
        with open(BEST_TRIAL_NUMBER_FILE, 'r') as f:
            best_trial_number = int(f.read().strip())
        print(f"📋 Found best trial number: {best_trial_number}")
    else:
        raise FileNotFoundError(f"Best trial number file not found: {BEST_TRIAL_NUMBER_FILE}")

    if best_trial_number is not None:
        trial_dir = f"{RESULTS_DIR}/trial_{best_trial_number}"
        if os.path.exists(trial_dir):
            checkpoints = [d for d in os.listdir(trial_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                best_model_path = os.path.join(trial_dir, latest_checkpoint)
                print(f"🔄 Loading model from best trial checkpoint: {best_model_path}")
                model = T5ForConditionalGeneration.from_pretrained(best_model_path)
                tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH, use_fast=True)
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id
                model.to(DEVICE)
                model.eval()
                print(f"✅ Model loaded successfully, using device: {DEVICE}")
                return model, tokenizer
            else:
                raise FileNotFoundError("❌ No checkpoint found for the best trial")
        else:
            raise FileNotFoundError("❌ Best trial directory not found")
    else:
        raise ValueError("❌ Best trial number not found")


def predict_batch(model, tokenizer, texts: List[str]) -> List[str]:
    inputs = [INPUT_TEMPLATE.format(text=text) for text in texts]
    encodings = tokenizer(
        inputs,
        truncation=True,
        padding=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    predictions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return predictions


def postprocess_json(raw_json: str) -> str:
    raw_json = raw_json.strip()
    if not raw_json.startswith("{"):
        raw_json = "{" + raw_json
    if not raw_json.endswith("}"):
        raw_json = raw_json + "}"
    return raw_json


def get_processed_count() -> int:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return int(f.read().strip())
        except Exception:
            return 0
    return 0


def update_progress(count: int):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        f.write(str(count))
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def append_bib_entry(entry: str):
    with open(OUTPUT_BIB_PATH, 'a', encoding='utf-8') as f:
        f.write(entry)
        f.write("\n\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def append_failed_record(header: str, details: List[str]):
    with open(FAILED_TXT_PATH, 'a', encoding='utf-8') as f:
        f.write(header + "\n")
        for line in details:
            f.write(line + "\n")
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


# ---------- Main Pipeline ----------
def main():
    print("🚀 Starting T5 model inference")
    print("=" * 60)

    if not os.path.exists(INPUT_TXT_PATH):
        print(f"❌ Input file does not exist: {INPUT_TXT_PATH}")
        return

    # Ensure output paths exist
    os.makedirs(os.path.dirname(OUTPUT_BIB_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(FAILED_TXT_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(PROGRESS_FILE) or ".", exist_ok=True)

    # Load model
    try:
        model, tokenizer = load_best_model_and_tokenizer()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load input records
    try:
        txt_records = load_txt_records(INPUT_TXT_PATH)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    total = len(txt_records)
    if total == 0:
        print("❌ No valid text records found")
        return

    # Resume from checkpoint
    start_idx = get_processed_count()
    if start_idx >= total:
        print(f"✅ All {total} records already processed (progress.txt shows {start_idx}), nothing to do.")
        return
    if start_idx > 0:
        print(f"🔁 Resuming from record {start_idx + 1} (skipping first {start_idx} records)")

    print(f"🔮 Total {total} records, starting inference from {start_idx + 1} ...")
    print(f"📊 Inference configuration: batch size {BATCH_SIZE}, beam size {NUM_BEAMS}, device {DEVICE}")

    successful_predictions = 0
    failed_predictions = 0

    with tqdm(total=total, initial=start_idx, desc="Prediction progress") as pbar:
        for i in range(start_idx, total, BATCH_SIZE):
            batch_texts = txt_records[i:i + BATCH_SIZE]
            batch_indices = list(range(i, min(i + BATCH_SIZE, total)))

            try:
                batch_preds = predict_batch(model, tokenizer, batch_texts)
            except Exception as e:
                print(f"❌ Batch {i // BATCH_SIZE + 1} prediction failed: {e}")
                for idx_in_batch, text in zip(batch_indices, batch_texts):
                    header = f"=== Failed record (original index: {idx_in_batch + 1}) ==="
                    details = [
                        f"Error message: batch prediction failed: {e}",
                        f"DOI: {extract_doi_from_txt(text) or 'N/A'}",
                        f"Original text snippet: {text[:500]}...",
                    ]
                    append_failed_record(header, details)
                    failed_predictions += 1
                    update_progress(idx_in_batch + 1)
                    pbar.update(1)
                continue

            for raw_pred, text, idx in zip(batch_preds, batch_texts, batch_indices):
                try:
                    processed_json = postprocess_json(raw_pred)
                    # First attempt: strict JSON parsing
                    try:
                        parsed_data = json.loads(processed_json)
                    except json.JSONDecodeError:
                        parsed_data = manual_parse_json(processed_json)

                    # Fill missing ENTRYTYPE and ID if necessary
                    if 'ENTRYTYPE' not in parsed_data or not parsed_data.get('ENTRYTYPE'):
                        parsed_data['ENTRYTYPE'] = 'article'
                    if 'ID' not in parsed_data or not parsed_data.get('ID'):
                        parsed_data['ID'] = generate_unique_id(text, idx)

                    # Generate BibTeX
                    try:
                        bib_entry = dict_to_bib(parsed_data)
                    except Exception:
                        # Fallback: use string-based json_to_bib for robustness
                        try:
                            bib_entry = json_to_bib(processed_json)
                        except Exception as e2:
                            raise ValueError(f"BibTeX generation failed (both attempts failed): {e2}")

                    append_bib_entry(bib_entry)
                    successful_predictions += 1

                except Exception as e:
                    header = f"=== Failed record (original index: {idx + 1}) ==="
                    details = [
                        f"Error message: {e}",
                        f"DOI: {extract_doi_from_txt(text) or 'N/A'}",
                        f"Raw prediction output: {raw_pred[:200]}...",
                        f"Original text snippet: {text[:500]}...",
                    ]
                    append_failed_record(header, details)
                    failed_predictions += 1

                # Update checkpoint after each record
                update_progress(idx + 1)
                pbar.update(1)

    # Summary
    processed = get_processed_count()
    print("\n🎉 Inference completed!")
    print("=" * 60)
    print("📊 Summary for this run:")
    print(f"   Total records: {total}")
    print(f"   Processed records (including previous runs): {processed}")
    print(f"   Newly successful predictions: {successful_predictions}")
    print(f"   Newly failed predictions: {failed_predictions}")
    print("💾 Output locations:")
    print(f"   BibTeX file: {OUTPUT_BIB_PATH}")
    print(f"   Failed record log: {FAILED_TXT_PATH}")
    if processed < total:
        print(f"➡️ Next run will start from record {processed + 1}.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n✨ All done!")


if __name__ == "__main__":
    main()