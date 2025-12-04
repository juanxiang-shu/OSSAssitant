# -*- coding: utf-8 -*-
import os
import json
import time
import random
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import anthropic
import requests

# ========== CONFIGURABLE SETTINGS ==========
CONFIG = {
    # Data & output
    "TXT_PATH": "path your TXT",
    "SAVE_PREDICTIONS": "path your json",

    "ANTHROPIC_API_KEY": "sk-xxx",  # your api key
    "ANTHROPIC_MODEL": "claude-sonnet-4-20250514",  # "claude-3-5-haiku-20241022",

    # Generation parameters
    "TIMEOUT": 600,
    "TEMPERATURE": 0.1,
    "MAX_TOKENS": 64000,
    "SLEEP_MIN": 10,   # Minimum wait time (seconds)
    "SLEEP_MAX": 20,   # Maximum wait time (seconds)

    "START_IDX": 0,  # Start processing from which sample, 0 means from the first one
}

# ========== JSON TEMPLATE ==========
JSON_TEMPLATE = r"template.json"


def load_json_template(template_path: str) -> str:
    """
    Load a JSON template file that may contain // comments.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"JSON template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    return content


# Load template content
json_template_content = load_json_template(JSON_TEMPLATE)

# ========== SYSTEM & USER PROMPTS ==========
system_prompt = """ 
Role: You are an expert in on-surface synthesis and scientific text analysis. You can spend a lot of time thinking and completing the structure and content.
Extract structured information from the provided scientific text into the given JSON template. Follow these rules strictly:
1. Output **only** a valid JSON object — no explanations, no markdown formatting, no extra text.
2. Use the provided JSON template as-is. Do not add, delete, or rename keys.
3. For each field, fill values exactly as they appear in the article. Do not modify, summarize, or embellish.
4. If information is missing, set the field to null.
5. Wrap the JSON in triple backticks and add JSON tags on the first and last lines:
    ```json
    {{...}}
    ```
JSON template:
{json_template_content}
"""

user_prompt = """
Scientific text:\n
{text}

Carefully read the text, then fill the JSON template strictly according to the rules above. 
"""

# ========== DATA LOADING & SAMPLING ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_txt_from_folders(annotated_root: str) -> List[Dict[str, Any]]:
    txt_dir = os.path.join(annotated_root)
    processed_data = []
    if not os.path.exists(txt_dir):
        print(f"❌ txt directory not found: {txt_dir}")
        return processed_data

    for filename in os.listdir(txt_dir):
        if not filename.endswith(".txt"):
            continue
        file_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(txt_dir, filename)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        except Exception as e:
            print(f"⚠️ Failed to read text file {txt_path}: {e}")
            continue

        processed_data.append(
            {
                "file_name": file_name,
                "text": text_content,
            }
        )

    print(f"Loaded {len(processed_data)} txt samples from {txt_dir}")
    return processed_data


# ========== JSON OUTPUT VALIDATION & CLEANING ==========
def validate_json_output(json_str: str) -> Tuple[bool, Dict]:
    """
    Try to parse a JSON string, with fallback cleaning for common issues:
    - Stripping markdown ```json ... ``` blocks
    - Adding missing leading/ending braces
    """
    try:
        parsed = json.loads(json_str)
        return True, parsed
    except json.JSONDecodeError:
        cleaned = json_str.strip()
        if not cleaned:
            return False, {}

        # Handle markdown code block format: remove ```json ... ```
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = cleaned[7:]  # remove leading ```json
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]  # remove trailing ```
            cleaned = cleaned.strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            # Handle generic ``` ... ``` wrapper
            cleaned = cleaned[3:-3].strip()

        if not cleaned:
            return False, {}
        if not cleaned.startswith("{"):
            cleaned = "{" + cleaned
        if not cleaned.endswith("}"):
            cleaned = cleaned + "}"
        try:
            parsed = json.loads(cleaned)
            return True, parsed
        except Exception:
            return False, {}


# ========== ERROR LOGGING UTILITIES ==========
def init_error_log(save_dir: str) -> str:
    """
    Ensure that the output directory exists and return the full path to error.log.
    """
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, "error.log")


def log_error(
    error_log_path: str,
    file_name: str,
    stage: str,
    message: str,
    raw: Optional[str] = None,
    raw_max_len: int = 2000,
):
    """
    Append an error record to the error log.
    stage: "API_CALL" | "JSON_PARSE"
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{stage}] {file_name} | {message}\n")
            if raw:
                raw_snippet = raw[:raw_max_len]
                f.write(f"RAW: {raw_snippet}\n")
            f.write("-" * 80 + "\n")
    except Exception as e:
        # If logging fails, at least show a console warning
        print(f"⚠️ Failed to write error log: {e}")


# ========== CLAUDE / ANTHROPIC API CALL ==========
def call_anthropic_api_conversation(
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str],
    model: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Call the Anthropic API for a single system + user message.
    """
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise RuntimeError("A valid ANTHROPIC_API_KEY is not set. Please fill it in CONFIG.")

    start_time = time.time()

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )
        elapsed_time_1 = time.time() - start_time
        print(f"✅ API call completed, time used: {elapsed_time_1:.1f} seconds")

        response = message.content[0].text
        return response.strip()

    except requests.exceptions.Timeout:
        print(f"⏰ API call timed out (>{timeout} seconds)")
        raise RuntimeError(f"API call timeout: inference time exceeded {timeout} seconds")
    except Exception as e:
        print(f"Detailed API error: {e}")
        raise


# ========== MAIN PIPELINE ==========
def main():
    # If needed, you can enable deterministic sampling
    # set_seed(CONFIG["SEED"])

    data = load_txt_from_folders(CONFIG["TXT_PATH"])
    if not data:
        print("❌ No valid annotated data found")
        return

    sampled = sorted(data, key=lambda x: x["file_name"])
    n = len(sampled)
    print(f"📦 Using all {n} samples for extraction")

    # Output directory & error log
    os.makedirs(CONFIG["SAVE_PREDICTIONS"], exist_ok=True)
    error_log_path = init_error_log(CONFIG["SAVE_PREDICTIONS"])

    start_idx = CONFIG.get("START_IDX", 0)
    if start_idx >= n:
        print(f"⚠️ START_IDX={start_idx} exceeds total sample count {n}, nothing to process")
        return

    for idx, sample in enumerate(sampled[start_idx:], start=start_idx + 1):
        file_name = sample["file_name"]
        print(f"\n🔄 Processing sample {idx}/{n}: {file_name}")
        try:
            formatted_system_prompt = system_prompt.format(json_template_content=json_template_content)
            formatted_user_prompt = user_prompt.format(text=sample["text"])
            print(f"File name: {file_name}")
            try:
                response = call_anthropic_api_conversation(
                    system_prompt=formatted_system_prompt,
                    user_prompt=formatted_user_prompt,
                    api_key=CONFIG["ANTHROPIC_API_KEY"],
                    model=CONFIG["ANTHROPIC_MODEL"],
                    timeout=CONFIG["TIMEOUT"],
                    temperature=CONFIG["TEMPERATURE"],
                    max_tokens=CONFIG["MAX_TOKENS"],
                )
            except requests.exceptions.Timeout:
                msg = f"API call timed out (>{CONFIG['TIMEOUT']} seconds)"
                print(f"⏰ {msg}")
                log_error(error_log_path, file_name, "API_CALL", msg)
                response = "API call failed"
            except Exception as e:
                msg = f"Detailed API error: {e}"
                print(msg)
                log_error(error_log_path, file_name, "API_CALL", msg)
                response = "API call failed"

            wait_time = random.uniform(CONFIG["SLEEP_MIN"], CONFIG["SLEEP_MAX"])
            print(f"⏳ Waiting {wait_time:.1f} seconds before the next request")
            time.sleep(wait_time)

        except Exception as e:
            # Fallback for any non-API exceptions in the main loop
            msg = f"Unexpected error in main loop: {e}"
            print(f"❌ {msg}")
            log_error(error_log_path, file_name, "API_CALL", msg)
            response = "API call failed"

        is_valid, pred_data = validate_json_output(response)
        if not is_valid:
            print(f"❌ JSON parsing failed for [{file_name}]")
            # Log raw response (truncated)
            log_error(
                error_log_path,
                file_name,
                "JSON_PARSE",
                "Response could not be parsed as valid JSON",
                raw=response,
            )
            continue

        # Save prediction as an individual JSON file, containing only the extracted content
        json_path = os.path.join(CONFIG["SAVE_PREDICTIONS"], f"{file_name}.json")
        with open(json_path, "w", encoding="utf-8") as fout:
            json.dump(pred_data, fout, ensure_ascii=False, indent=2)
        print(f"✅ Prediction saved to: {json_path}")


if __name__ == "__main__":
    main()