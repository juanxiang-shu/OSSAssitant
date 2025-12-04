# -*- coding: utf-8 -*-
import os
import json
import time
import random
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import anthropic
import re

# ========== Config (modifiable section) ==========
CONFIG = {
    # Data & output
    "TXT_PATH": "path your TXT",
    "JSON_PATH": "path your JSON",
    "SAVE_QA_DATA": "path your QA_output",

    "ANTHROPIC_API_KEY": "YOUR_ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL": "claude-sonnet-4-20250514",
    
    # Generation params
    "TIMEOUT": 600,
    "TEMPERATURE": 0.7,
    "MAX_TOKENS": 64000,
    "SLEEP_MIN": 3,
    "SLEEP_MAX": 5,

    "START_IDX": 0,  # Index to start from 0 means from the first sample
    
    # ========== New: batch mode switch ==========
    "USE_BATCH_API": True,  # True = use batch API (cheaper), False = use real-time API
    "BATCH_REQUESTS_FILE": "path your batch_requests.jsonl",
}

# ========== Prompt templates ==========
system_prompt = """ 
You are a research assistant tasked with generating a high-quality set of question-answer pairs based on a given literature text and its corresponding JSON extraction file. Please strictly adhere to the following requirements.
Question Formulation: Independent and Complete: Each question must include necessary background information (experimental subject, stage, conditions) and should not rely on context from other questions.
Avoid Metadata: Do not include content unrelated to research findings, such as DOI, title, or authors.
Avoid Ambiguous References: Do not use vague references like "this paper," "this study," or "this."
Preserve Synthetic and Retrosynthetic Questions: For example, "What kind of reaction does precursor XXX undergo on surface XXX? What are the products?" or "How can product XXX be synthesized on surface XXX?"
Generalization: Questions should focus on the system itself (molecules using IUPAC names, reactions, surfaces, etc.) rather than using "this paper" as the subject.
Content Design
Must Cover Key Elements of the Article: Precursor characteristics, deposition conditions, initial adsorption behavior, intermediate structures, final covalent products, comparisons between different surfaces, and mechanistic insights.
Each question should be independently meaningful and reflect the core content of the article on its own.
Answer Formulation
Layered Explanation: Answers should be logically structured: state facts → explain reasons → provide inferences or implications.
Avoid Brief Answers: Answers must not be overly concise; they should include detailed explanations and reasoning chains.
Reasoning Field: Between the question and answer, a reasoning field must be included to demonstrate the reasoning or thought process. This may incorporate the model's own knowledge of chemistry/physics and is not limited to the content of the literature.
Natural Expression: Answers do not need to follow a rigid "fact/reason/significance" format but should be expressed in natural paragraphs.
Format Requirements
Output in JSONL format (one object per line), with the following fields:
{
"question": "...",
"reasoning": "...",
"answer": "..."
}
Preferred format: JSONL. Output each QA object on a separate line. If you return multiple items, DO NOT use a JSON array; just output multiple lines.
Quantity and Quality
Generate 5–10 question-answer pairs each time.
Each question-answer pair must be independently meaningful and able to summarize the core information of the article when read in isolation.
Length Requirement: Each question-answer pair (including question + reasoning + answer) must contain no fewer than 4096 tokens.
"""

user_prompt = """
Scientific text:\n
{text}\n
JSON extraction template:
{json_template_content}
"""

# ========== Random seed ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

# ========== Data loading ==========
def load_txt_map(txt_dir: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not os.path.exists(txt_dir):
        print(f"❌ TXT directory not found: {txt_dir}")
        return m
    for fn in os.listdir(txt_dir):
        if not fn.lower().endswith(".txt"):
            continue
        name = os.path.splitext(fn)[0]
        path = os.path.join(txt_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                m[name] = f.read()
        except Exception as e:
            print(f"⚠️ Failed to read TXT {path}: {e}")
    return m

def load_json_map(json_dir: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not os.path.exists(json_dir):
        print(f"❌ JSON directory not found: {json_dir}")
        return m
    for fn in os.listdir(json_dir):
        if not fn.lower().endswith(".json"):
            continue
        name = os.path.splitext(fn)[0]
        path = os.path.join(json_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                m[name] = f.read()
        except Exception as e:
            print(f"⚠️ Failed to read JSON {path}: {e}")
    return m

# ========== JSONL parsing / validation ==========
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.startswith("```json") and s.endswith("```"):
        return s[7:-3].strip()
    if s.startswith("```") and s.endswith("```"):
        return s[3:-3].strip()
    return s

def _extract_top_level_arrays(s: str) -> list:
    arrays = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
            elif ch == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ']':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        arrays.append(s[start:i+1])
                        start = None
    return arrays

def _extract_top_level_objects(s: str) -> list:
    """Extract all top-level JSON object blocks ({...}), ignoring outer noise/newlines/text."""
    objs = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objs.append(s[start:i+1])
                        start = None
    return objs

def _collect_dicts(obj) -> list:
    """Recursively collect dict objects; also supports re-parsing array elements that are JSON strings."""
    out = []
    if isinstance(obj, dict):
        out.append(obj)
    elif isinstance(obj, str):
        t = obj.strip()
        try:
            inner = json.loads(t)
            out.extend(_collect_dicts(inner))
        except Exception:
            pass
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_collect_dicts(it))
    return out

def validate_jsonl_output(resp: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Parsing strategy priority:
      1) Extract all top-level array blocks `[...]` and parse them.
      2) Extract all top-level object blocks `{...}` and parse (for pretty-printed non-JSONL formats).
      3) Fallback: parse the entire string as JSON.
      4) Fallback: parse line by line as JSONL/NDJSON.
    """
    s = _strip_code_fences(resp)

    # 1) Top-level arrays
    objs_all: List[Dict[str, Any]] = []
    for ch in _extract_top_level_arrays(s):
        try:
            data = json.loads(ch)
            objs_all.extend(_collect_dicts(data))
        except Exception:
            continue
    if objs_all:
        return True, objs_all

    # 2) Top-level objects (important supplement for multi-line pretty-printed objects)
    for ch in _extract_top_level_objects(s):
        try:
            data = json.loads(ch)
            objs_all.extend(_collect_dicts(data))
        except Exception:
            continue
    if objs_all:
        return True, objs_all

    # 3) Whole-string JSON
    try:
        whole = json.loads(s)
        objs_all = _collect_dicts(whole)
        if objs_all:
            return True, objs_all
    except Exception:
        pass

    # 4) Line-by-line JSONL (last resort)
    objs_all = []
    for ln in (ln for ln in s.splitlines() if ln.strip()):
        try:
            item = json.loads(ln)
            objs_all.extend(_collect_dicts(item))
        except Exception:
            continue
    return (len(objs_all) > 0), objs_all

# ========== QA object normalization ==========
def normalize_qa_object(obj: Dict[str, Any]) -> Dict[str, str]:
    """Keep only question / reasoning / answer fields; fill missing fields with empty strings."""
    return {
        "question": str(obj.get("question", "") or ""),
        "reasoning": str(obj.get("reasoning", "") or ""),
        "answer": str(obj.get("answer", "") or "")
    }

# ========== Error logging ==========
def init_error_log(save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    p = os.path.join(save_dir, "error.log")
    # Separate each run
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"\n=== RUN START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return p

def log_error(
    error_log_path: str,
    file_name: str,
    stage: str,
    message: str,
    raw: Optional[str] = None,
    raw_max_len: int = 2000
):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{stage}] {file_name} | {message}\n")
            if raw:
                f.write(f"RAW: {raw[:raw_max_len]}\n")
            f.write("-" * 80 + "\n")
    except Exception as e:
        print(f"⚠️ Failed to write error log: {e}")

# ========== Real-time Anthropic API call ==========
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
    Perform a single Anthropic messages.create API call (real-time mode).
    """
    if not api_key or api_key == "YOUR_API_KEY_HERE" or api_key == "YOUR_ANTHROPIC_API_KEY":
        raise RuntimeError("No valid ANTHROPIC_API_KEY set; please set it in CONFIG.")

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
                    "content": user_prompt
                }
            ]
        )
        elapsed_time_1 = time.time() - start_time
        print(f"✅ Call completed, elapsed: {elapsed_time_1:.1f}s")

        response = message.content[0].text
        return response.strip()
        
    except Exception as e:
        print(f"Detailed API error: {e}")
        raise

# ========== Batch mode helpers ==========
def sanitize_custom_id(name: str, idx: int, start_idx: int) -> str:
    """
    Generate a valid custom_id for Anthropic API:
    - Allowed chars: a-z, A-Z, 0-9, _, -
    - Length: 1–64 characters
    """
    # Keep only allowed characters
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Prefix with index
    prefix = f"req_{start_idx + idx:06d}"
    
    # Remaining allowed length
    remaining_len = 64 - len(prefix) - 1  # -1 for underscore
    
    if remaining_len > 0 and clean_name:
        custom_id = f"{prefix}_{clean_name[:remaining_len]}"
    else:
        custom_id = prefix
    
    return custom_id

def create_batch_requests_file(common_names: List[str], txt_map: Dict, json_map: Dict, start_idx: int):
    """Create a JSONL file with batch requests for Anthropic batch API."""
    requests_file = CONFIG["BATCH_REQUESTS_FILE"]
    os.makedirs(os.path.dirname(requests_file), exist_ok=True)
    
    # Mapping file to track custom_id -> original file name
    mapping_file = requests_file.replace(".jsonl", "_mapping.json")
    name_mapping = {}
    
    with open(requests_file, "w", encoding="utf-8") as f:
        for idx, name in enumerate(common_names[start_idx:]):
            txt = txt_map[name]
            json_content = json_map[name]
            
            formatted_user_prompt = user_prompt.format(
                text=txt,
                json_template_content=json_content
            )
            
            # Generate API-compliant custom_id
            custom_id = sanitize_custom_id(name, idx, start_idx)
            
            # Store mapping
            name_mapping[custom_id] = name
            
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": CONFIG["ANTHROPIC_MODEL"],
                    "max_tokens": CONFIG["MAX_TOKENS"],
                    "temperature": CONFIG["TEMPERATURE"],
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": formatted_user_prompt
                        }
                    ]
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    
    # Save mapping file
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(name_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Batch request file created: {requests_file}")
    print(f"✅ Mapping file created: {mapping_file}")
    print(f"📦 Contains {len(common_names[start_idx:])} requests (starting from index {start_idx})")
    return requests_file

def submit_batch_job(requests_file: str) -> str:
    """Submit a batch job to Anthropic."""
    client = anthropic.Anthropic(api_key=CONFIG["ANTHROPIC_API_KEY"])
    
    print("📤 Reading and submitting batch requests...")
    requests_list = []
    with open(requests_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                requests_list.append(json.loads(line))
    
    print(f"   Total requests: {len(requests_list)}")
    batch = client.beta.messages.batches.create(requests=requests_list)
    
    print(f"✅ Batch job created!")
    print(f"   Batch ID: {batch.id}")
    print(f"   Status: {batch.processing_status}")
    print(f"\n💡 Batch runs asynchronously and may take some time to complete.")
    print(f"   Cost: approx. 50% cheaper than real-time API.")
    print(f"\n   Check status: python {__file__} check_batch {batch.id}")
    print(f"   Get results: python {__file__} get_results {batch.id}")
    
    return batch.id

def check_batch_status(batch_id: str):
    """Check the status of a batch job."""
    client = anthropic.Anthropic(api_key=CONFIG["ANTHROPIC_API_KEY"])
    batch = client.beta.messages.batches.retrieve(batch_id)
    
    print(f"\n📊 Batch status:")
    print(f"   Batch ID: {batch.id}")
    print(f"   Status: {batch.processing_status}")
    total_reqs = (
        batch.request_counts.processing
        + batch.request_counts.succeeded
        + batch.request_counts.errored
    )
    print(f"   Total requests: {total_reqs}")
    print(f"   Processing: {batch.request_counts.processing}")
    print(f"   Succeeded: {batch.request_counts.succeeded}")
    print(f"   Errored: {batch.request_counts.errored}")
    
    if batch.processing_status == "ended":
        print(f"\n✅ Batch completed, you can now retrieve results.")
    elif batch.processing_status == "in_progress":
        print(f"\n⏳ Batch still in progress, please check again later.")

def retrieve_batch_results(batch_id: str):
    """Fetch batch results and save QA files."""
    client = anthropic.Anthropic(api_key=CONFIG["ANTHROPIC_API_KEY"])
    
    batch = client.beta.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"❌ Batch is not finished yet, current status: {batch.processing_status}")
        return
    
    print(f"📥 Fetching batch results...")
    
    # Load mapping file
    mapping_file = CONFIG["BATCH_REQUESTS_FILE"].replace(".jsonl", "_mapping.json")
    if not os.path.exists(mapping_file):
        print(f"⚠️ Warning: mapping file {mapping_file} not found, using custom_id as file name")
        name_mapping = {}
    else:
        with open(mapping_file, "r", encoding="utf-8") as f:
            name_mapping = json.load(f)
        print(f"✅ Loaded mapping file with {len(name_mapping)} entries")
    
    qa_out_dir = CONFIG["SAVE_QA_DATA"]
    os.makedirs(qa_out_dir, exist_ok=True)
    error_log_path = os.path.join(qa_out_dir, "batch_errors.log")
    
    success_count = 0
    error_count = 0
    
    for result in client.beta.messages.batches.results(batch_id):
        custom_id = result.custom_id
        # Map back to original file name
        file_name = name_mapping.get(custom_id, custom_id)
        
        if result.result.type == "succeeded":
            response_text = result.result.message.content[0].text
            
            ok, objs = validate_jsonl_output(response_text)
            if not ok:
                error_count += 1
                log_error(error_log_path, file_name, "PARSE_ERROR", "Failed to parse JSONL from response", raw=response_text)
                continue
            
            norm_objs = [normalize_qa_object(o) for o in objs if isinstance(o, dict)]
            
            # Save as .jsonl
            out_path_jsonl = os.path.join(qa_out_dir, f"{file_name}.jsonl")
            with open(out_path_jsonl, "w", encoding="utf-8") as f:
                for obj in norm_objs:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
            # Save as .json (single object or list)
            out_path_json = os.path.join(qa_out_dir, f"{file_name}.json")
            with open(out_path_json, "w", encoding="utf-8") as f:
                if len(norm_objs) == 1:
                    json.dump(norm_objs[0], f, ensure_ascii=False, indent=2)
                else:
                    json.dump(norm_objs, f, ensure_ascii=False, indent=2)
            
            success_count += 1
            print(f"✅ [{success_count}] {file_name}")
            
        else:
            error_count += 1
            log_error(
                error_log_path,
                file_name,
                "API_ERROR",
                f"API call failed: {result.result.error.type}"
            )
            print(f"❌ {file_name}: {result.result.error.type}")
    
    print(f"\n📊 Batch result summary:")
    print(f"   Succeeded: {success_count}")
    print(f"   Failed: {error_count}")

# ========== Main ==========
def main():
    import sys
    
    # Handle batch commands
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "check_batch" and len(sys.argv) > 2:
            check_batch_status(sys.argv[2])
            return
        elif cmd == "get_results" and len(sys.argv) > 2:
            retrieve_batch_results(sys.argv[2])
            return
    
    # Load data
    txt_map = load_txt_map(CONFIG["TXT_PATH"])
    json_map = load_json_map(CONFIG["JSON_PATH"])

    n_txt, n_json = len(txt_map), len(json_map)
    common_names = sorted(set(txt_map.keys()) & set(json_map.keys()))

    print(f"📄 Total TXT: {n_txt} | 🧩 Total JSON: {n_json} | ✅ Matched samples: {len(common_names)}")
    if not common_names:
        print("❌ No matched samples found (same-name .txt and .json)")
        return

    start_idx = CONFIG.get("START_IDX", 0)
    if start_idx >= len(common_names):
        print(f"⚠️ START_IDX={start_idx} >= number of matched samples {len(common_names)}, nothing to process.")
        return

    total = len(common_names)
    print(f"📦 Using {total} matched samples (starting from {start_idx+1}/{total})")
    
    # Choose mode
    if CONFIG["USE_BATCH_API"]:
        print("\n🚀 Using batch API mode (cheaper than real-time)")
        requests_file = create_batch_requests_file(common_names, txt_map, json_map, start_idx)
        batch_id = submit_batch_job(requests_file)
        print(f"\n💾 Batch ID saved; later you can fetch results with:")
        print(f"   python {__file__} get_results {batch_id}")
        return
    
    # Real-time mode
    print("\n⚡ Using real-time API mode")
    qa_out_dir = CONFIG["SAVE_QA_DATA"]
    os.makedirs(qa_out_dir, exist_ok=True)
    error_log_path = init_error_log(qa_out_dir)
    
    for i, name in enumerate(common_names[start_idx:], start=start_idx+1):
        print(f"\n🔄 Processing sample {i}/{total}: {name}")
        file_name = name

        txt = txt_map[name]
        json_template_content = json_map[name]

        formatted_user_prompt = user_prompt.format(
            text=txt,
            json_template_content=json_template_content
        )

        # Call API
        try:
            resp = call_anthropic_api_conversation(
                system_prompt=system_prompt,
                user_prompt=formatted_user_prompt,
                api_key=CONFIG["ANTHROPIC_API_KEY"],
                model=CONFIG["ANTHROPIC_MODEL"],
                timeout=CONFIG["TIMEOUT"],
                temperature=CONFIG["TEMPERATURE"],
                max_tokens=CONFIG["MAX_TOKENS"],
            )
        except Exception as e:
            log_error(error_log_path, file_name, "API_CALL", str(e))
            continue

        print("📝 Model response preview:", resp[:200])

        # Parse / validate JSONL
        ok, objs = validate_jsonl_output(resp)
        if not ok:
            print(f"❌ JSONL parse failed [{file_name}]")
            log_error(
                error_log_path,
                file_name,
                "JSON_PARSE",
                "Response cannot be parsed as valid JSONL",
                raw=resp
            )
        else:
            norm_objs = [normalize_qa_object(o) for o in objs if isinstance(o, dict)]
            # Save as .jsonl
            out_path_jsonl = os.path.join(qa_out_dir, f"{file_name}.jsonl")
            with open(out_path_jsonl, "w", encoding="utf-8") as f:
                for obj in norm_objs:
                    f.write(json.dumps(obj, ensure_ascii=False))
                    f.write("\n")

            # Save as .json
            out_path_json = os.path.join(qa_out_dir, f"{file_name}.json")
            with open(out_path_json, "w", encoding="utf-8") as f:
                if len(norm_objs) == 1:
                    json.dump(norm_objs[0], f, ensure_ascii=False, indent=2)
                else:
                    json.dump(norm_objs, f, ensure_ascii=False, indent=2)
            print(f"✅ QA results saved.")

        # Throttle / sleep between requests
        wait_time = random.uniform(CONFIG["SLEEP_MIN"], CONFIG["SLEEP_MAX"])
        print(f"⏳ Sleeping for {wait_time:.1f} seconds")
        time.sleep(wait_time)

if __name__ == "__main__":
    main()