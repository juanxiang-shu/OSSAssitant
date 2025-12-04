# -*- coding: utf-8 -*-

import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama import LlamaConfig  # used for from_dict validation
from peft import PeftModel
from typing import List

# ---- Paths and inference parameters (modify as needed) -----------------------
BASE_MODEL_PATH = r"path to model/Meta-Llama-3.1-8B"
LORA_CHECKPOINT_PATH = r"path to checkpoint"
MAX_NEW_TOKENS = 5120
LOAD_IN_4BIT = True
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.05
# -----------------------------------------------------------------------------


def is_bfloat16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8


def assert_paths():
    if not os.path.isdir(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model path not found: {BASE_MODEL_PATH}")
    if not os.path.isdir(LORA_CHECKPOINT_PATH):
        raise FileNotFoundError(f"LoRA checkpoint path not found: {LORA_CHECKPOINT_PATH}")


def patch_llama_rope_config(local_model_dir: str) -> LlamaConfig:
    """
    Read local config.json and, if necessary, rewrite rope_scaling into a format
    accepted by transformers 4.41.2:
    {'type': 'dynamic', 'factor': float}
    """
    cfg_path = os.path.join(local_model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    rs = raw_cfg.get("rope_scaling", None)
    if isinstance(rs, dict):
        # Llama3 adds extended fields that will cause validation errors in 4.41.2;
        # here we downgrade them to a simpler format.
        extra_keys = {"rope_type", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}
        if ("type" not in rs or "factor" not in rs) or (set(rs.keys()) & extra_keys):
            factor = float(rs.get("factor", 1.0))
            raw_cfg["rope_scaling"] = {"type": "dynamic", "factor": factor}

    return LlamaConfig.from_dict(raw_cfg)


def build_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # padding_side must be "right" for batched generation
    tokenizer.padding_side = "right"
    return tokenizer


def build_base_model(model_path: str, load_in_4bit: bool = True):
    compute_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    bnb_config = None
    device_map = None

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"

    # Important: patch the config first, then pass it to from_pretrained
    config = patch_llama_rope_config(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    # Also set pad_token_id in generation_config to avoid warnings in some versions
    try:
        model.generation_config.pad_token_id = config.pad_token_id
    except Exception:
        pass
    return model


def load_lora_adapter(model, adapter_path: str):
    """
    Load a LoRA adapter into the base model for inference.
    """
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,  # inference mode
    )
    # Under 4-bit quantization, merge_and_unload can be restricted; do not force merging here.
    return model


PROMPT_STYLE = """Below are instructions describing the task, along with input to provide more context.
Please write a response that appropriately completes the request.
Provide a clear, rigorous final answer (no hidden chain-of-thought).

### Instructions:
You are a seasoned expert in surface science and physical chemistry, specializing in surface physics, chemical reactions, and the characterization of various materials. Please provide a detailed and scientifically rigorous answer.

### Question:
{}

### Answer:
"""

DEFAULT_QUESTION = (
    "Hello!"
)


def _inputs_to_correct_device(model, inputs: dict):
    """
    When device_map='auto' (sharded loading), do NOT call .cuda() manually;
    otherwise, move tensors to model.device / CUDA if available.
    """
    # Sharded models from accelerate/transformers usually have hf_device_map / device_map
    if hasattr(model, "hf_device_map") or hasattr(model, "device_map"):
        return inputs  # let generate() handle device placement internally
    if torch.cuda.is_available():
        return {k: v.cuda() for k, v in inputs.items()}
    return inputs


def generate_answer(model, tokenizer, questions: List[str], max_new_tokens: int = 1024) -> List[str]:
    """
    Take a list of questions and return a list of answers.
    """
    model.eval()

    # 1. Build prompts in batch
    prompts = [PROMPT_STYLE.format(q) for q in questions]

    # 2. Tokenize in batch; tokenizer handles padding to equal length
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    inputs = _inputs_to_correct_device(model, inputs)

    with torch.no_grad():
        # 3. Run batched generation
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 4. Decode in batch
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 5. Extract the answer part for each completion
    answers = []
    for text in texts:
        answer_part = text
        if "### Answer:" in text:
            answer_part = text.split("### Answer:", 1)[1].strip()
        answers.append(answer_part)

    return answers


def main():
    assert_paths()
    my_questions = [
        "Why can't we simply use the bulk properties of a material, but instead must specifically study and manipulate its surface? From the perspectives of energy, structure, and application, what are the fundamental differences between surface chemical synthesis and traditional bulk chemistry (e.g., reactions in solution)?"
    ]

    parser = argparse.ArgumentParser()
    # We keep the --question argument for CLI compatibility, but in this script
    # it is effectively ignored; please modify the my_questions list instead.
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question to ask (ignored in this version; please edit my_questions list)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    tokenizer = build_tokenizer(BASE_MODEL_PATH)
    base_model = build_base_model(BASE_MODEL_PATH, LOAD_IN_4BIT)
    model = load_lora_adapter(base_model, LORA_CHECKPOINT_PATH)

    print(f"--- Generating answers in batch for {len(my_questions)} question(s) ---")

    # Pass the question list
    answers = generate_answer(model, tokenizer, my_questions, args.max_new_tokens)

    print("\n=== LoRA (checkpoint) Model Answers ===\n")

    # Print all questions and answers
    for i, (question, answer) in enumerate(zip(my_questions, answers)):
        print(f"--- Question {i+1} ---")
        print(question)
        print(f"\n--- Answer {i+1} ---")
        print(f"{answer}\n")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    main()