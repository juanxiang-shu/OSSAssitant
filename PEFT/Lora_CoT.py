import huggingface_hub
import swanlab as swl
import torch
import sys

# Make sure is_bfloat16_supported exists in the runtime, or check directly in quantization config
def is_bfloat16_supported():
    """Simple check for bfloat16 support"""
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8


run = swl.init(
    project='Finetune-server-Meta-Llama-3.1-8B-total',
    job_type="training",
    anonymous="allow",
    mode="local"
)

"""
Step 2: Load model and tokenizer
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Model configuration parameters
max_seq_length = 4096  # Maximum sequence length
dtype = None           # Data type; None means auto-select
load_in_4bit = True    # Use 4-bit quantization to save GPU memory

# Exact path of the model on local disk
local_model_path = "/data/home/models/Meta-Llama-3.1-8B"  # Change to your model path
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

# 1. Define BitsAndBytes configuration
bnb_config = None
if load_in_4bit:
    # Automatically select compute dtype
    compute_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # Use NF4 quantization
        bnb_4bit_compute_dtype=compute_dtype,  # Set compute type
        bnb_4bit_use_double_quant=True,     # Enable double quantization
    )

# 2. Load pretrained model and tokenizer
# Use local_model_path instead of a remote model name
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,      # Use local path
    trust_remote_code=True,
    local_files_only=True,  # Force loading from local files only
)
# Note: some models do not define a padding token by default, so we set it explicitly
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

device_map = "auto" if load_in_4bit else None

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=bnb_config,  # Use defined quantization configuration
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    device_map=device_map,
)

# 3. Prepare model for k-bit training (important step)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Disable cache during training to save memory

"""
Step 3: Define prompt template and run inference test before fine-tuning
"""
prompt_style = """Below are instructions describing the task, along with input to provide more context.
Please write a response that appropriately completes the request.
Before generating your final response, please think through the steps and ensure your answer is logical and scientifically rigorous.

### Instructions:
You are a seasoned expert in surface science and physical chemistry, specializing in surface physics, chemical reactions, and the characterization of various materials. Please provide a detailed and scientifically rigorous answer to the question you provided.

### Question:
{}

### Answer:
<think>{}"""

# Surface science question for testing
question = "On the Au(111) surface, the formation of free radicals by C-Br bond cleavage is the first step in the Ullmann coupling reaction. What is the optimal temperature range for bromine desorption? Is this process thermally activated desorption or surface-assisted homolytic/heterolytic cleavage? How can I observe surface vacancies or free radical intermediates after desorption using STM?"

# Set model to inference mode
model.eval()
model.config.use_cache = True  # Re-enable cache for inference

inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt")
if not load_in_4bit:
    inputs = inputs.to(device)

# Generate answer
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=4096,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print("### Inference result before fine-tuning:")
print(response[0].split("### Answer:")[1].replace(tokenizer.eos_token, "").strip())

"""
Step 4: Dataset formatting function
"""
train_prompt_style = """Below are instructions describing the task, along with input to provide more context.
Please write a response that appropriately completes the request.
Before generating your final response, please think through the steps and ensure your answer is logical and scientifically rigorous.

### Instructions:
You are a seasoned expert in surface science and physical chemistry, specializing in surface physics, chemical reactions, and the characterization of various materials. Please provide a detailed and scientifically rigorous answer to the question you provided.

### Question:
{}

### Answer:
<think>
{}
</think>
{}"""

EOS_TOKEN = tokenizer.eos_token  # Add an end-of-sequence marker

# Formatting function, used to process samples in the dataset
def formatting_prompts_func(examples):
    # Extract questions, chains-of-thought, and answers from examples
    inputs = examples["question"]   # List of surface science questions
    cots = examples["reasoning"]    # List of chains-of-thought
    outputs = examples["answer"]    # List of answers

    # Store formatted texts
    texts = []

    # Iterate over each sample and combine question, reasoning, and answer with the template
    for input_text, cot, output in zip(inputs, cots, outputs):
        # Format text using the train_prompt_style template and append EOS
        text = train_prompt_style.format(input_text, cot, output) + EOS_TOKEN
        texts.append(text)

    # Return dictionary of formatted texts
    return {
        "text": texts,
    }


# Load dataset and apply formatting
from datasets import load_dataset
raw_dataset = load_dataset("json", data_files={"train": "/data/home/peft/total.jsonl"}, split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

# Apply formatting function to train and validation splits
train_dataset = split_dataset["train"].map(formatting_prompts_func, batched=True, remove_columns=raw_dataset.column_names)
eval_dataset = split_dataset["test"].map(formatting_prompts_func, batched=True, remove_columns=raw_dataset.column_names)

"""
Step 5: Configure LoRA fine-tuning parameters

Use LoRA for parameter-efficient fine-tuning
"""
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType, PeftConfig

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # LoRA rank, controls the dimension of low-rank matrices; higher values mean more trainable params
    # and potentially better performance but higher training cost. Recommended range: 8–32
    r=8,
    # Target modules to which LoRA will be applied
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention-related layers
        # "gate_proj", "up_proj", "down_proj",   # FFN-related layers
    ],
    # LoRA scaling factor; larger values increase the impact of LoRA updates
    lora_alpha=16,
    # Dropout rate in LoRA layers to prevent overfitting; 0 means no dropout
    # If the dataset is small, a value around 0.1 is recommended
    lora_dropout=0,
    # Whether to fine-tune bias parameters; "none" means do not fine-tune bias
    # "none": do not fine-tune biases
    # "all": fine-tune all biases
    # "lora_only": fine-tune only LoRA-related biases
    bias="none",
    # Whether to use rank-stabilized LoRA; not used here
    # It slightly slows training but can significantly reduce memory usage
    use_rslora=False,
    # LoFTQ configuration; not used here, otherwise can further compress model size
    loftq_config=None,
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Print trainable parameters (optional)
model.print_trainable_parameters()

"""
Step 6: Configure training parameters and initialize trainer
"""
from trl import SFTTrainer  # Supervised fine-tuning trainer
from transformers import TrainingArguments  # Training configuration

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,              # Model to be trained
    tokenizer=tokenizer,      # Tokenizer
    train_dataset=train_dataset,  # Training split
    eval_dataset=eval_dataset,    # Validation split
    dataset_text_field="text",    # Name of the text field in the dataset
    max_seq_length=max_seq_length,  # Maximum sequence length
    dataset_num_proc=1,            # Number of processes for dataset preprocessing
    args=TrainingArguments(
        per_device_train_batch_size=1,  # Training batch size per GPU
        gradient_accumulation_steps=4,  # Gradient accumulation steps to simulate larger batch size
        warmup_steps=50,                # Warmup steps for gradually increasing learning rate
        learning_rate=2e-4,             # Learning rate
        lr_scheduler_type="cosine",
        # max_steps=100,                # Max training steps (one step = one batch)
        gradient_checkpointing=True,
        # Enable bfloat16 for gradient checkpointing if supported
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # Choose training precision based on hardware support
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,               # Log every 10 steps
        optim="adamw_8bit",             # 8-bit AdamW to save memory with minimal effect on performance
        weight_decay=0.01,              # Weight decay for regularization
        seed=3407,                      # Random seed
        output_dir="swanlab",           # Directory to save checkpoints and logs
        # evaluation_strategy="steps",   # Enable step-based evaluation if desired
        evaluation_strategy="epoch",    # Evaluate once per epoch
        # eval_steps=10,                # Evaluate every 10 steps (if using step-based strategy)
    ),
)

"""
Step 7: Start training
"""
print("===================== Start Training =====================")
trainer.train()
print("===================== Training Finished =====================")

"""
Step 8: Inference test after fine-tuning
"""
question = "On the Au(111) surface, the formation of free radicals by C-Br bond cleavage is the first step in the Ullmann coupling reaction. What is the optimal temperature range for bromine desorption? Is this process thermally activated desorption or surface-assisted homolytic/heterolytic cleavage? How can I observe surface vacancies or free radical intermediates after desorption using STM?"

model.eval()
model.config.use_cache = True  # Re-enable cache for inference

# Encode the question, convert to tensors, and move to GPU if needed
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt")
if not load_in_4bit:
    inputs = inputs.to(device)

# Generate answer
outputs = model.generate(
    input_ids=inputs.input_ids,           # Input token ID sequence
    attention_mask=inputs.attention_mask, # Attention mask marking valid positions
    max_new_tokens=4096,                  # Max number of new tokens to generate
    use_cache=True,                       # Use KV cache to speed up generation
)

# Decode model output
response = tokenizer.batch_decode(outputs)
print("### Inference result after fine-tuning:")
print(response[0].split("### Answer:")[1].replace(tokenizer.eos_token, "").strip())

"""
Step 9: Save model (standard PEFT workflow)
"""
new_model_local = "/data/home/models/Meta-Llama-3.1-8B-total"

# 1. Save LoRA weights and tokenizer
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)
print(f"LoRA weights saved to: {new_model_local}")

# 2. Merge LoRA weights into the base model and save the full model
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model_path = "/data/home/models/Meta-Llama-3.1-8B"
merged_model_path = f"{new_model_local}_merged"

# 2. Merge LoRA weights into base model and save full 16-bit model
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # or torch.bfloat16
        trust_remote_code=True,
    )

    config = PeftConfig.from_pretrained(new_model_local)

    # If loftq_config is None, set it to {} to avoid 'NoneType' errors inside peft
    if config.loftq_config is None:
        config.loftq_config = {}

    # Load LoRA weights from the saved path and merge them
    merged_model = PeftModel.from_pretrained(base_model, new_model_local, config=config)

    merged_model = merged_model.merge_and_unload()

    # Save merged full-precision model
    merged_model.save_pretrained(merged_model_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged 16-bit model saved to: {merged_model_path}")
except Exception as e:
    print(
        f"Failed to merge model; possible reasons: insufficient memory or outdated/buggy Peft version: {e}",
        file=sys.stderr,
    )