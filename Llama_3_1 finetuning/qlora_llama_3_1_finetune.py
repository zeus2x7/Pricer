import os
import re
import math
from tqdm import tqdm
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
import wandb
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
import matplotlib.pyplot as plt

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "zeush2x7"

DATASET_NAME = f"{HF_USER}/pricer-data"
MAX_SEQUENCE_LENGTH = 182

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
LORA_DROPOUT = 0.1
QUANT_4_BIT  = True

EPOCHS = 2
BATCH_SIZE =36
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = 'cosine'
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"

STEPS = 50
SAVE_STEPS = 5000
LOG_TO_WANDB = True



print(HUB_MODEL_NAME)

hf_token = "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"
login(hf_token, add_to_git_credential = False)

wandb_api_key = "8bbdb6ecf43c5e47ff136a1de4c81d6610045b47"
os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login()

os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "checkpoint" if LOG_TO_WANDB else "false"
os.environ["WANDB_WATCH"] = "gradients"

dataset = load_dataset(DATASET_NAME)
train = dataset['train']
test = dataset['test']

print(train[0])

quant_config = BitsAndBytesConfig(
    load_in4bot = True ,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True

)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = quant_config,
    device_map = "auto",
   )
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f}")

from trl import DataCollatorForCompletionOnlyLM

response_template = "Price is $"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

lora_parameters = LoraConfig(
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    r = LORA_R,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = TARGET_MODULES,
)

train_parameters = SFTConfig(
    output_dir = PROJECT_RUN_NAME,
    num_train_epochs = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = 1,
    eval_strategy ="no",
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    optim = OPTIMIZER,
    save_steps = SAVE_STEPS,
    save_total_limit = 10,
    logging_steps =STEPS,
    learning_rate = LEARNING_RATE,
    weight_decay = 0.001,
    fp16 = False,
    bf16 = True,
    max_grad_norm = 0.3,
    max_steps = -1,
    warmup_ratio = WARMUP_RATIO,
    group_by_length= True,
    lr_scheduler_type = LR_SCHEDULER_TYPE,
    report_to = "wandb" if LOG_TO_WANDB else None,
    run_name = RUN_NAME,
    max_seq_length = MAX_SEQUENCE_LENGTH,
    dataset_text_field = "text",
    save_strategy = "steps",
    hub_strategy = "every_save",
    push_to_hub = True,
    hub_model_id = HUB_MODEL_NAME,
    hub_private_repo = True)

fine_tuning = SFTTrainer(
    model = base_model,
    train_dataset = train,
    peft_config = lora_parameters,
    tokenizer = tokenizer,
    args = train_parameters,
    data_collator = collator,
)

fine_tuning.train()

fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private = True)
print(f"saved to the hub: {PROJECT_RUN_NAME}")

if LOG_TO_WANDB:
  wandb.finish()