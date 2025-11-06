# train_nexa_fixed.py
"""
train_nexa_fixed.py
A corrected LoRA/PEFT training script for Nexa (Friday cinematic persona).
Backwards-compatible with older `transformers` versions that expect eval_steps/save_steps.
Defaults tuned for a laptop with RTX 3050 Ti (6-8 GB VRAM) on Fedora 42.

Environment variables you may set:
  NEXA_BASE_MODEL      - base model name (default: "gpt2")
  NEXA_TRAIN_FILE      - path to training JSONL (default: ./nexa_train_cinematic.jsonl)
  NEXA_VAL_FILE        - path to validation JSONL (default: ./nexa_val_cinematic.jsonl)
  NEXA_OUTPUT_DIR      - where to save model (default: ./peft_nexa_cinematic)
  NEXA_MAX_LENGTH      - tokenization max length (default: 512)
  NEXA_BATCH_SIZE      - per-device batch size (default: 2)
  NEXA_GRAD_ACCUM      - gradient accumulation steps (default: 8)
  NEXA_EPOCHS          - number of epochs (default: 3)
  NEXA_LR              - learning rate (default: 3e-4)
  NEXA_LORA_R          - LoRA r (default: 8)
  NEXA_LORA_ALPHA      - LoRA alpha (default: 32)
  NEXA_LORA_DROPOUT    - LoRA dropout (default: 0.05)

Notes:
 - If you have limited VRAM, prefer a small base model like "gpt2" or "distilgpt2".
 - For larger models you'll need quantization + bitsandbytes + careful device_map settings.
"""

import os
import logging
import sys

# ---- Configuration from environment (with sane defaults) ----
TRAIN_FILE = os.environ.get("NEXA_TRAIN_FILE", "./nexa_train_cinematic.jsonl")
VAL_FILE = os.environ.get("NEXA_VAL_FILE", "./nexa_val_cinematic.jsonl")
OUTPUT_DIR = os.environ.get("NEXA_OUTPUT_DIR", "./peft_nexa_cinematic")
MODEL_NAME = os.environ.get("NEXA_BASE_MODEL", "gpt2")

MAX_LENGTH = int(os.environ.get("NEXA_MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("NEXA_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("NEXA_GRAD_ACCUM", "8"))
NUM_EPOCHS = int(os.environ.get("NEXA_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("NEXA_LR", "3e-4"))

LORA_R = int(os.environ.get("NEXA_LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("NEXA_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("NEXA_LORA_DROPOUT", "0.05"))

# ---- Logging ----
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("train_nexa_fixed")

def main():
    # Try imports and show helpful error if missing
    try:
        from datasets import load_dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            Trainer,
            TrainingArguments,
        )
        from transformers import DataCollatorForLanguageModeling
        from peft import get_peft_model, LoraConfig, TaskType
        import torch
    except Exception as e:
        logger.error("Missing required Python packages. Please install: transformers datasets peft accelerate safetensors torch")
        logger.exception(e)
        raise

    # Basic sanity checks
    if not os.path.exists(TRAIN_FILE):
        logger.error(f"Train file not found: {TRAIN_FILE}")
        raise FileNotFoundError(TRAIN_FILE)
    if not os.path.exists(VAL_FILE):
        logger.error(f"Val file not found: {VAL_FILE}")
        raise FileNotFoundError(VAL_FILE)

    logger.info(f"Using base model: {MODEL_NAME}")
    logger.info(f"Train file: {TRAIN_FILE}")
    logger.info(f"Val file: {VAL_FILE}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    # Load dataset
    logger.info("Loading dataset...")
    ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    logger.info(f"Loaded dataset sizes -> train: {len(ds['train'])}, val: {len(ds['validation'])}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        logger.info("Tokenizer has no pad token—adding [PAD] token.")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model (attempt GPU auto device_map; fallback to cpu if needed)
    logger.info("Loading base model (this may take time)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    except Exception as e:
        logger.warning("Auto device_map load failed or insufficient GPU memory; falling back to low_cpu_mem_usage load.")
        logger.debug("Model load exception:", exc_info=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)

    # Resize embeddings if tokenizer changed
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA (PEFT)
    logger.info("Applying LoRA adapters (PEFT)...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=None,  # auto-detect; set to ["q_proj","v_proj"] for LLaMA-like models if needed
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # Preprocessing: combine prompt+completion, tokenize
    def preprocess(example):
        # expected keys: "prompt" and "completion"
        text = (example.get("prompt", "") + example.get("completion", "")).strip()
        tok = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok

    logger.info("Tokenizing dataset...")
    tokenized = ds.map(preprocess, remove_columns=ds["train"].column_names, batched=False)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TrainingArguments (backwards-compatible: using eval_steps/save_steps)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        # Use eval_steps/save_steps for compatibility with older transformers
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training run — monitor GPU with `nvidia-smi` if available.")
    trainer.train()

    logger.info("Training finished — saving model and tokenizer.")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
