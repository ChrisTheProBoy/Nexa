import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
# --- ADD THIS IMPORT ---
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ⚙️ BitsAndBytes 4-bit quantization config (QLoRA ready)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 📂 Paths (Corrected for Colab)
log_dir = "/content/drive/MyDrive/nexa-assistant/logs"
os.makedirs(log_dir, exist_ok=True)
dataset_path = "/content/drive/MyDrive/nexa-assistant/nexa_jarvis_persona_dataset.jsonl"
output_dir = "/content/drive/MyDrive/nexa-assistant/nexa_finetuned"

# 📝 Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "finetune_mistral.log"), encoding="utf-8")],
)

def main():
    try:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        # 🔄 Load tokenizer & model
        logging.info(f"Loading model {model_name} with 4-bit quantization")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        # --- PREPARE MODEL FOR K-BIT TRAINING ---
        model = prepare_model_for_kbit_training(model)
        
        # --- PEFT/LORA CONFIGURATION ---
        model.config.pretraining_tp = 1
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        # 📊 Load dataset
        logging.info(f"Loading dataset from {dataset_path}")
        dataset = load_dataset("json", data_files={"train": dataset_path})

        # 🔧 Preprocessing
        def preprocess_function(examples):
            texts = []
            for msgs in examples["messages"]:
                if len(msgs) >= 3:
                    text = (
                        f"<s>[INST] {msgs[0]['content']}\n{msgs[1]['content']} [/INST]"
                        f"{msgs[2]['content']}</s>"
                    )
                    texts.append(text)
            
            tokenized_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"][:]
            return tokenized_inputs

        logging.info("Preprocessing dataset")
        dataset = dataset.map(preprocess_function, batched=True, remove_columns=["messages"])

        # 🎯 Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            save_steps=200,
            save_total_limit=2,
            logging_dir=log_dir,
            logging_steps=50,
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            report_to="none",
        )

        # 🚀 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        logging.info("Starting fine-tuning")
        trainer.train()

        # 💾 Save fine-tuned model
        logging.info(f"Saving fine-tuned model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"✅ Fine-tuning complete. Model saved to {output_dir}")

    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        print(f"❌ An error occurred during fine-tuning: {e}")

if __name__ == "__main__":
    main()