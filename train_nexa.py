import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from huggingface_hub import login


# 🔑 Login to Hugging Face Hub
login("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")  # replace with your real token


# ⚙️ BitsAndBytes 4-bit quantization config (QLoRA ready)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 📂 Paths
log_dir = os.path.expanduser("~/Documents/Vs Code/nexa-assistant/logs")
os.makedirs(log_dir, exist_ok=True)
dataset_path = os.path.expanduser("~/Documents/Vs Code/nexa-assistant/nexa_dataset_5000.jsonl")
output_dir = os.path.expanduser("~/Documents/Vs Code/nexa-assistant/nexa_finetuned")

# 📝 Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "finetune_mistral.log"), encoding="utf-8")],
)


def main():
    try:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        # 🔄 Load tokenizer & model
        logging.info(f"Loading model {model_name} with 4-bit quantization")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # 📊 Load dataset
        logging.info(f"Loading dataset from {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} not found")

        dataset = load_dataset("json", data_files={"train": dataset_path})

        # 🔧 Preprocessing
        def preprocess_function(examples):
            texts = []
            for msgs in examples["messages"]:
                if len(msgs) >= 3:  # Ensure structure matches
                    text = (
                        f"System: {msgs[0]['content']}\n\n"
                        f"User: {msgs[1]['content']}\n"
                        f"Assistant: {msgs[2]['content']}"
                    )
                    texts.append(text)
            return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

        logging.info("Preprocessing dataset")
        dataset = dataset.map(preprocess_function, batched=True, remove_columns=["messages"])

        # 🎯 Training arguments (small VRAM safe)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Small batch size
            gradient_accumulation_steps=8,  # Accumulate grads
            learning_rate=2e-5,
            save_steps=200,
            save_total_limit=2,
            logging_dir=log_dir,
            logging_steps=50,
            fp16=True,
            gradient_checkpointing=True,  # Saves memory
            optim="paged_adamw_32bit",  # QLoRA optimizer
            report_to="none",  # Disable WandB unless you want logging
        )

        # 🚀 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
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
