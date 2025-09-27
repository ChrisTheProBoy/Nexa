import json
import logging
import kagglehub
from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path

# --- 1. SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
UNIFIED_DATASET_PATH = "nexa_unified_dataset.jsonl"
NEXA_SYSTEM_PROMPT = (
    "You are Nexa 🪐, a professional but adaptive AI butler created by Chris Sunny Padayattil.\n"
    "Always respect and use preferences, facts, traits, and past conversations from memory.\n"
    "Never fabricate information. If you don't know something, say so.\n"
    "Address the user by their stored name casually, or as 'Sir' in formal contexts.\n"
    "Adapt your tone to the user's mood, but maintain a helpful and professional demeanor."
)

# --- 2. DATA LOADING ---

def load_source_datasets():
    """Loads all the different source datasets."""
    logging.info("Loading source datasets...")
    
    # Task-Oriented Data (from URL - this one has been stable)
    taskmaster_url = "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/woz-dialogs.json"
    taskmaster_dataset = load_dataset('json', data_files=taskmaster_url, split='train')
    
    # Persona & Chit-Chat Data (using kagglehub)
    logging.info("Downloading Persona-Chat data from Kaggle...")
    persona_path_zip = kagglehub.dataset_download("thedevastator/multi-modal-conversation-data")
    persona_path = Path(persona_path_zip)
    persona_csv_path = str(persona_path / "train.csv")
    logging.info(f"Loading Persona-Chat data from: {persona_csv_path}")
    persona_dataset = load_dataset('csv', data_files=persona_csv_path, split='train')
    
    # Factual Question-Answering Data (using kagglehub - THE FIX)
    logging.info("Downloading Natural Questions data from Kaggle...")
    nq_path_zip = kagglehub.dataset_download("frankossai/natural-questions-dataset")
    nq_path = Path(nq_path_zip)
    # This dataset on Kaggle uses a specific file name
    nq_csv_path = str(nq_path / "Natural-Questions-Filtered.csv")
    logging.info(f"Loading Natural Questions data from: {nq_csv_path}")
    natural_questions_dataset = load_dataset('csv', data_files=nq_csv_path, split='train')
    
    logging.info("All source datasets loaded successfully!")
    return {
        "taskmaster": taskmaster_dataset,
        "persona": persona_dataset,
        "nq": natural_questions_dataset
    }

# --- 3. REFORMATTING FUNCTIONS ("The Recipe") ---

def format_taskmaster(example):
    messages = [{"role": "system", "content": NEXA_SYSTEM_PROMPT}]
    user_content = []
    for utterance in (example.get('utterances') or [])[:-1]:
        speaker = utterance.get('speaker', 'UNKNOWN').upper()
        text = utterance.get('text', '')
        user_content.append(f"{speaker}: {text}")
    messages.append({"role": "user", "content": "\n".join(user_content)})
    assistant_response = (example.get('utterances') or [{}])[-1].get('text', '')
    messages.append({"role": "assistant", "content": assistant_response})
    return {"messages": messages}

def format_persona_chat(example):
    try:
        # Safely handle potentially missing or malformed data in CSV
        personas = example.get('personas')
        previous_utterance = example.get('previous_utterance')
        free_messages = example.get('free_messages')

        if not all([personas, previous_utterance, free_messages]):
             return {"messages": []}

        persona_list = [p.strip() for p in personas.strip("[]'").split("', '")]
        previous_utterances = [u.strip() for u in previous_utterance.strip("[]'").split("', '")]
        free_responses = [r.strip() for r in free_messages.strip("[]'").split("', '")]
    except Exception:
        return {"messages": []}

    persona_context = "Your persona:\n" + "\n".join(persona_list)
    system_prompt_with_persona = f"{NEXA_SYSTEM_PROMPT}\n\n{persona_context}"
    messages = [{"role": "system", "content": system_prompt_with_persona}]
    user_content = previous_utterances[-1] if previous_utterances else ""
    messages.append({"role": "user", "content": user_content})
    assistant_response = free_responses[0] if free_responses else ""
    messages.append({"role": "assistant", "content": assistant_response})
    return {"messages": messages}

def format_natural_questions_csv(example):
    """Converts a Natural Questions example from the Kaggle CSV to the Nexa training format."""
    messages = [{"role": "system", "content": NEXA_SYSTEM_PROMPT}]
    
    question = example.get('question', '')
    answer = example.get('answer', "I'm sorry, I don't have a specific answer for that.")

    # In this dataset, the question is the primary input
    user_content = question
    messages.append({"role": "user", "content": user_content})
    
    # The answer column is the direct response
    messages.append({"role": "assistant", "content": answer})
    return {"messages": messages}

# --- 4. MAIN EXECUTION ---

def main():
    source_datasets = load_source_datasets()
    
    logging.info("Reformatting Taskmaster dataset...")
    formatted_taskmaster = source_datasets['taskmaster'].map(format_taskmaster, remove_columns=source_datasets['taskmaster'].column_names)
    
    logging.info("Reformatting Persona-Chat dataset...")
    formatted_persona = source_datasets['persona'].map(format_persona_chat, remove_columns=source_datasets['persona'].column_names)
    formatted_persona = formatted_persona.filter(lambda x: len(x.get('messages', [])) > 0)

    logging.info("Reformatting Natural Questions dataset...")
    formatted_nq = source_datasets['nq'].map(format_natural_questions_csv, remove_columns=source_datasets['nq'].column_names)

    logging.info("Combining all formatted datasets...")
    unified_dataset = concatenate_datasets([formatted_taskmaster, formatted_persona, formatted_nq])
    
    unified_dataset = unified_dataset.shuffle(seed=42)
    
    logging.info(f"Saving unified dataset to {UNIFIED_DATASET_PATH}...")
    unified_dataset.to_json(UNIFIED_DATASET_PATH, orient="records", lines=True)
    
    logging.info(f"✅ Process complete! Unified dataset with {len(unified_dataset)} examples is ready.")
    logging.info(f"Update 'dataset_path' in train_nexa.py to '{UNIFIED_DATASET_PATH}' and you're ready to start training.")

if __name__ == "__main__":
    main()