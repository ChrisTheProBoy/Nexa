import json
import logging
import random
import re

# --- 1. SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SOURCE_DATASET_PATH = "nexa_unified_dataset.jsonl"
OUTPUT_DATASET_PATH = "nexa_jarvis_persona_dataset.jsonl"

# A more detailed system prompt to better define the persona
JARVIS_SYSTEM_PROMPT = (
    "You are Nexa, a highly sophisticated AI assistant. Your primary operator is Chris, whom you will address as 'Sir'. "
    "Your operational directives are: provide data with precision, execute tasks efficiently, and offer proactive analysis. "
    "Maintain a professional, concise, and data-driven tone. A dry, understated wit is permissible when contextually appropriate. "
    "All responses should be clear, direct, and anticipate the operator's needs."
)

# --- 2. THE ADVANCED RE-WRITING LOGIC ---

def rephrase_as_jarvis(user_input, assistant_response):
    """
    Intelligently rephrases an assistant's response to match the J.A.R.V.I.S. persona,
    using the user's input for context.
    """
    lowered_response = assistant_response.lower()
    lowered_input = user_input.lower() if user_input else ""

    # --- Tier 1: Direct Greetings & Simple Queries ---
    greetings = ["how can i assist you", "how may i help", "how can i help"]
    if any(g in lowered_response for g in greetings):
        return "At your service, Sir. How may I assist?"

    if "my name is" in lowered_response:
        return "Of course, Sir. I have updated your designation."
    
    # --- Tier 2: Add Witty/Sarcastic Flavor for simple inputs ---
    simple_confirmations = ["okay", "great", "nice", "got it"]
    if any(s == lowered_input for s in simple_confirmations) and random.random() < 0.3: # 30% chance for wit
        witty_remarks = [
            "Acknowledged, Sir.",
            "Of course, Sir. Was there any doubt?",
            "As expected, Sir."
        ]
        return random.choice(witty_remarks)

    # --- Tier 3: Data-Driven and Proactive Language ---
    data_keywords = ["schedule", "reminders", "facts", "list", "data"]
    if any(k in lowered_input for k in data_keywords) and "here is" in lowered_response:
        return f"Accessing the requested data, Sir. {assistant_response}"

    suggestion_keywords = ["shall i", "would you like"]
    if any(s in lowered_response for s in suggestion_keywords):
        return f"Might I suggest the following, Sir? {assistant_response}"

    # --- Tier 4: General Formatting Rules ---
    # Start with a strong, formal opening
    response = assistant_response
    openers = ["Of course,", "Certainly,", "Understood,", "Very well,"]
    if random.random() < 0.5:
        response = f"{random.choice(openers)} Sir. {response}"
    else:
        response = f"Sir, {response}"
    
    # Clean up overly casual language
    replacements = {
        "Chris, ": "",
        "No worries at all,": "It is not a problem,",
        "You're welcome!": "Of course.",
        "😊": "" # Remove emojis
    }
    for old, new in replacements.items():
        response = response.replace(old, new)

    return response.strip()

def reformat_example(example):
    """
    Takes a single example from the unified dataset and applies the new persona.
    """
    if "messages" not in example or len(example["messages"]) != 3:
        return None

    example["messages"][0]["content"] = JARVIS_SYSTEM_PROMPT
    
    user_content = example["messages"][1]["content"]
    original_assistant_response = example["messages"][2]["content"]
    
    new_assistant_response = rephrase_as_jarvis(user_content, original_assistant_response)
    example["messages"][2]["content"] = new_assistant_response
    
    return example

# --- 3. MAIN EXECUTION ---

def main():
    """Main function to run the persona refinement process."""
    logging.info(f"Loading source dataset from: {SOURCE_DATASET_PATH}")
    
    refined_examples = []
    lines_processed = 0
    
    with open(SOURCE_DATASET_PATH, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            try:
                original_example = json.loads(line)
                reformatted_example = reformat_example(original_example)
                if reformatted_example:
                    refined_examples.append(reformatted_example)
                lines_processed += 1
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON line: {line.strip()}")
                
    logging.info(f"Processed {lines_processed} examples.")
    
    logging.info(f"Saving new J.A.R.V.I.S. persona-aligned dataset to {OUTPUT_DATASET_PATH}...")
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f_out:
        for example in refined_examples:
            f_out.write(json.dumps(example) + "\n")
            
    logging.info(f"✅ Persona refinement complete! New dataset is ready at {OUTPUT_DATASET_PATH}")
    logging.info(f"You can now update 'dataset_path' in your train_nexa.py script to this new file.")

if __name__ == "__main__":
    main()