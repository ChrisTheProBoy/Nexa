import json
import os

existing_entries = set()
with open("nexa_dataset_5000.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        user_input = entry["messages"][1]["content"]
        existing_entries.add(user_input)

with open("new_interactions.jsonl", "r", encoding="utf-8") as f, open("nexa_dataset_5000_updated.jsonl", "w", encoding="utf-8") as out:
    for line in f:
        entry = json.loads(line)
        if entry["messages"][1]["content"] not in existing_entries:
            json.dump(entry, out)
            out.write("\n")

# Replace original dataset
os.rename("nexa_dataset_5000_updated.jsonl", "nexa_dataset_5000.jsonl")
print("Merged new interactions into nexa_dataset_5000.jsonl")