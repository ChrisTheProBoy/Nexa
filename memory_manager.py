import os
import json
from datetime import datetime
from pathlib import Path


class MemoryManager:
    def __init__(self, user_id: str, memory_dir: str = "memory"):
        self.user_id = user_id
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.memory_path = self.memory_dir / f"{user_id}.json"
        self.dataset_path = self.memory_dir / f"{user_id}_dataset.jsonl"

        self.data = {
            "facts": {},
            "traits": {},
            "preferences": {},
            "reminders": [],
            "interactions": [],
        }
        self.load_memory()

    # ----------------- Memory Persistence -----------------
    def load_memory(self):
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"[MemoryManager] Failed to load memory: {e}")

    def save_memory(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[MemoryManager] Failed to save memory: {e}")

    # ----------------- Interaction Logging -----------------
    def save_interaction(self, role: str, content: str, **kwargs):
        timestamp = datetime.now().isoformat()
        interaction = {"role": role, "content": content, "time": timestamp}
        if kwargs:
            interaction.update(kwargs)

        self.data["interactions"].append(interaction)
        self.save_memory()

        # Log to dataset
        self.log_dataset_entry(role, content)

    def log_dataset_entry(self, role: str, content: str):
        try:
            with open(self.dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"role": role, "content": content}) + "\n")
        except Exception as e:
            print(f"[MemoryManager] Failed to log dataset entry: {e}")

    # ----------------- Knowledge Storage -----------------
    def remember_fact(self, key: str, value: str):
        self.data["facts"][key] = value
        self.save_memory()

    def set_user_name(self, name: str):
        """
        Legacy support: sets the preferred name to address the user as.
        Equivalent to updating the 'address_me_as' preference.
        """
        self.update_preference("address_me_as", name)

    def get_user_name(self) -> str:
        """
        Legacy support: returns the preferred name to address the user as.
        Looks up 'address_me_as' in preferences, or defaults to 'User'.
        """
        return self.get_preferences().get("address_me_as", "User")

    def update_trait(self, key: str, value: str):
        self.data["traits"][key] = value
        self.save_memory()

    def update_preference(self, key: str, value: str):
        """
        Update or add a user preference (e.g., programming_language=C, tone=brief).
        """
        self.data.setdefault("preferences", {})
        self.data["preferences"][key] = value
        self.save_memory()

    def add_reminder(self, reminder: str, time: str):
        self.data["reminders"].append({"message": reminder, "time": time})
        self.save_memory()

    # ----------------- Retrieval -----------------
    def get_facts(self) -> dict:
        return self.data.get("facts", {})

    def get_traits(self) -> dict:
        return self.data.get("traits", {})

    def get_preferences(self) -> dict:
        return self.data.get("preferences", {})

    def get_reminders(self) -> list:
        return self.data.get("reminders", [])

    def search_interactions(self, query: str, top_k: int = 5):
        """
        Simple keyword search in interactions.
        """
        matches = []
        for i in reversed(self.data.get("interactions", [])):
            if query.lower() in i["content"].lower():
                matches.append(i["content"])
            if len(matches) >= top_k:
                break
        return matches

    def retrieve_context(self, query: str = "", top_k: int = 5) -> str:
        """
        Retrieve a combined context string from memory: facts, traits, reminders, preferences, and recent interactions.
        """
        parts = []

        facts = self.get_facts()
        if facts:
            parts.append(f"Facts: {facts}")

        traits = self.get_traits()
        if traits:
            parts.append(f"Traits: {traits}")

        reminders = self.get_reminders()
        if reminders:
            parts.append(f"Reminders: {reminders}")

        prefs = self.get_preferences()
        if prefs:
            parts.append(f"Preferences: {prefs}")

        if query:
            interactions = self.search_interactions(query, top_k=top_k)
        else:
            interactions = [i["content"] for i in self.data.get("interactions", [])[-top_k:]]

        if interactions:
            parts.append("Recent interactions: " + "; ".join(interactions))

        return "\n".join(parts) if parts else "No context available."
