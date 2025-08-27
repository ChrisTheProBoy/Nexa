import os
import json
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")


class MemoryManager:
    """
    Advanced memory system with embeddings, facts, traits, reminders, and interactions.
    Also logs all interactions into a dataset JSONL file for fine-tuning later.
    """

    def __init__(self, user_id: str, memory_dir: str = "memory"):
        self.user_id = user_id.lower().replace(" ", "_")
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

        self.memory_path = os.path.join(self.memory_dir, f"{self.user_id}.json")
        self.dataset_path = os.path.join(self.memory_dir, f"{self.user_id}_dataset.jsonl")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Default structure
        self.data = {
            "facts": {},
            "traits": {},
            "reminders": [],
            "interactions": [],
            "relationships": {}
        }
        self.load_memory()

    # ---------------- Persistence ----------------
    def load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logging.info(f"[MemoryManager] Loaded memory for {self.user_id}")
            except Exception as e:
                logging.error(f"[MemoryManager] Failed to load memory: {e}")

    def save_memory(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"[MemoryManager] Failed to save memory: {e}")

    # ---------------- Dataset logging ----------------
    def log_dataset_entry(self, role: str, content: str):
        """Append an interaction to dataset JSONL file."""
        entry = {"role": role, "content": content}
        try:
            with open(self.dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.error(f"[Dataset Log Error] {e}")

    # ---------------- Interaction handling ----------------
    def save_interaction(self, role: str, content: str, **kwargs):
        timestamp = datetime.now().isoformat()
        interaction = {"role": role, "content": content, "time": timestamp}

        # Add any extra metadata (relationship, tags, mood, etc.)
        if kwargs:
            interaction.update(kwargs)

        self.data["interactions"].append(interaction)
        self.save_memory()

        # Log to dataset
        self.log_dataset_entry(role, content)


    def search_interactions(self, query: str, top_k: int = 5):
        if not self.data["interactions"]:
            return []
        try:
            embeddings = [self.model.encode(i["content"], convert_to_tensor=True)
                          for i in self.data["interactions"]]
            query_emb = self.model.encode(query, convert_to_tensor=True)
            scores = [util.cos_sim(query_emb, emb).item() for emb in embeddings]
            ranked = sorted(zip(scores, self.data["interactions"]), key=lambda x: x[0], reverse=True)
            return [i["content"] for _, i in ranked[:top_k]]
        except Exception as e:
            logging.error(f"[Search Error] {e}")
            return []

    # ---------------- Facts, Traits, Reminders ----------------
    def remember_fact(self, key: str, value: str):
        self.data["facts"][key] = value
        self.save_memory()

    def recall_fact(self, key: str):
        return self.data["facts"].get(key)

    def get_facts(self) -> dict:
        return dict(self.data.get("facts", {}))

    def update_trait(self, key: str, value: str):
        self.data["traits"][key] = value
        self.save_memory()

    def get_traits(self) -> dict:
        return dict(self.data.get("traits", {}))

    def add_reminder(self, message: str, time: str):
        self.data["reminders"].append({"message": message, "time": time})
        self.save_memory()

    def get_reminders(self):
        return list(self.data.get("reminders", []))

    # ---------------- Preferences & Relationships ----------------
    def get_preferences(self) -> dict:
        return dict(self.data.get("preferences", {})) if "preferences" in self.data else {}

    def set_relationship(self, person: str, relation: str):
        self.data["relationships"][person] = relation
        self.save_memory()

    def get_relationships(self):
        return dict(self.data.get("relationships", {}))

    # ---------------- User name ----------------
    def set_user_name(self, name: str):
        self.data["user_name"] = name
        self.save_memory()

    def get_user_name(self):
        return self.data.get("user_name", self.user_id)
    def retrieve_context(self, query: str = "", top_k: int = 5) -> str:
        """
        Retrieve a combined context string from memory: facts, traits, reminders, and recent interactions.
        Optionally perform a semantic search on past interactions with a query.
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

