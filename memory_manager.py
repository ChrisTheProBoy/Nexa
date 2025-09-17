import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

class MemoryManager:
    """
    Unified memory system with dataset logging + backward compatibility.
    Stores:
      - facts (dict)
      - traits (dict)
      - preferences (dict)
      - reminders (list[{message, time}])
      - interactions (list[{role, content, time, ...metadata}])
      - conflicts (list[{key, old, new, time, resolved}])
      - user_name (explicit key, synced with preferences["address_me_as"])
    """

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
            "conflicts": [],
            "user_name": None,
        }
        self.load_memory()

    # ----------------- Clarifications -----------------
    def set_clarification(self, key: str, value: str):
        self.data.setdefault("clarifications", {})
        self.data["clarifications"][key] = {
            "value": value,
            "time": datetime.now().isoformat()
        }
        self.save_memory()

    def get_clarification(self, key: str) -> str | None:
        clarifications = self.data.get("clarifications", {})
        if key in clarifications:
            return clarifications[key]["value"]
        return None

    def clear_clarifications(self):
        self.data["clarifications"] = {}
        self.save_memory()

    # ----------------- Persistence -----------------
    def load_memory(self):
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.data.update(json.load(f))
            except Exception as e:
                print(f"[MemoryManager] Failed to load memory: {e}")

    def save_memory(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[MemoryManager] Failed to save memory: {e}")

    # ----------------- Dataset logging -----------------
    def log_dataset_entry(self, role: str, content: str):
        """Append to dataset, filtering noise/error messages."""
        if not content or content.strip() in {"{", "}", ";"}:
            return
        lower = content.lower()
        if "traceback" in lower or "object has no attribute" in lower:
            return
        try:
            with open(self.dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[MemoryManager] Failed to log dataset entry: {e}")

    # ----------------- Interactions -----------------
    def save_interaction(self, role: str, content: str, log_to_dataset: bool = True, **kwargs):
        ts = datetime.now().isoformat()
        interaction = {"role": role, "content": content, "time": ts}
        if kwargs:
            interaction.update(kwargs)
        self.data["interactions"].append(interaction)
        self.save_memory()
        if log_to_dataset:
            self.log_dataset_entry(role, content)

    def search_interactions(
        self,
        keyword: Optional[str] = None,
        time_range: Optional[str] = None,
        type_filter: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict]:
        results = []
        for i in reversed(self.data.get("interactions", [])):
            if type_filter and i.get("role") != type_filter:
                continue
            if keyword and keyword.lower() not in i.get("content", "").lower():
                continue
            results.append({
                "timestamp": i.get("time"),
                "message": i.get("content"),
                "role": i.get("role"),
            })
            if len(results) >= max_results:
                break
        return results

    # ----------------- Facts / Traits / Prefs -----------------
    def remember_fact(self, key: str, value: str):
        old_value = self.data["facts"].get(key)
        if old_value and old_value != value:
            # conflict detected
            self.track_conflict(key, old_value, value)
        self.data["facts"][key] = value
        self.save_memory()

    def recall_fact(self, key: str) -> Optional[str]:
        return self.data.get("facts", {}).get(key)

    @property
    def facts(self) -> dict:
        return self.data.get("facts", {})

    def update_trait(self, key: str, value: str):
        self.data["traits"][key] = value
        self.save_memory()

    def get_traits(self) -> dict:
        return self.data.get("traits", {})

    def update_preference(self, key: str, value):
        self.data.setdefault("preferences", {})
        self.data["preferences"][key] = value
        self.save_memory()

    def get_preferences(self) -> dict:
        return self.data.get("preferences", {})

    # ----------------- User Name -----------------
    def set_user_name(self, name: str):
        self.data["user_name"] = name
        self.update_preference("address_me_as", name)
        self.save_memory()

    def get_user_name(self) -> str:
        return self.data.get("user_name") or self.get_preferences().get("address_me_as", "User")

    # ----------------- Reminders -----------------
    def add_reminder(self, message: str, time_str: str):
        self.data["reminders"].append({"message": message, "time": time_str})
        try:
            self.data["reminders"].sort(
                key=lambda r: datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S")
            )
        except Exception:
            pass
        self.save_memory()

    def get_reminders(self) -> List[Dict]:
        return list(self.data.get("reminders", []))

    def save_reminders(self, reminders: List[Dict]):
        self.data["reminders"] = reminders
        self.save_memory()

    def get_upcoming_reminders(self, now_iso: Optional[str] = None) -> List[Dict]:
        now = datetime.fromisoformat(now_iso) if now_iso else datetime.now()
        out = []
        for r in self.get_reminders():
            t = r.get("time")
            try:
                dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                if dt >= now:
                    out.append(r)
            except Exception:
                out.append(r)
        return out

    def delete_reminder(self, contains_text: str) -> bool:
        before = len(self.data.get("reminders", []))
        self.data["reminders"] = [
            r for r in self.data.get("reminders", [])
            if contains_text.lower() not in r.get("message", "").lower()
        ]
        self.save_memory()
        return len(self.data["reminders"]) < before

    def clear_reminders(self):
        self.data["reminders"] = []
        self.save_memory()

    # ----------------- Conflict Handling -----------------
    def track_conflict(self, key: str, old_value: str, new_value: str):
        self.data.setdefault("conflicts", [])
        self.data["conflicts"].append({
            "key": key,
            "old": old_value,
            "new": new_value,
            "time": datetime.now().isoformat(),
            "resolved": False
        })
        self.save_memory()

    def get_unresolved_conflicts(self):
        return [c for c in self.data.get("conflicts", []) if not c.get("resolved")]

    def resolve_conflict(self, idx: int, decision: str):
        conflicts = self.data.get("conflicts", [])
        if 0 <= idx < len(conflicts):
            conflict = conflicts[idx]
            if decision == "old":
                self.data["facts"][conflict["key"]] = conflict["old"]
            elif decision == "new":
                self.data["facts"][conflict["key"]] = conflict["new"]
            conflict["resolved"] = True
            conflict["resolution"] = decision
            self.save_memory()
            return True
        return False

    # ----------------- Context Retrieval -----------------
    def retrieve_context(self, query: str = "", top_k: int = 5) -> str:
        parts = []
        if self.facts:
            parts.append(f"Facts: {self.facts}")
        traits = self.get_traits()
        if traits:
            parts.append(f"Traits: {traits}")
        prefs = self.get_preferences()
        if prefs:
            parts.append(f"Preferences: {prefs}")
        reminders = self.get_upcoming_reminders()
        if reminders:
            parts.append(f"Reminders: {reminders}")

        if query:
            hits = [i["message"] for i in self.search_interactions(keyword=query, max_results=top_k)]
        else:
            hits = [i.get("content") for i in self.data.get("interactions", [])[-top_k:]]
        if hits:
            parts.append("Recent interactions: " + " | ".join(hits))
        return "\n".join(parts) if parts else "No context available."

    # ----------------- Dataset Embedding Stub -----------------
    def embed_dataset(self, dataset_path: str):
        try:
            Path(dataset_path).exists()
        except Exception:
            pass
