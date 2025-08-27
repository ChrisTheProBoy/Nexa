import os
import csv
import logging
from datetime import datetime
import dateparser

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class NexaMemory:
    def __init__(self, user_id, base_path="/home/chris_sunny/Documents/Vs Code"):
        """
        Initialize memory system for a user, using CSV for persistent storage.
        
        Args:
            user_id: Unique identifier for the user (e.g., name or ID).
            base_path: Directory to store the CSV file.
        """
        self.user_id = user_id
        self.base_path = base_path
        self.memory_file = os.path.join(base_path, f"nexa_memory_{user_id}.csv")
        self._ensure_directory_exists()
        self._initialize_memory_file()

    def _ensure_directory_exists(self):
        """Ensure the directory for the memory file exists and is writable."""
        try:
            os.makedirs(self.base_path, exist_ok=True)
            logging.debug(f"Ensured directory exists: {self.base_path}")
            # Test write permissions
            test_file = os.path.join(self.base_path, ".write_test")
            with open(test_file, "w") as f:
                f.write("")
            os.remove(test_file)
            logging.debug(f"Directory {self.base_path} is writable")
        except Exception as e:
            logging.error(f"Failed to create or verify directory {self.base_path}: {e}")
            raise

    def _initialize_memory_file(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        try:
            if not os.path.exists(self.memory_file):
                with open(self.memory_file, "w", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "type", "message", "embedding"])
                logging.debug(f"Created CSV file with headers: {self.memory_file}")
            logging.info(f"Initialized memory for user_id: {self.user_id}")
        except Exception as e:
            logging.error(f"Failed to initialize memory file {self.memory_file}: {e}")
            raise

    def save_interaction(self, interaction_type: str, message: str):
        """Save an interaction to the CSV file."""
        try:
            if not message.strip():
                logging.warning(f"Attempted to save empty message for type={interaction_type}")
                return
            if interaction_type == "trait" and " is " not in message:
                logging.warning(f"Invalid trait format: {message}. Expected 'key is value'.")
                return
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.memory_file, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, interaction_type, message, ""])
            logging.debug(f"Saved interaction: type={interaction_type}, message={message}")
        except Exception as e:
            logging.error(f"Error saving interaction: {e}")
            raise

    def set_user_name(self, name: str):
        """Set the user's name and save it to memory."""
        try:
            self.save_interaction("user_name", name)
            logging.info(f"Set user name: {name}")
        except Exception as e:
            logging.error(f"Error setting user name: {e}")
            raise

    def get_user_name(self) -> str:
        """Retrieve the user's name from memory."""
        try:
            with open(self.memory_file, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reversed(list(reader)):
                    if row["type"] == "user_name":
                        return row["message"]
            return ""
        except Exception as e:
            logging.error(f"Error getting user name: {e}")
            raise

    def get_facts(self) -> dict:
        """Retrieve all stored facts about the user."""
        try:
            facts = {}
            with open(self.memory_file, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["type"] == "fact":
                        try:
                            key, value = row["message"].split(" is ", 1)
                            facts[key.strip()] = value.strip()
                        except ValueError:
                            logging.warning(f"Invalid fact format: {row['message']}")
            logging.debug(f"Retrieved facts: {facts}")
            return facts
        except Exception as e:
            logging.error(f"Error getting facts: {e}")
            return {}

    def get_preferences(self) -> dict:
        """Retrieve all stored preferences for the user."""
        try:
            preferences = {}
            with open(self.memory_file, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["type"] == "preference":
                        try:
                            key, value = row["message"].split(" is ", 1)
                            preferences[key.strip()] = value.strip()
                        except ValueError:
                            logging.warning(f"Invalid preference format: {row['message']}")
            logging.debug(f"Retrieved preferences: {preferences}")
            return preferences
        except Exception as e:
            logging.error(f"Error getting preferences: {e}")
            return {}

    def get_traits(self) -> dict:
        """Retrieve all stored user traits (e.g., communication style, humor preferences)."""
        try:
            traits = {}
            with open(self.memory_file, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["type"] == "trait":
                        try:
                            key, value = row["message"].split(" is ", 1)
                            traits[key.strip()] = value.strip()
                        except ValueError:
                            logging.warning(f"Invalid trait format: {row['message']}")
            logging.debug(f"Retrieved traits: {traits}")
            return traits
        except Exception as e:
            logging.error(f"Error getting traits: {e}")
            return {}

    def store_trait(self, key: str, value: str):
        """Store a user trait (e.g., communication style, humor preference)."""
        try:
            self.save_interaction("trait", f"{key} is {value}")
            logging.info(f"Stored trait: {key} = {value}")
        except Exception as e:
            logging.error(f"Error storing trait: {e}")
            raise

    def remember_fact(self, key: str, value: str):
        """Store a fact about the user."""
        try:
            self.save_interaction("fact", f"{key} is {value}")
            logging.info(f"Stored fact: {key} = {value}")
        except Exception as e:
            logging.error(f"Error storing fact: {e}")
            raise

    def recall_fact(self, key: str) -> str:
        """Recall a fact about the user by key."""
        try:
            facts = self.get_facts()
            return facts.get(key, "")
        except Exception as e:
            logging.error(f"Error recalling fact: {e}")
            return ""

    def add_reminder(self, message: str, time: str):
        """Add a reminder with a specified time."""
        try:
            self.save_interaction("reminder", f"{message} at {time}")
            logging.info(f"Added reminder: {message} at {time}")
        except Exception as e:
            logging.error(f"Error adding reminder: {e}")
            raise

    def get_reminders(self) -> list:
        """Retrieve all reminders, ensuring only valid reminder entries are included."""
        try:
            reminders = []
            with open(self.memory_file, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["type"] == "reminder":
                        try:
                            if " at " in row["message"]:
                                message, time = row["message"].rsplit(" at ", 1)
                                if dateparser.parse(time):
                                    reminders.append({"message": message.strip(), "time": time.strip()})
                                else:
                                    logging.warning(f"Invalid time format in reminder: {row['message']}")
                            else:
                                logging.warning(f"Invalid reminder format: {row['message']}")
                        except ValueError:
                            logging.warning(f"Invalid reminder format: {row['message']}")
            logging.debug(f"Retrieved reminders: {reminders}")
            return reminders
        except Exception as e:
            logging.error(f"Error getting reminders: {e}")
            return []

    def delete_reminder(self, keyword: str) -> bool:
        """Delete reminders containing the keyword."""
        try:
            with open(self.memory_file, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            new_rows = [row for row in rows if not (row["type"] == "reminder" and keyword.lower() in row["message"].lower())]
            if len(new_rows) < len(rows):
                with open(self.memory_file, "w", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "type", "message", "embedding"])
                    for row in new_rows:
                        writer.writerow([row["timestamp"], row["type"], row["message"], row["embedding"]])
                logging.info(f"Deleted reminder(s) containing: {keyword}")
                return True
            logging.debug(f"No reminders found with keyword: {keyword}")
            return False
        except Exception as e:
            logging.error(f"Error deleting reminder: {e}")
            return False

    def clear_reminders(self):
        """Clear all reminders."""
        try:
            with open(self.memory_file, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            new_rows = [row for row in rows if row["type"] != "reminder"]
            with open(self.memory_file, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "type", "message", "embedding"])
                for row in new_rows:
                    writer.writerow([row["timestamp"], row["type"], row["message"], row["embedding"]])
            logging.info("Cleared all reminders")
        except Exception as e:
            logging.error(f"Error clearing reminders: {e}")
            raise

    def search_interactions(self, keyword: str = None, time_range: str = None, type_filter: str = None) -> list:
        """Search interactions by keyword, time range, and optional type filter."""
        try:
            results = []
            with open(self.memory_file, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if type_filter and row["type"] != type_filter:
                        continue
                    if keyword and keyword.lower() not in row["message"].lower():
                        continue
                    if time_range:
                        parsed_time = dateparser.parse(time_range)
                        if parsed_time:
                            row_time = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                            if row_time < parsed_time:
                                continue
                    results.append(row)
            results = results[-10:]  # Limit to most recent 10 results
            logging.info(f"Search results for keyword={keyword}, time_range={time_range}, type_filter={type_filter}: {len(results)} matches")
            return results
        except Exception as e:
            logging.error(f"Error searching interactions: {e}")
            return []