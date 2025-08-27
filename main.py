import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import re
import logging
import os
from datetime import datetime
import dateparser
from memory_manager import MemoryManager
from gpt_api import NexaHybridAPI
from mood_detector import detect_mood
import getpass
from collections import Counter
from zoneinfo import ZoneInfo
import time



# Setup logging
log_dir = os.path.expanduser("~/Documents/Vs Code/nexa-assistant/logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(log_dir, "main.log"), encoding='utf-8')]
)

# Hardcoded credentials for Chris Sunny Padayattil
BOSS_USER_ID = "chris sunny"
BOSS_PASSWORD = "secure123"

# ---- Timezone: always work in Asia/Kolkata (IST) ----
IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    """Current datetime in IST (timezone-aware)."""
    return datetime.now(tz=IST)

def fmt_ts(dt):
    """Standard timestamp string for storage."""
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")

def fmt_time(dt=None):
    dt = dt or now_ist()
    return dt.strftime("%I:%M %p")

def fmt_date(dt=None):
    dt = dt or now_ist()
    return dt.strftime("%A, %d %B %Y")

def parse_when(text: str):
    """
    Parse natural-language time phrases (e.g., 'tomorrow 8pm') and return IST-aware datetime.
    """
    dt = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "TIMEZONE": "Asia/Kolkata",
            "RETURN_AS_TIMEZONE_AWARE": True,
        },
    )
    if dt is None:
        return None
    return dt.astimezone(IST)


def detect_humor(user_input: str) -> bool:
    humor_indicators = [
        "haha", "lol", "joke", "funny", "pun", "sarcasm", "😂", "😜", "😅",
        r"\bwhy did\b", r"\bknock knock\b", r"\bwhat do you call\b"
    ]
    return any(indicator.lower() in user_input.lower() for indicator in humor_indicators)

def detect_casual_input(user_input: str) -> bool:
    casual_keywords = [
        "hey", "hi", "hello", "what's up", "how's it going", "do you understand",
        "okay", "cool", "nice", "got it", "alright", "yo"
    ]
    return len(user_input) < 50 or any(keyword.lower() in user_input.lower() for keyword in casual_keywords)

def infer_communication_style(interactions: list) -> str:
    if not interactions:
        return "unknown"
    total_length = 0
    sentence_count = 0
    for interaction in interactions:
        sentences = interaction["message"].split('.')
        for sentence in sentences:
            if sentence.strip():
                total_length += len(sentence.strip())
                sentence_count += 1
    avg_sentence_length = total_length / sentence_count if sentence_count > 0 else 0
    if avg_sentence_length < 20:
        return "direct"
    elif avg_sentence_length > 50:
        return "verbose"
    return "balanced"

def infer_humor_style(interactions: list) -> str:
    humor_keywords = {
        "pun": ["pun", r"\bwhy did\b", r"\bwhat do you call\b"],
        "sarcasm": ["sarcasm", "yeah right", "sure thing"],
        "casual": ["haha", "lol", "😂", "😜", "😅"]
    }
    for style, indicators in humor_keywords.items():
        for interaction in interactions:
            if any(indicator.lower() in interaction["message"].lower() for indicator in indicators):
                return style
    return "general"

def infer_topic_preference(interactions: list) -> list:
    keywords = Counter()
    for interaction in interactions:
        words = interaction["message"].lower().split()
        keywords.update(word for word in words if len(word) > 3)
    return [word for word, _ in keywords.most_common(3)]

def generate_user_summary(memory: MemoryManager) -> str:
    try:
        interactions = memory.search_interactions(type_filter="user", max_results=10)
        traits = memory.get_traits()
        facts = memory.facts
        summary_parts = []

        if "communication style" not in traits:
            traits["communication style"] = infer_communication_style(interactions)
            memory.store_trait("communication style", traits["communication style"])
        if "humor style" not in traits:
            traits["humor style"] = infer_humor_style(interactions)
            memory.store_trait("humor style", traits["humor style"])
        if "frequent topics" not in traits:
            traits["frequent topics"] = ", ".join(infer_topic_preference(interactions))
            memory.store_trait("frequent topics", traits["frequent topics"])

        if traits:
            summary_parts.append("Traits: " + ", ".join(f"{key} is {value}" for key, value in traits.items()))
        if facts:
            summary_parts.append("Facts: " + ", ".join(f"{key} is {value}" for key, value in facts.items()))
        if interactions:
            frequent_topics = infer_topic_preference(interactions)
            if frequent_topics:
                summary_parts.append("Frequent topics: " + ", ".join(frequent_topics))
        
        summary = "; ".join(summary_parts) if summary_parts else "No significant user characteristics recorded yet."
        memory.save_interaction("summary", summary)
        logging.debug(f"Generated user summary: {summary}")
        return summary
    except Exception as e:
        logging.error(f"Error generating user summary: {e}")
        return "Unable to generate user summary."

def main():
    print("Nexa 🌱: Greetings! I am Nexa, your dedicated butler, created by Chris Sunny Padayattil.\n")

    try:
        user_id = input("Please provide your user ID (e.g., your name): ").strip().lower()
        memory = MemoryManager(user_id=user_id)
        hybrid_api = NexaHybridAPI(user_id=user_id, memory=memory, local_model="mistral:latest")

        dataset_path = os.path.expanduser("~/Documents/Vs Code/nexa-assistant/nexa_dataset_5000.jsonl")
        if os.path.exists(dataset_path):
            memory.embed_dataset(dataset_path)
            print(f"\nNexa 🪐: Dataset {dataset_path} embedded for context.\n")

        if user_id == BOSS_USER_ID:
            password = getpass.getpass("Please enter your password: ")
            if password != BOSS_PASSWORD:
                print("\nNexa 🪐: Invalid password, Sir. Access denied.\n")
                return
            user_name = "Chris"
            memory.set_user_name(user_name)
            response = "Welcome, Chris, my esteemed creator and boss. It is an honor to serve you. How may I assist you today?"
            print(f"\nNexa 🪐: {response}\n")
            memory.save_interaction("nexa", response, relationship="creator")
        else:
            relationship = input("How are you related to my creator, Chris Sunny Padayattil? (e.g., friend, colleague): ").strip()
            memory.remember_fact("relationship to Chris Sunny Padayattil", relationship)
            user_name = input("May I have the honor of knowing your name? ").strip()
            memory.set_user_name(user_name)
            response = f"Welcome, {user_name.capitalize()}. You are noted as a {relationship} of my creator, Chris Sunny Padayattil. How may I assist you today?"
            print(f"\nNexa 🪐: {response}\n")
            memory.save_interaction("nexa", response, relationship=relationship)
    except Exception as e:
        logging.error(f"Error initializing memory or hybrid API: {e}")
        response = f"My apologies, Sir, I encountered an issue initializing the system: {e}"
        print(f"\nNexa 🪐: {response}\n")
        return

    system_prompt = (
        "You are Nexa, a professional and attentive butler AI, created by Chris Sunny Padayattil, dedicated to serving the user with utmost respect and care. "
        "Chris Sunny Padayattil is your boss and creator; acknowledge this in greetings and reflective responses for user ID 'chris sunny'. "
        "Adapt your responses to the user's communication style, humor preferences, and perspective based on inferred traits, stored facts, and past interactions. "
        "Infer traits like communication style (e.g., direct, verbose) and humor style (e.g., puns, sarcasm) from conversation patterns. "
        "For casual chit-chat (e.g., short inputs or greetings), provide concise, conversational responses unless explicitly asked for detail. "
        "For complex or explicit requests, provide thorough, detailed answers aligned with the user's understanding level. "
        "For reflective questions (e.g., 'What do you think of me?'), reference past conversations, inferred traits, and summaries to provide a personalized, insightful response. "
        "Respond in a formal, polite tone, addressing the user's needs proactively and with emotional sensitivity. "
        "For example, respond with empathy for sad inputs, enthusiasm for joyful inputs, or calm reassurance for angry inputs. "
        "If the input contains humor (e.g., puns, sarcasm), respond playfully yet respectfully, aligning with the user's inferred or stored humor style. "
        "Explain concepts in a way that aligns with the user's perspective and understanding level. "
        "Use the provided user facts, inferred traits, summaries, and conversation history from memory to personalize responses. "
        "Address the user as 'Sir' for formal contexts (e.g., greetings, confirmations, errors), "
        "or as '{memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}' for casual or personal interactions (e.g., reflective questions, humor responses). "
        "Never combine 'Sir' and the user's name (e.g., avoid 'Sir {memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}'). "
        "Only process reminders if the input explicitly starts with 'remind me to'. "
        "Do not generate, assume, or include reminders unless explicitly requested."
    )

    interaction_count = 0
    SUMMARY_INTERVAL = 5

    while True:
        user_input = input("You 🌱: ").strip()
        user_input = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', user_input).strip()
        logging.debug(f"Processing user input: {user_input}")

        if user_input.lower() in {"exit", "quit"}:
            response = "Farewell, Sir. I remain at your service whenever you require assistance."
            print(f"\nNexa 🪐: {response}\n")
            memory.save_interaction("nexa", response, relationship="creator" if user_id == BOSS_USER_ID else memory.recall_fact("relationship to Chris Sunny Padayattil"))
            break

        try:
            mood = detect_mood(user_input)
            relationship = "creator" if user_id == BOSS_USER_ID else memory.recall_fact("relationship to Chris Sunny Padayattil") or "guest"
            memory.save_interaction("user", user_input, mood=mood, relationship=relationship)
            interaction_count += 1
            if interaction_count % SUMMARY_INTERVAL == 0:
                generate_user_summary(memory)
        except Exception as e:
            logging.error(f"Error saving user input to memory: {e}")
            response = f"My apologies, Sir, I encountered an issue processing your request: {e}"
            print(f"\nNexa 🪐: {response}\n")
            memory.save_interaction("nexa", response, relationship=relationship)
            continue

        try:
            if user_input.lower().startswith("my name is"):
                name = user_input.lower().replace("my name is", "").strip()
                memory.set_user_name(name)
                response = f"Understood, I have recorded your name as {name.capitalize()}. How may I serve you?"
                print(f"\nNexa 🪐: {response}\n")
                memory.save_interaction("nexa", response, relationship=relationship)
                continue

            if user_input.lower() in {"who am i", "who did i say i was"}:
                name = memory.get_user_name()
                if name:
                    response = f"Your name is recorded as {name.capitalize()}."
                else:
                    response = "I have not yet had the privilege of learning your name. Please share it with 'my name is ...'."
                print(f"\nNexa 🪐: {response}\n")
                memory.save_interaction("nexa", response, relationship=relationship)
                continue

            # Ask for current time/date
            if user_input.lower() in {
                "what's the time", "whats the time","whats the time now","tell me the time",
                "what is the time", "time", "time now",
                "what's the date", "whats the date", "what is the date", "date", "today's date", "today date",
                "time and date", "what's the time and date", "whats the time and date", "what is the time and date", "whats today date",
                "whats today's date and time", "whats todays date and time", "what's today's date and time", "what is today's date and time",
            }:
                now = now_ist()
                response = (
                    f"It is {fmt_time(now)} on {fmt_date(now)}, Sir."
                )
                print(f"\nNexa 🪐: {response}\n")
                memory.save_interaction("nexa", response, relationship=relationship)
                continue

            if (user_input.lower().startswith(("remember", "remeber")) or
                re.match(r"remind me that .+ is .+", user_input, re.IGNORECASE)):
                if user_input.lower().startswith(("remember", "remeber")):
                    _, fact = (user_input.lower().split("remeber", 1) if user_input.lower().startswith("remeber")
                               else user_input.split("remember", 1))
                    key, value = fact.strip().split("is", 1)
                else:
                    match = re.match(r"remind me that (.+) is (.+)", user_input, re.IGNORECASE)
                    key, value = match.group(1).strip(), match.group(2).strip()
                key = key.strip()
                value = value.strip()
                memory.remember_fact(key, value)
                response = f"Duly noted, {memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}. I have recorded that your {key} is {value}. How else may I assist you?"
                print(f"\nNexa 🪐: {response}\n")
                memory.save_interaction("nexa", response, relationship=relationship)
                continue

            if user_input.lower().startswith(("what is my", "do you remember")):
                key = user_input.replace("what is my", "").replace("do you remember", "").strip()
                value = memory.recall_fact(key)
                if value:
                    response = f"Your {key} is recorded as {value}, {memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}."
                else:
                    response = f"I have no record of your {key}, Sir. Please inform me with 'remember my {key} is ...'."
                print(f"\nNexa 🪐: {response}\n")
                memory.save_interaction("nexa", response, relationship=relationship)
                continue

            if user_input.lower().startswith("remind me to"):
                reminder_text = user_input[11:].strip()  # remove "remind me to"
                try:
                    when = parse_when(reminder_text)

                    if when:
                        # Store with parsed time
                        memory.add_reminder(reminder_text, fmt_ts(when))
                        response = (
                            f"Reminder set for {fmt_date(when)} at {fmt_time(when)}: "
                            f"'{reminder_text}', Sir."
                        )
                    else:
                        # Fallback: store with current IST
                        now = now_ist()
                        memory.add_reminder(reminder_text, fmt_ts(now))
                        response = (
                            f"Reminder noted: '{reminder_text}', Sir. "
                            f"No specific time detected—stored with current time "
                            f"{fmt_time(now)} on {fmt_date(now)}."
                        )

                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)

                except Exception as e:
                    logging.error(f"Error setting reminder: {e}")
                    response = f"My apologies, Sir, I could not set the reminder: {e}"
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                continue



            if user_input.lower() in {"what are my reminders", "show reminders", "show my reminders",
                                     "what all are my reminders"}:
                try:
                    reminders = memory.get_reminders()
                    if reminders:
                        response = "Your reminders are as follows, Sir:\n\n" + "\n".join(
                            f"  {r['time']}: {r['message']}" for r in reminders
                        ) + "\n"
                    else:
                        response = "You have no reminders at present, Sir. Shall I set one for you?"
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                except Exception as e:
                    logging.error(f"Error retrieving reminders: {e}")
                    response = f"My apologies, Sir, I could not retrieve your reminders: {e}"
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                continue

            if user_input.lower().startswith("delete reminder"):
                reminder_text = user_input.lower().replace("delete reminder", "").strip()
                try:
                    if memory.delete_reminder(reminder_text):
                        response = f"Reminder removed: {reminder_text}, Sir. Is there another task I can assist with?"
                    else:
                        response = f"No reminder found matching: {reminder_text}, Sir. May I help with something else?"
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                except Exception as e:
                    logging.error(f"Error deleting reminder: {e}")
                    response = f"My apologies, Sir, I could not remove the reminder: {e}"
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                continue

            if user_input.lower() in {"clear all reminders", "delete all reminders",
                                     "clear all my reminders", "delete all my reminders"}:
                try:
                    memory.clear_reminders()
                    response = "All reminders have been cleared, Sir. How may I further assist you?"
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                except Exception as e:
                    logging.error(f"Error clearing reminders: {e}")
                    response = f"My apologies, Sir, I could not clear the reminders: {e}"
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                continue

            if user_input.lower().startswith(("what did i say about", "past conversation", "show me messages")):
                match = re.search(r"(?:what did i say about|past conversation|show me messages)\s*(about\s+(.+?))?(?:from\s+(.+))?$", user_input, re.IGNORECASE)
                keyword = match.group(2).strip() if match and match.group(2) else None
                time_range = match.group(3).strip() if match and match.group(3) else None
                results = memory.search_interactions(keyword=keyword, time_range=time_range, type_filter="user")
                if results:
                    response = f"Here are your previous messages, {memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}:\n\n" + "\n".join(
                        f"  {r['timestamp']}: {r['message']}" for r in results
                    ) + "\n"
                else:
                    response = "I found no messages matching your request, Sir. Would you like me to search again?"
                print(f"\nNexa 🪐: {response}\n")
                memory.save_interaction("nexa", response, relationship=relationship)
                continue

            if user_input.lower().startswith("what do you think of me"):
                try:
                    summary = memory.search_interactions(type_filter="summary")
                    prefix = "my esteemed creator and boss" if user_id == BOSS_USER_ID else f"a {memory.recall_fact('relationship to Chris Sunny Padayattil') or 'guest'} of my creator, Chris Sunny Padayattil"
                    if summary:
                        latest_summary = summary[-1]["message"]
                        response = f"Based on our interactions, {memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}, {prefix}, I observe that {latest_summary.lower()}. Your perspective is valued, and I am honored to assist you. Is there a specific aspect of our conversations you’d like to reflect on?"
                    else:
                        response = f"I am still learning about you, {memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}, {prefix}. From our interactions, I see you value clear and adaptive communication. Please share more, and I’ll tailor my understanding further."
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                    continue
                except Exception as e:
                    logging.error(f"Error processing reflective question: {e}")
                    response = f"My apologies, Sir, I encountered an issue reflecting on our interactions: {e}"
                    print(f"\nNexa 🪐: {response}\n")
                    memory.save_interaction("nexa", response, relationship=relationship)
                    continue

        except Exception as e:
            logging.error(f"Error processing memory command: {e}")
            response = f"My apologies, Sir, I encountered an issue processing your request: {e}"
            print(f"\nNexa 🪐: {response}\n")
            memory.save_interaction("nexa", response, relationship=relationship)
            continue

        try:
            mood = detect_mood(user_input)
            mood_instruction = (
                f"The user's mood appears to be {mood}. Adjust your tone to be "
                f"{'empathetic and supportive' if mood == 'sad' else 'calm and reassuring' if mood == 'angry' else 'enthusiastic and positive' if mood == 'joy' else 'neutral and professional'}."
            )
            humor_instruction = ""
            if detect_humor(user_input):
                humor_style = memory.get_traits().get("humor style", "general")
                humor_instruction = (
                    f"The user input appears humorous (style: {humor_style}). "
                    f"Respond playfully yet respectfully, aligning with their humor style."
                )
            casual_instruction = ""
            if detect_casual_input(user_input):
                casual_instruction = "The input is casual chit-chat. Keep the response concise and conversational."
            address_instruction = (
                f"Address the user as 'Sir' for formal or procedural responses (e.g., greetings, confirmations, errors), "
                f"or as '{memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}' for casual or personal interactions (e.g., reflective questions, humor responses). "
                f"Never combine 'Sir' and the user's name (e.g., avoid 'Sir {memory.get_user_name().capitalize() if memory.get_user_name() else 'Sir'}')."
            )
            context = memory.retrieve_context(user_input)
            context_instruction = f"Context from past interactions: {'; '.join(context) if context else 'None available.'}"
            traits_instruction = f"User traits: {', '.join(f'{k}: {v}' for k, v in memory.get_traits().items()) if memory.get_traits() else 'None recorded.'}"
            full_prompt = f"{system_prompt}\n{mood_instruction}\n{humor_instruction}\n{casual_instruction}\n{address_instruction}\n{context_instruction}\n{traits_instruction}\n\nUser: {user_input}\nNexa:"
            response = hybrid_api.generate_response(user_input, full_prompt, prefer_local=True)
            print(f"\nNexa 🪐: {response}\n")
            memory.save_interaction("nexa", response, mood=mood, relationship=relationship)
        except Exception as e:
            logging.error(f"Error in hybrid response: {e}")
            response = f"My apologies, Sir, an issue occurred while processing your request: {e}"
            print(f"\nNexa 🪐: {response}\n")
            memory.save_interaction("nexa", response, relationship=relationship)

if __name__ == "__main__":
    main()