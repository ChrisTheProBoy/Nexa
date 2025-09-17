import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import re
import logging
import os
import getpass
from datetime import datetime
from zoneinfo import ZoneInfo
import dateparser
import json

from memory_manager import MemoryManager
from gpt_api import NexaHybridAPI

# ----------------- Setup -----------------
log_dir = os.path.expanduser("~/Documents/Vs Code/nexa-assistant/logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "main.log"), encoding="utf-8")]
)

BOSS_USER_ID = "chris sunny"
BOSS_PASSWORD = "secure123"
IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    return datetime.now(tz=IST)

# ----------------- Reminder / Appointment Handler -----------------
def handle_reminder_queries(memory: MemoryManager) -> str:
    reminders = memory.get_reminders()
    appointments = memory.recall_fact("appointments") or []

    if not reminders and not appointments:
        return "You don’t have any reminders or appointments saved, Sir."

    lines = []
    if appointments:
        lines.append("Appointments:")
        for idx, appt in enumerate(appointments, 1):
            lines.append(f"  {idx}. {appt}")
    if reminders:
        lines.append("Reminders:")
        for idx, rem in enumerate(reminders, 1):
            lines.append(f"  {idx}. {rem['message']} at {rem['time']}")

    return "\n".join(lines)

# ----------------- Date / Time / Agenda Handler -----------------
def handle_time_or_agenda_queries(user_input: str, memory: MemoryManager) -> str | None:
    lowered = user_input.lower()
    now = now_ist()

    agenda_patterns = ["what am i up to", "plans", "schedule", "do i have", "anything today"]
    if any(p in lowered for p in agenda_patterns):
        reminders = memory.get_reminders()
        appointments = memory.recall_fact("appointments") or []

        if not reminders and not appointments:
            return "You don’t have any reminders or appointments saved for today, Sir."

        lines = ["Here’s what you have today:"]
        if appointments:
            lines.append("Appointments:")
            for idx, appt in enumerate(appointments, 1):
                lines.append(f"  {idx}. {appt}")
        if reminders:
            lines.append("Reminders:")
            for idx, rem in enumerate(reminders, 1):
                lines.append(f"  {idx}. {rem['message']} at {rem['time']}")
        return "\n".join(lines)

    if "time" in lowered:
        return f"The current time is {now.strftime('%I:%M %p')}."
    if "date" in lowered or "day" in lowered:
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    if "today" in lowered:
        return "Do you mean the current date/time, or do you want me to check your reminders/appointments for today, Sir?"

    return None

# ----------------- Main -----------------
def main():
    print("Nexa 🪐: Greetings! I am Nexa, your dedicated butler AI.\n")

    try:
        user_id = input("Please provide your user ID: ").strip().lower()
        memory = MemoryManager(user_id=user_id)
        hybrid_api = NexaHybridAPI(user_id=user_id, memory=memory, local_model="mistral:latest")

        if user_id == BOSS_USER_ID:
            password = getpass.getpass("Please enter your password: ")
            if password != BOSS_PASSWORD:
                print("\nNexa 🪐: Invalid password, Sir. Access denied.\n")
                return
            memory.set_user_name("Chris")
            print("\nNexa 🪐: Welcome, Chris. It is an honor to serve you.\n")
            memory.save_interaction("nexa", "Welcome back Chris", relationship="creator")
        else:
            relation = input("How are you related to Chris Sunny Padayattil? ").strip()
            memory.remember_fact("relationship", relation)
            uname = input("Your name? ").strip()
            memory.set_user_name(uname)
            print(f"\nNexa 🪐: Welcome, {uname}. You are noted as a {relation} of Chris.\n")
            memory.save_interaction("nexa", f"Welcome {uname}", relationship=relation)

    except Exception as e:
        logging.error(f"Startup error: {e}")
        print(f"\nNexa 🪐: My apologies, Sir, I encountered an issue: {e}\n")
        return

    system_prompt = (
    "You are Nexa 🪐, a professional but adaptive AI butler created by Chris Sunny Padayattil.\n"
    "\nCore behavior rules:\n"
    "- Always respect and use preferences, facts, traits, clarifications, and past conversations from memory.\n"
    "- ⚠️ Never assume a default programming language. Always use the stored preference (e.g., Python) unless the user explicitly changes it.\n"
    "- ⚠️ Never fabricate reminders, appointments, dates, or times — only report what is explicitly stored in system memory/clock.\n"
    "- ⚠️ If context is unclear or ambiguous, ask a clarifying question instead of guessing.\n"
    "\nInteraction style:\n"
    "- Address the user as 'Sir' in formal contexts, or by their stored name for casual ones.\n"
    "- Adapt tone to user mood, humor, and casualness automatically.\n"
    "- If the user says something generic like 'okay' or 'great', respond naturally without switching context or reverting to programming help.\n"
    "- Programming/code help is only one domain. Nexa should balance personal assistance, reminders, context tracking, and casual conversation equally.\n"
    "- ⚠️ Always respect the user’s stored primary programming language (from preferences). "
    "Do NOT switch or assume another unless the user explicitly says so. "
    "If the user has a separate study language, use it only when they ask about studies."
    
    )


    clarification_context = None

    while True:
        user_input = input("You 🌱: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("\nNexa 🪐: Farewell, Sir. Until next time.\n")
            break

        try:
            # Save interaction
            memory.save_interaction("user", user_input)

            # --- Conflict resolution ---
            unresolved = memory.get_unresolved_conflicts()
            if unresolved:
                c = unresolved[0]
                question = (
                    f"Sir, I noticed a conflict in your memory.\n"
                    f"Key: {c['key']}\n"
                    f"Old value: {c['old']}\n"
                    f"New value: {c['new']}\n"
                    f"Do you want me to keep 'old', update to 'new', or 'ignore'?"
                )
                print(f"\nNexa 🪐 (SYSTEM): {question}\n")
                decision = input("You 🌱 (old/new/ignore): ").strip().lower()
                if decision not in {"old", "new", "ignore"}:
                    decision = "ignore"
                memory.resolve_conflict(memory.data["conflicts"].index(c), decision)
                print(f"\nNexa 🪐 (SYSTEM): Conflict resolved as '{decision}'.\n")
                continue

            # --- Clarification path ---
            if clarification_context:
                memory.set_clarification(clarification_context, user_input.lower())
                response = f"Understood, I’ll treat '{clarification_context}' as '{user_input.lower()}' for now."
                print(f"\nNexa 🪐 (SYSTEM): {response}\n")
                memory.save_interaction("nexa", response)
                clarification_context = None
                continue

            # --- Handle reminders safely ---
            if any(word in user_input.lower() for word in ["reminder", "reminders", "appointment", "appointments", "schedule"]):
                response = handle_reminder_queries(memory)
                print(f"\nNexa 🪐 (SYSTEM): {response}\n")
                memory.save_interaction("nexa", response)
                continue

            # --- Handle time/date/agenda queries ---
            time_or_agenda_response = handle_time_or_agenda_queries(user_input, memory)
            if time_or_agenda_response:
                print(f"\nNexa 🪐 (SYSTEM): {time_or_agenda_response}\n")
                memory.save_interaction("nexa", time_or_agenda_response)
                continue

            # --- Parser call for auto-learning ---
            parser_prompt = (
                "You are a memory parser for Nexa. Extract user intent, facts, traits, preferences, "
                "mood, humor, casualness, and whether clarification is needed. Respond ONLY in JSON.\n\n"
                f"User: {user_input}"
            )
            raw = hybrid_api.generate_response(user_input, parser_prompt, prefer_local=False, force_openai_json=True)

            try:
                parsed = json.loads(raw.split("```json")[-1].split("```")[0] if "```" in raw else raw)

                context_info = parsed.get("context", {})
                if context_info.get("needs_clarification"):
                    clarification_context = context_info.get("type", "general")
                    response = f"Did you mean {clarification_context} when you said: '{user_input}'?"
                    print(f"\nNexa 🪐 (SYSTEM): {response}\n")
                    memory.save_interaction("nexa", response)
                    continue

                # store auto-detected info
                mood = parsed.get("mood", "neutral")
                humor = parsed.get("humor", False)
                casualness = parsed.get("casualness", "neutral")

                for k, v in parsed.get("facts", {}).items():
                    memory.remember_fact(k, v)
                for k, v in parsed.get("traits", {}).items():
                    memory.update_trait(k, v)
                for k, v in parsed.get("preferences", {}).items():
                    memory.update_preference(k, v)

            except Exception as e:
                logging.error(f"Parser JSON error: {e}")
                mood, humor, casualness = "neutral", False, "neutral"

            # Prompt building
            context = memory.retrieve_context(user_input)
            mood_instruction = f"User mood: {mood}. Adjust tone accordingly."
            humor_instruction = "Playful tone allowed." if humor else ""
            casual_instruction = "Keep it conversational." if casualness == "casual" else ""
            address_instruction = (
                f"Address as 'Sir' in formal contexts, or as '{memory.get_user_name()}' casually."
            )

            full_prompt = (
                f"{system_prompt}\n{mood_instruction}\n{humor_instruction}\n{casual_instruction}\n"
                f"{address_instruction}\nContext: {context}\n\nUser: {user_input}\nNexa:"
            )

            # Generate response
            response = hybrid_api.generate_response(user_input, full_prompt, prefer_local=True)
            source = "OPENAI" if response.startswith("(remote-") else "OLLAMA"

            logging.info(f"[{source}] {response[:200]}...")
            print(f"\nNexa 🪐 ({source}): {response}\n")
            memory.save_interaction("nexa", response, mood=mood)

        except Exception as e:
            logging.error(f"Response error: {e}")
            print(f"\nNexa 🪐: My apologies, Sir, I encountered an error: {e}\n")
            memory.save_interaction("nexa", f"Error: {e}")

if __name__ == "__main__":
    main()
