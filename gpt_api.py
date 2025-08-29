import os
import logging
import openai
import re
from memory_manager import MemoryManager
from local_llm import generate_local_response

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

if "OPENAI_API_KEY" not in os.environ:
    logging.warning("OPENAI_API_KEY not found. GPT calls will fail unless set.")


def detect_and_store_preferences(memory: MemoryManager, user_input: str):
    """
    Detect preferences in user input and store them in memory.
    Includes specific rules + a generic fallback for 'Call me X' or 'Don't do Y'.
    """
    lowered = user_input.lower()

    # --- Specific Rules ---

    # Programming language preferences
    if "in python" in lowered:
        memory.update_preference("programming_language", "Python")
    elif "in c" in lowered:
        memory.update_preference("programming_language", "C")
    elif "in java" in lowered:
        memory.update_preference("programming_language", "Java")

    # Tone preferences
    if "be brief" in lowered or "short answer" in lowered:
        memory.update_preference("tone", "brief")
    elif "explain in detail" in lowered or "be detailed" in lowered:
        memory.update_preference("tone", "detailed")
    elif "be professional" in lowered:
        memory.update_preference("tone", "professional")
    elif "be friendly" in lowered:
        memory.update_preference("tone", "friendly")

    # Units preferences
    if "use metric" in lowered:
        memory.update_preference("units", "metric")
    elif "use imperial" in lowered:
        memory.update_preference("units", "imperial")

    # Emojis preferences
    if "with emojis" in lowered:
        memory.update_preference("emojis", "yes")
    elif "no emojis" in lowered or "without emojis" in lowered:
        memory.update_preference("emojis", "no")

    # --- Generic Rules ---

    # "Call me <X>" → address_me_as
    match_name = re.search(r"call me ([a-zA-Z0-9_ ]+)", lowered)
    if match_name:
        nickname = match_name.group(1).strip().title()
        memory.update_preference("address_me_as", nickname)

    # "Don't <do X>" → store as negative preference
    match_dont = re.search(r"don['’]t (.+)", lowered)
    if match_dont:
        action = match_dont.group(1).strip()
        memory.update_preference(f"avoid_{action.replace(' ', '_')}", True)

    # "Always <do X>" → store as positive preference
    match_always = re.search(r"always (.+)", lowered)
    if match_always:
        action = match_always.group(1).strip()
        memory.update_preference(f"always_{action.replace(' ', '_')}", True)

    # "I prefer <X>" → generic preference
    match_prefer = re.search(r"i prefer ([a-zA-Z0-9_ ]+)", lowered)
    if match_prefer:
        preference = match_prefer.group(1).strip()
        memory.update_preference("general_preference", preference)



class NexaHybridAPI:
    def __init__(self, user_id: str, memory: MemoryManager, local_model: str = "mistral:latest"):
        self.user_id = user_id
        self.memory = memory
        self.local_model = local_model

    def generate_openai_response(self, prompt: str, system_prompt: str = "") -> str:
        try:
            # Detect and save preferences
            detect_and_store_preferences(self.memory, prompt)

            context = self.memory.retrieve_context(query=prompt)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})

            # Inject preferences
            prefs = self.memory.get_preferences()
            if prefs:
                messages.append({"role": "system", "content": f"User preferences: {prefs}"})

            messages.append({"role": "user", "content": prompt})
            logging.debug(f"[OpenAI Request] Messages: {messages}")

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            answer = response.choices[0].message["content"]

            self.memory.save_interaction("user", prompt)
            self.memory.save_interaction("nexa", answer)
            return answer
        except Exception as e:
            error_msg = f"[OpenAI API Error] {str(e)}"
            logging.error(error_msg)
            self.memory.save_interaction("nexa", error_msg)
            return error_msg

    def generate_mistral_response(self, prompt: str, system_prompt: str = "") -> str:
        try:
            # Detect and save preferences
            detect_and_store_preferences(self.memory, prompt)

            context = self.memory.retrieve_context(query=prompt)
            if context:
                full_prompt = f"{system_prompt}\nContext: {context}\nUser: {prompt}"
            else:
                full_prompt = f"{system_prompt}\nUser: {prompt}"

            # Inject preferences
            prefs = self.memory.get_preferences()
            if prefs:
                full_prompt += f"\nUser preferences: {prefs}"

            logging.debug(f"[Mistral Request] {full_prompt}")
            answer = generate_local_response(full_prompt)

            self.memory.save_interaction("user", prompt)
            self.memory.save_interaction("nexa", answer)
            return answer
        except Exception as e:
            error_msg = f"[Mistral Error] {str(e)}"
            logging.error(error_msg)
            self.memory.save_interaction("nexa", error_msg)
            return error_msg

    def generate_response(self, prompt: str, system_prompt: str, prefer_local: bool = False) -> str:
        if prefer_local and self.local_model:
            response = self.generate_mistral_response(prompt, system_prompt)
            if len(response) > 10 and not response.startswith("[Mistral Error]"):
                return response
            logging.debug("Mistral response inadequate; falling back to OpenAI.")
        return self.generate_openai_response(prompt, system_prompt)


def generate_gpt_response(user_input: str, system_prompt: str = "") -> str:
    try:
        memory = MemoryManager(user_id="default-router")
        api = NexaHybridAPI(user_id="default-router", memory=memory, local_model="mistral:latest")
        return api.generate_openai_response(user_input, system_prompt)
    except Exception as e:
        logging.error(f"[Router GPT Wrapper] {e}")
        return f"[OpenAI API Error] {e}"
