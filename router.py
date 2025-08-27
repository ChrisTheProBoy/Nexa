import os
import logging
import openai
from memory_manager import MemoryManager
from local_llm import generate_local_response

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

# Ensure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    logging.warning("OPENAI_API_KEY not found in environment. GPT calls will fail unless set.")

class NexaHybridAPI:
    """
    Hybrid API that integrates both OpenAI GPT and local LLM (Mistral via Ollama).
    It uses a shared MemoryManager for persistence and personalization.
    """

    def __init__(self, user_id: str, memory: MemoryManager, local_model: str = "mistral:latest"):
        """
        Initialize the hybrid API.

        Args:
            user_id: The user's ID for memory access.
            memory: MemoryManager instance for shared storage and retrieval.
            local_model: Ollama model name (e.g., 'mistral:latest').
        """
        self.user_id = user_id
        self.memory = memory
        self.local_model = local_model

    def build_context(self) -> str:
        """Build conversation context from memory."""
        try:
            facts = self.memory.get_facts()
            traits = self.memory.get_traits()
            reminders = self.memory.get_reminders()
            preferences = self.memory.get_preferences()
            interactions = self.memory.search_interactions("")

            context_parts = []
            if facts:
                context_parts.append(f"User facts: {facts}")
            if traits:
                context_parts.append(f"Inferred traits: {traits}")
            if reminders:
                context_parts.append(f"Reminders: {reminders}")
            if preferences:
                context_parts.append(f"Preferences: {preferences}")
            if interactions:
                context_parts.append("Recent interactions: " + "; ".join(interactions[-5:]))

            return "\n".join(context_parts)
        except Exception as e:
            logging.error(f"[Context Build Error] {e}")
            return ""

    def generate_openai_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a response using OpenAI GPT, with context from memory.
        """
        try:
            context = self.build_context()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
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
        """
        Generate a response using local Mistral model via Ollama.
        """
        try:
            context = self.build_context()
            if context:
                full_prompt = f"{system_prompt}\nContext: {context}\nUser: {prompt}"
            else:
                full_prompt = f"{system_prompt}\nUser: {prompt}"

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
        """
        Generate response, choosing between Mistral and OpenAI, using shared memory.
        
        Args:
            prompt: The user's input prompt.
            system_prompt: The system prompt for the model.
            prefer_local: If True, try Mistral first.
        
        Returns:
            The generated response or an error message.
        """
        if prefer_local and self.local_model:
            response = self.generate_mistral_response(prompt, system_prompt)
            if len(response) > 10 and not response.startswith("[Mistral Error]"):
                return response
            logging.debug("Mistral response inadequate; falling back to OpenAI.")
        
        return self.generate_openai_response(prompt, system_prompt)


def generate_gpt_response(user_input: str, system_prompt: str = "") -> str:
    """
    Convenience wrapper so router.py can call OpenAI GPT directly.
    Uses MemoryManager for context.
    """
    try:
        memory = MemoryManager(user_id="default-router")
        api = NexaHybridAPI(user_id="default-router", memory=memory, local_model="mistral:latest")
        return api.generate_openai_response(user_input, system_prompt)
    except Exception as e:
        import logging
        logging.error(f"[Router GPT Wrapper] {e}")
        return f"[OpenAI API Error] {e}"
