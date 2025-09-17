import logging
import openai
import requests
import subprocess
import json

class NexaHybridAPI:
    """
    Hybrid API for Nexa:
      1. Try Ollama via REST API (fastest).
      2. If that fails, try Ollama CLI via subprocess.
      3. If both fail, fall back to OpenAI.
    Preferences from MemoryManager are always injected.
    Clarification rule: if input is vague, Nexa must ask clarifying questions.
    """

    def __init__(self, user_id, memory, local_model="mistral:latest", remote_model="gpt-4o-mini"):
        self.user_id = user_id
        self.memory = memory
        self.local_model = local_model
        self.remote_model = remote_model

    # ----------------- Preferences -----------------
    def _apply_preferences(self, base_prompt: str, user_input: str) -> str:
        prefs = self.memory.get_preferences()
        preference_lines = []

        if "programming_language" in prefs:
            preference_lines.append(
                f"⚠️ The user’s primary programming language is {prefs['programming_language']}. "
                f"Never default to another language unless explicitly requested."
            )

        if "tone" in prefs:
            preference_lines.append(f"Adopt a {prefs['tone']} tone.")
        if "units" in prefs:
            preference_lines.append(f"Use {prefs['units']} units when applicable.")
        if "emojis" in prefs:
            preference_lines.append("Include emojis." if prefs["emojis"] == "yes" else "Do not include emojis.")
        if "address_me_as" in prefs:
            preference_lines.append(f"Always address the user as {prefs['address_me_as']}.")

        for key, value in prefs.items():
            if key.startswith("always_") and value is True:
                preference_lines.append(f"Always {key.replace('always_', '').replace('_', ' ')}.")
            elif key.startswith("avoid_") and value is True:
                preference_lines.append(f"Avoid {key.replace('avoid_', '').replace('_', ' ')}.")
            elif key == "general_preference":
                preference_lines.append(f"Note user prefers: {value}.")

        clarification_rule = (
            "⚠️ IMPORTANT: If the user's input is vague, ambiguous, or unclear "
            f"(example: '{user_input}'), DO NOT guess. Instead, politely ask a clarifying question first."
        )
        preference_lines.append(clarification_rule)

        if preference_lines:
            base_prompt += "\nUser preferences & rules:\n" + "\n".join(preference_lines)

        return base_prompt

    # ----------------- Local Ollama (REST) -----------------
    def _call_ollama_rest(self, prompt: str) -> str | None:
        try:
            url = "http://localhost:11434/api/generate"
            payload = {"model": self.local_model, "prompt": prompt, "stream": False}
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return f"(local-{self.local_model}) {data.get('response', '').strip()}"
        except Exception as e:
            logging.warning(f"[Ollama REST] failed: {e}")
            return None

    # ----------------- Local Ollama (CLI) -----------------
    def _call_ollama_cli(self, prompt: str) -> str | None:
        try:
            result = subprocess.run(
                ["ollama", "run", self.local_model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=120
            )
            if result.returncode == 0:
                return f"(local-{self.local_model}) {result.stdout.decode('utf-8').strip()}"
            else:
                logging.error(f"[Ollama CLI] error: {result.stderr.decode('utf-8')}")
                return None
        except Exception as e:
            logging.error(f"[Ollama CLI] call failed: {e}")
            return None

    # ----------------- OpenAI -----------------
    def _call_openai(self, prompt: str, force_json: bool = False) -> str:
        try:
            if force_json:
                resp = openai.chat.completions.create(
                    model=self.remote_model,
                    messages=[
                        {"role": "system", "content": "You are a JSON-only output generator. Always reply with a valid JSON object, no prose."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=500,
                    response_format={"type": "json_object"}  # ✅ ensures raw JSON output
                )
                return resp.choices[0].message["content"].strip()
            else:
                resp = openai.chat.completions.create(
                    model=self.remote_model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.7,
                    max_tokens=600,
                )
                return f"(remote-{self.remote_model}) {resp.choices[0].message['content'].strip()}"
        except Exception as e:
            logging.error(f"[OpenAI] request failed: {e}")
            return f"My apologies, Sir, the OpenAI fallback also failed: {e}"

    # ----------------- Public Wrapper -----------------
    def generate_response(self, user_input: str, system_prompt: str, prefer_local=True, force_openai_json=False) -> str:
        try:
            final_prompt = self._apply_preferences(system_prompt, user_input)

            if prefer_local and not force_openai_json:
                logging.debug(f"[LOCAL-REST] Prompt: {final_prompt[:200]}...")
                resp = self._call_ollama_rest(final_prompt)
                if resp:
                    return resp

                logging.debug(f"[LOCAL-CLI fallback] Prompt: {final_prompt[:200]}...")
                resp = self._call_ollama_cli(final_prompt)
                if resp:
                    return resp

            logging.debug(f"[REMOTE] Prompt: {final_prompt[:200]}...")
            return self._call_openai(final_prompt, force_json=force_openai_json)

        except Exception as e:
            logging.error(f"[NexaHybridAPI] generate_response error: {e}")
            return f"My apologies, Sir, an error occurred while generating your response: {e}"
