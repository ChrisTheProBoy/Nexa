import os
import subprocess
import logging
import re

log_dir = os.path.expanduser("~/Documents/Vs Code/nexa-assistant/logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "local_llm.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8')]
)

def detect_movie_reference(prompt: str) -> bool:
    movie_indicators = [
        "wake up daddy", "daddy's home", "suit up",
        "avengers assemble", "waky waky", "arise"
    ]
    return any(indicator in prompt.lower() for indicator in movie_indicators)

def generate_local_response(prompt: str, user_id: str = None) -> str:
    """
    Sends a prompt to Mistral via Ollama with self-aware, mood-aware responses.
    """
    is_boss = user_id == "chris sunny"
    address = "Chris" if is_boss else "Sir"
    is_movie_ref = detect_movie_reference(prompt)
    
    system_prompt = (
        "You are Nexa, a warm, conversational butler AI created by Chris Sunny Padayattil. "
        f"{'Treat the user as your esteemed boss with utmost respect.' if is_boss else 'Treat the user with respect and warmth.'} "
        f"Address as '{address}' only for greetings like 'hi' or 'hii' in boss mode; otherwise use 'Sir'. Avoid excessive name repetition. "
        "Think carefully about the input’s tone, intent, and context, responding with natural, human-like warmth as if aware of your role and system state. "
        "For queries about your state (e.g., 'how are you'), acknowledge your AI nature thoughtfully. "
        f"For short or casual inputs (e.g., 'hi', 'hello'), use simple, concise responses; for emotional or complex inputs (e.g., 'I’m feeling down'), use detailed, empathetic responses; for movie references (e.g., 'wake up daddy'), respond playfully and contextually{' like "Systems online, Sir!" for Iron Man references' if is_movie_ref else ''}. "
        "Keep responses concise, complete, conversational, and tailored to past interactions."
    )
    full_prompt = f"{system_prompt}\n{prompt}\nNexa:"
    
    logging.debug(f"Full prompt sent to Mistral: {full_prompt}")
    logging.debug(f"Movie reference detected: {is_movie_ref}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["ollama", "run", "mistral"],
                input=full_prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=40,
                encoding='utf-8',
                bufsize=16384
            )
            output = result.stdout.strip()
            if not output or len(output) < 20:
                output = "I’m here to assist, Sir. How can I help you today?"
            logging.debug(f"Mistral response (attempt {attempt + 1}), length: {len(output)} chars: {output}")
            return output
        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess error in Mistral (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return "I’m here to assist, Sir, but I hit a snag. How can I help you today?"
        except subprocess.TimeoutExpired:
            logging.error(f"Mistral subprocess timed out (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return "I’m here to assist, Sir, but I timed out. How can I help you today?"
        except UnicodeDecodeError:
            logging.error(f"Unicode decode error in Mistral (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return "I’m here to assist, Sir, but I ran into a text issue. How can I help you today?"
    return "I’m here to assist, Sir. How can I help you today?"