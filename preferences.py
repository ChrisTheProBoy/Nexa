import re
from memory_manager import MemoryManager

def detect_and_store_preferences(memory: MemoryManager, user_input: str):
    """
    Detect preferences in user input and store them in memory.
    Includes specific rules + generic fallbacks.
    """
    lowered = user_input.lower()

    # --- Specific Rules ---

    # Programming language preferences
    if "in python" in lowered:
        memory.update_preference("programming_language", "Python")
    elif "in c++" in lowered:
        memory.update_preference("programming_language", "C++")
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
