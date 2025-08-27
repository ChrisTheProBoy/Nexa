import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

def detect_mood(text: str, user_id: str = None, memory_manager=None) -> str:
    """
    Detects mood with context-aware scoring, favoring simple responses for casual inputs.
    Returns: 'joy', 'excited', 'sad', 'angry', 'stressed', 'hopeful', 'mixed', 'calm', or 'neutral'.
    """
    if not text:
        return 'neutral'
    
    text = text.lower().strip()
    mood_scores = defaultdict(float)
    
    # Keyword groups with weights
    mood_keywords = {
        'joy': {'happy': 1.0, 'great': 1.0, 'awesome': 1.0},
        'excited': {'thrilled': 1.5, 'excited': 1.5, 'fantastic': 1.5, 'amazing': 1.5, 'super': 1.5},
        'sad': {'sad': 1.0, 'upset': 1.0, 'depressed': 1.2, 'down': 1.0, 'hurt': 1.0},
        'angry': {'angry': 1.0, 'frustrated': 1.0, 'annoyed': 1.0, 'mad': 1.0, 'pissed': 1.2},
        'stressed': {'stressed': 1.0, 'overwhelmed': 1.0, 'tired': 0.8, 'exhausted': 0.8},
        'hopeful': {'hope': 1.0, 'hopeful': 1.0, 'optimistic': 1.0, 'looking forward': 1.0},
        'calm': {'hi': 0.7, 'hii': 0.7, 'heloo': 0.7, 'hello': 0.7, 'how are you': 0.9, 'how\'s it going': 0.9, 'how are you doing': 0.9, 'how was your day': 0.9, 'what\'s up': 0.9, 'okay': 0.9, 'alright': 0.9}
    }
    
    # Score moods
    for mood, keywords in mood_keywords.items():
        for keyword, weight in keywords.items():
            if keyword in text:
                mood_scores[mood] += weight
    
    # Context-aware adjustments for simplicity
    input_length = len(text.split())
    is_repetitive = any(text.count(word) > 1 for word in ['hi', 'hii', 'hello', 'heloo'])
    if input_length < 6 or is_repetitive:
        mood_scores['calm'] += 0.7  # Boost calm for short or repetitive inputs
    
    # History-based adjustments
    if memory_manager and user_id:
        recent_mood = memory_manager.get_mood_trend()
        recent_interactions = memory_manager.retrieve_context(text, top_k=2)
        if recent_mood == 'calm' or any('hi' in i.lower() or 'hello' in i.lower() for i in recent_interactions):
            mood_scores['calm'] += 0.8  # Strong boost for conversational flow
        elif recent_mood in ['joy', 'excited', 'hopeful']:
            mood_scores[recent_mood] += 0.3
        elif recent_mood in ['sad', 'stressed', 'angry'] and 'calm' in mood_scores:
            mood_scores['calm'] += 0.5
    
    # Handle mixed emotions
    if mood_scores:
        max_score = max(mood_scores.values())
        dominant_moods = [mood for mood, score in mood_scores.items() if score >= max_score * 0.9]
        if len(dominant_moods) > 1 and 'sad' in dominant_moods and 'hopeful' in dominant_moods:
            return 'mixed'
        if 'calm' in dominant_moods:
            return 'calm'
        return dominant_moods[0]
    
    return 'neutral'