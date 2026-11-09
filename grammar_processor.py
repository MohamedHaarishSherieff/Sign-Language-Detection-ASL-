import re

def levenshtein_distance(s1, s2):
    """Calculates Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


class GrammarProcessor:
    def __init__(self):
        # Direct slang, abbreviations & common ASL fingerspelled shorthands
        self.common_shorthand = {
            "hllo": "hello", "hlo": "hello", "helo": "hello", "hell": "hello",
            "thnk": "thank", "thx": "thanks", "thnks": "thanks",
            "plz": "please", "pls": "please",
            "wrld": "world",
            "sign": "sign", "asl": "ASL",
            "hw": "how", "ru": "are you",
            "gd": "good", "mrg": "morning", "nt": "night",
            "yes": "yes", "no": "no", "bye": "goodbye",
            "welcm": "welcome", "sorr": "sorry", "sry": "sorry",
            "wht": "what", "whr": "where", "wen": "when", "y": "why",
            "help": "help", "name": "name", "nice": "nice", "meet": "meet",
            "you": "you", "friend": "friend", "love": "love"
        }

        # Core English vocabulary dictionary for 1-letter typo auto-correction
        self.vocabulary = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
            "hello", "thanks", "thank", "please", "welcome", "world", "friend", "happy",
            "morning", "night", "today", "tomorrow", "yesterday", "fine", "great", "name",
            "nice", "meet", "water", "food", "help", "stop", "again", "slow", "understand",
            "learn", "sign", "language", "school", "home", "family", "mother", "father"
        ]

    def _snap_to_vocabulary(self, word):
        """Finds closest English word if edit distance is 1 (or 2 for long words)."""
        if len(word) < 3:
            return word

        if word in self.vocabulary:
            return word

        best_match = None
        min_dist = 999
        max_allowed_dist = 1 if len(word) <= 5 else 2

        for vocab_word in self.vocabulary:
            if abs(len(vocab_word) - len(word)) > max_allowed_dist:
                continue

            dist = levenshtein_distance(word, vocab_word)
            if dist < min_dist and dist <= max_allowed_dist:
                min_dist = dist
                best_match = vocab_word

        return best_match if best_match else word

    def correct_sentence(self, text):
        if not text or not text.strip():
            return ""

        words = text.strip().split()
        cleaned_words = []

        for w in words:
            lower_w = w.lower()

            # Deduplicate runs of repeated letters (e.g. 'heelllo' -> 'hello')
            dedup = re.sub(r'(.)\1{2,}', r'\1', lower_w)

            # 1. Exact shorthand lookup (e.g. 'hllo' -> 'hello', 'thx' -> 'thanks')
            if lower_w in self.common_shorthand:
                cleaned_words.append(self.common_shorthand[lower_w])
            elif dedup in self.common_shorthand:
                cleaned_words.append(self.common_shorthand[dedup])
            else:
                # 2. Vocabulary snap using edit distance
                corrected_w = self._snap_to_vocabulary(dedup)
                cleaned_words.append(corrected_w)

        sentence = " ".join(cleaned_words)
        return sentence.capitalize() if sentence else ""
