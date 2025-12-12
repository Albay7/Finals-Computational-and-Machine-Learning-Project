import re
from typing import List, Tuple

class TextProcessor:
    """Manual text preprocessing without external NLP libraries"""

    def __init__(self, stop_words_file: str = "data/stop_words.txt"):
        """Initialize with custom stop words"""
        self.stop_words = self._load_stop_words(stop_words_file)

    def _load_stop_words(self, filepath: str) -> set:
        """Load stop words from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return set(word.strip().lower() for word in f.readlines())
        except FileNotFoundError:
            # Default English stop words if file not found
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                'who', 'when', 'where', 'why', 'how', 'as', 'if', 'so', 'than', 'then'
            }

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation using manual regex pattern"""
        pattern = r'[^\w\s]'  # Keep only alphanumeric and whitespace
        return re.sub(pattern, '', text)

    def tokenize(self, text: str) -> List[str]:
        """Split text into individual words (manual tokenization)"""
        text = text.lower()
        words = []
        current_word = ""

        for char in text:
            if char.isalnum():
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""

        if current_word:
            words.append(current_word)

        return words

    def preprocess(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Complete preprocessing pipeline"""
        # Convert to lowercase and remove punctuation
        text = self.remove_punctuation(text)

        # Tokenize
        tokens = self.tokenize(text)

        # Remove stop words if requested
        if remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        # Remove empty strings and very short words
        tokens = [word for word in tokens if len(word) > 2]

        return tokens

    def lemmatize_manual(self, word: str) -> str:
        """Manual lemmatization for common English words"""
        rules = {
            'ies': 'y',
            'ed': '',
            'ing': '',
            'es': '',
            's': ''
        }

        for suffix, replacement in rules.items():
            if word.endswith(suffix) and len(word) > 4:
                return word[:-len(suffix)] + replacement

        return word
