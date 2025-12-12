from typing import Dict, List, Tuple
import math

class NaiveBayesTextClassifier:
    """Manual Naive Bayes implementation for text classification"""

    def __init__(self):
        self.class_word_freq = {}      # Word frequencies per class
        self.class_doc_count = {}      # Document count per class
        self.vocabulary = set()         # Unique words in training data
        self.total_docs = 0

    def train(self, documents: List[Tuple[List[str], str]]) -> None:
        """
        Train the classifier
        documents: List of (token_list, class_label) tuples
        """
        self.total_docs = len(documents)

        # Initialize data structures
        self.class_word_freq = {}
        self.class_doc_count = {}

        for tokens, label in documents:
            # Count documents per class
            if label not in self.class_doc_count:
                self.class_doc_count[label] = 0
                self.class_word_freq[label] = {}

            self.class_doc_count[label] += 1

            # Count word frequencies per class
            for token in tokens:
                self.vocabulary.add(token)

                if token not in self.class_word_freq[label]:
                    self.class_word_freq[label][token] = 0

                self.class_word_freq[label][token] += 1

    def predict(self, tokens: List[str]) -> Tuple[str, Dict[str, float]]:
        """Predict class for given tokens with probability scores"""
        scores = {}

        for label in self.class_doc_count.keys():
            # Prior probability: P(Class) = Doc count / Total docs
            prior = self.class_doc_count[label] / self.total_docs
            scores[label] = math.log(prior)

            # Likelihood: P(Words|Class)
            for token in tokens:
                if token in self.class_word_freq[label]:
                    word_count = self.class_word_freq[label][token]
                else:
                    word_count = 0

                # Add-1 smoothing to avoid zero probability
                word_prob = (word_count + 1) / (
                    sum(self.class_word_freq[label].values()) +
                    len(self.vocabulary)
                )

                scores[label] += math.log(word_prob)

        # Predict the class with highest score
        predicted_class = max(scores, key=scores.get)

        return predicted_class, scores
