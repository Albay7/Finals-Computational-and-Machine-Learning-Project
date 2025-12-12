from typing import Dict, List, Tuple

class FrequencyCounter:
    """Manual word frequency counting using hash map principles"""

    def __init__(self):
        self.frequency_map = {}
        self.total_words = 0

    def count_frequencies(self, tokens: List[str]) -> Dict[str, int]:
        """
        Count word frequencies with O(n) time complexity
        Manual hash map implementation using Python dict
        """
        self.frequency_map = {}
        self.total_words = 0

        for token in tokens:
            self.total_words += 1

            # Manual frequency increment
            if token in self.frequency_map:
                self.frequency_map[token] += 1
            else:
                self.frequency_map[token] = 1

        return self.frequency_map

    def get_top_words(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get top N words by frequency using manual sorting"""
        return self._bubble_sort_frequencies(list(self.frequency_map.items()), n)

    def _bubble_sort_frequencies(self, items: List[Tuple[str, int]],
                                 limit: int) -> List[Tuple[str, int]]:
        """Manual bubble sort implementation for word frequencies"""
        arr = items.copy()
        n = len(arr)

        # Bubble sort by frequency (descending)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j][1] < arr[j + 1][1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

        return arr[:limit]

    def get_statistics(self) -> Dict:
        """Calculate text statistics manually"""
        if not self.frequency_map:
            return {}

        frequencies = list(self.frequency_map.values())

        # Manual mean calculation
        mean = sum(frequencies) / len(frequencies)

        # Manual median calculation
        sorted_freq = self._bubble_sort_frequencies(
            [(w, f) for w, f in self.frequency_map.items()],
            len(self.frequency_map)
        )
        sorted_values = [f for _, f in sorted_freq]

        if len(sorted_values) % 2 == 0:
            median = (sorted_values[len(sorted_values)//2 - 1] +
                     sorted_values[len(sorted_values)//2]) / 2
        else:
            median = sorted_values[len(sorted_values)//2]

        # Manual standard deviation
        variance = sum((f - mean) ** 2 for f in frequencies) / len(frequencies)
        std_dev = variance ** 0.5

        # Type-Token Ratio (Lexical Diversity)
        ttr = len(self.frequency_map) / self.total_words

        return {
            'total_words': self.total_words,
            'unique_words': len(self.frequency_map),
            'mean_frequency': mean,
            'median_frequency': median,
            'std_deviation': std_dev,
            'type_token_ratio': ttr,
            'max_frequency': max(frequencies),
            'min_frequency': min(frequencies)
        }

    def zipf_law_analysis(self) -> Dict:
        """Analyze distribution using Zipf's Law"""
        top_words = self.get_top_words(len(self.frequency_map))

        zipf_data = []
        for rank, (word, freq) in enumerate(top_words, 1):
            expected_freq = 1 / rank if rank > 0 else 0
            zipf_data.append({
                'rank': rank,
                'word': word,
                'frequency': freq,
                'expected_zipf': expected_freq
            })

        return zipf_data
