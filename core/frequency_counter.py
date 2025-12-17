from typing import Dict, List, Tuple, Optional


class ManualHashMap:
    """A simple open-addressing hash map with linear probing.
    Stores string keys to integer values.
    """

    def __init__(self, initial_capacity: int = 1024):
        self.capacity = max(8, initial_capacity)
        self.size = 0
        self.keys: List[Optional[str]] = [None] * self.capacity
        self.vals: List[Optional[int]] = [None] * self.capacity

    def _hash(self, key: str) -> int:
        h = 0
        for ch in key:
            h = (h * 131 + ord(ch)) & 0x7FFFFFFF
        return h

    def _find_slot(self, key: str) -> int:
        idx = self._hash(key) % self.capacity
        start = idx
        while True:
            k = self.keys[idx]
            if k is None or k == key:
                return idx
            idx = (idx + 1) % self.capacity
            if idx == start:
                raise RuntimeError("HashMap full")

    def _resize(self) -> None:
        old_keys = self.keys
        old_vals = self.vals
        old_cap = self.capacity
        self.capacity *= 2
        self.keys = [None] * self.capacity
        self.vals = [None] * self.capacity
        self.size = 0
        for i in range(old_cap):
            k = old_keys[i]
            if k is not None:
                self.put(k, old_vals[i])

    def put(self, key: str, value: int) -> None:
        if (self.size + 1) / self.capacity > 0.7:
            self._resize()
        idx = self._find_slot(key)
        if self.keys[idx] is None:
            self.keys[idx] = key
            self.vals[idx] = value
            self.size += 1
        else:
            self.vals[idx] = value

    def get(self, key: str, default: Optional[int] = None) -> Optional[int]:
        idx = self._find_slot(key)
        if self.keys[idx] is None:
            return default
        return self.vals[idx]

    def increment(self, key: str, delta: int = 1) -> int:
        if (self.size + 1) / self.capacity > 0.7:
            self._resize()
        idx = self._find_slot(key)
        if self.keys[idx] is None:
            self.keys[idx] = key
            self.vals[idx] = delta
            self.size += 1
            return delta
        else:
            self.vals[idx] += delta
            return self.vals[idx]

    def items(self) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for i in range(self.capacity):
            if self.keys[i] is not None:
                out.append((self.keys[i], self.vals[i]))
        return out

    def keys_list(self) -> List[str]:
        out: List[str] = []
        for i in range(self.capacity):
            if self.keys[i] is not None:
                out.append(self.keys[i])
        return out

    def values(self) -> List[int]:
        out: List[int] = []
        for i in range(self.capacity):
            if self.keys[i] is not None:
                out.append(self.vals[i])
        return out

    def __len__(self) -> int:
        return self.size


class FrequencyCounter:
    """Manual word frequency counting using a custom hash map."""

    def __init__(self):
        self.frequency_map = ManualHashMap()
        self.total_words = 0

    def count_frequencies(self, tokens: List[str]) -> Dict[str, int]:
        """
        Count word frequencies using a custom hash map.
        Returns a transient dict view for compatibility with callers.
        """
        self.frequency_map = ManualHashMap()
        self.total_words = 0

        for token in tokens:
            self.total_words += 1
            self.frequency_map.increment(token, 1)

        # Transient dict view for external compatibility
        return {k: v for k, v in self.frequency_map.items()}

    def get_top_words(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get top N words by frequency using quicksort."""
        return self._quicksort_frequencies(self.frequency_map.items())[:n]

    def _quicksort_frequencies(self, items: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Manual quicksort implementation for word frequencies using a custom hash map.
        Sorts by frequency descending; ties broken alphabetically by word.
        """
        arr = items.copy()
        if len(arr) <= 1:
            return arr

        def qs(a: List[Tuple[str, int]], l: int, r: int) -> None:
            if l >= r:
                return
            pivot_freq = a[(l + r) // 2][1]
            pivot_word = a[(l + r) // 2][0]
            i, j = l, r
            while i <= j:
                # Find left element that should be on right (freq lower or same freq but word > pivot)
                while i <= r and (a[i][1] > pivot_freq or (a[i][1] == pivot_freq and a[i][0] < pivot_word)):
                    i += 1
                # Find right element that should be on left (freq higher or same freq but word < pivot)
                while j >= l and (a[j][1] < pivot_freq or (a[j][1] == pivot_freq and a[j][0] > pivot_word)):
                    j -= 1
                if i <= j:
                    a[i], a[j] = a[j], a[i]
                    i += 1
                    j -= 1
            if l < j:
                qs(a, l, j)
            if i < r:
                qs(a, i, r)

        qs(arr, 0, len(arr) - 1)
        return arr

    def get_statistics(self) -> Dict:
        """Calculate text statistics manually"""
        if len(self.frequency_map) == 0:
            return {}

        frequencies = self.frequency_map.values()

        # Mean
        total = 0
        count = 0
        for f in frequencies:
            total += f
            count += 1
        mean = total / count if count > 0 else 0.0

        # Median
        sorted_freq = self._quicksort_frequencies(self.frequency_map.items())
        sorted_values = [f for _, f in sorted_freq]
        if count % 2 == 0:
            median = (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2
        else:
            median = sorted_values[count // 2]

        # Standard deviation
        var_sum = 0.0
        for f in self.frequency_map.values():
            diff = f - mean
            var_sum += diff * diff
        variance = var_sum / count if count > 0 else 0.0
        std_dev = variance ** 0.5

        # Type-Token Ratio (Lexical Diversity)
        ttr = (len(self.frequency_map) / self.total_words) if self.total_words > 0 else 0.0

        # Min/Max
        min_f = None
        max_f = None
        for f in self.frequency_map.values():
            if min_f is None or f < min_f:
                min_f = f
            if max_f is None or f > max_f:
                max_f = f

        return {
            'total_words': self.total_words,
            'unique_words': len(self.frequency_map),
            'mean_frequency': mean,
            'median_frequency': median,
            'std_deviation': std_dev,
            'type_token_ratio': ttr,
            'max_frequency': max_f if max_f is not None else 0,
            'min_frequency': min_f if min_f is not None else 0
        }

    def zipf_law_analysis(self) -> List[Dict]:
        """Analyze distribution using Zipf's Law (expected ~ 1/r)."""
        top_words = self.get_top_words(len(self.frequency_map))

        zipf_data: List[Dict] = []
        rank = 1
        for word, freq in top_words:
            expected_freq = 1 / rank if rank > 0 else 0
            zipf_data.append({
                'rank': rank,
                'word': word,
                'frequency': freq,
                'expected_zipf': expected_freq
            })
            rank += 1

        return zipf_data
