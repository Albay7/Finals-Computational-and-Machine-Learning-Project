"""
Performance Benchmark Module (clean)
Manual implementations for research Results and Discussion
No external libraries (NumPy/SciPy) - standard library only
"""

import time
import sys
import math


class PerformanceBenchmark:
    """Research-grade performance analysis with manual implementations"""

    def measure_time_complexity(self, counter, sample_tokens):
        """
        Benchmark time complexity: O(n) counting + O(n^2) bubble sort
        Returns a list of timing records for multiple input sizes.
        """
        results = []
        test_sizes = [100, 500, 1000, 5000, 10000]

        if not sample_tokens:
            for size in test_sizes:
                results.append({
                    'input_size': size,
                    'count_time_ms': 0.0,
                    'sort_time_ms': 0.0,
                    'total_time_ms': 0.0,
                    'words_per_second': 0.0,
                })
            return results

        for size in test_sizes:
            test_tokens = sample_tokens[:min(size, len(sample_tokens))]
            if len(test_tokens) < size and len(test_tokens) > 0:
                repeats = size // len(test_tokens) + 1
                test_tokens = (test_tokens * repeats)[:size]

            start_count = time.perf_counter()
            counter.count_frequencies(test_tokens)
            count_time = time.perf_counter() - start_count

            start_sort = time.perf_counter()
            counter.get_top_words(len(counter.frequency_map))
            sort_time = time.perf_counter() - start_sort

            total = count_time + sort_time
            results.append({
                'input_size': size,
                'count_time_ms': count_time * 1000.0,
                'sort_time_ms': sort_time * 1000.0,
                'total_time_ms': total * 1000.0,
                'words_per_second': (size / total) if total > 0 else 0.0,
            })

        return results

    def measure_space_complexity(self, counter, sample_tokens):
        """
        Analyze space complexity of the hash map representation.
        Returns memory usage and efficiency metrics.
        """
        counter.count_frequencies(sample_tokens)

        total_words = counter.total_words
        unique_words = len(counter.frequency_map)

        hash_map_overhead = sys.getsizeof(counter.frequency_map)
        # ManualHashMap exposes keys_list() and values() helpers; there is no keys()
        keys_total = sum(sys.getsizeof(k) for k in counter.frequency_map.keys_list())
        values_total = sum(sys.getsizeof(v) for v in counter.frequency_map.values())

        total_bytes = hash_map_overhead + keys_total + values_total
        total_kb = total_bytes / 1024.0
        total_mb = total_kb / 1024.0

        bytes_per_unique = (total_bytes / unique_words) if unique_words else 0.0
        compression_ratio = (unique_words / total_words) if total_words else 0.0

        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'hash_map_overhead_bytes': hash_map_overhead,
            'keys_total_bytes': keys_total,
            'values_total_bytes': values_total,
            'total_bytes': total_bytes,
            'total_kb': total_kb,
            'total_mb': total_mb,
            'bytes_per_unique_word': bytes_per_unique,
            'compression_ratio': compression_ratio,
        }

    def compare_sorting_algorithms(self, counter, sample_tokens):
        """
        Compare manual sorting algorithms on (word, count) pairs:
        - insertion (O(n^2))
        - selection (O(n^2))
        - heapsort (O(n log n), guaranteed)
        - mergesort (O(n log n), guaranteed, stable)
        - quicksort (avg O(n log n))
        Returns timings, fastest algorithm, and top-20 consistency.
        """
        counter.count_frequencies(sample_tokens)
        items = list(counter.frequency_map.items())

        # Insertion
        t0 = time.perf_counter()
        insertion_sorted = self._insertion_sort(items)
        t1 = time.perf_counter()
        insertion_time = t1 - t0

        # Selection
        t0 = time.perf_counter()
        selection_sorted = self._selection_sort(items)
        t1 = time.perf_counter()
        selection_time = t1 - t0

        # Heapsort
        t0 = time.perf_counter()
        heap_sorted = self._heapsort(items)
        t1 = time.perf_counter()
        heap_time = t1 - t0

        # Mergesort
        t0 = time.perf_counter()
        merge_sorted = self._mergesort(items)
        t1 = time.perf_counter()
        merge_time = t1 - t0

        # Quicksort
        t0 = time.perf_counter()
        quick_sorted = self._quicksort(items)
        t1 = time.perf_counter()
        quick_time = t1 - t0

        # Top-20 set consistency (tolerate ties/order differences)
        def top_set(arr, n=20):
            return set(w for w, _ in (arr[:n] if len(arr) >= n else arr))

        ref = top_set(quick_sorted)
        consistent = (
            ref == top_set(insertion_sorted) and
            ref == top_set(selection_sorted) and
            ref == top_set(heap_sorted) and
            ref == top_set(merge_sorted)
        )

        times = {
            'insertion_ms': round(insertion_time * 1000.0, 4),
            'selection_ms': round(selection_time * 1000.0, 4),
            'heapsort_ms': round(heap_time * 1000.0, 4),
            'mergesort_ms': round(merge_time * 1000.0, 4),
            'quicksort_ms': round(quick_time * 1000.0, 4),
        }
        fastest = min(times, key=lambda k: times[k])

        rel = {}
        for k, ms in times.items():
            rel[k + '_vs_quicksort'] = round((times['quicksort_ms'] / ms) if ms > 0 else 0.0, 3)

        return {
            'unique_words': len(items),
            **times,
            **rel,
            'fastest': fastest,
            'top20_consistency': consistent,
        }

    def validate_statistics_accuracy(self, counter, sample_tokens):
        """
        Validate the stats produced by FrequencyCounter against a reference
        calculation (both manual, no external libs). Returns error metrics.
        """
        counter.count_frequencies(sample_tokens)
        stats = counter.get_statistics() or {}

        freq_values = list(counter.frequency_map.values())
        n = len(freq_values)
        if n == 0:
            return {
                'calculated_mean': 0,
                'reference_mean': 0,
                'mean_error': 0,
                'calculated_median': 0,
                'reference_median': 0,
                'median_error': 0,
                'calculated_std': 0,
                'reference_std': 0,
                'std_error': 0,
                'all_accurate': True,
            }

        ref_mean = sum(freq_values) / n

        sorted_vals = sorted(freq_values)
        if n % 2 == 0:
            ref_median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            ref_median = sorted_vals[n // 2]

        ref_variance = sum((x - ref_mean) ** 2 for x in freq_values) / n
        ref_std = ref_variance ** 0.5

        calc_mean = stats.get('mean_frequency', 0)
        calc_median = stats.get('median_frequency', 0)
        calc_std = stats.get('std_deviation', 0)

        mean_err = abs(calc_mean - ref_mean)
        median_err = abs(calc_median - ref_median)
        std_err = abs(calc_std - ref_std)

        threshold = 1e-10
        return {
            'calculated_mean': calc_mean,
            'reference_mean': ref_mean,
            'mean_error': mean_err,
            'calculated_median': calc_median,
            'reference_median': ref_median,
            'median_error': median_err,
            'calculated_std': calc_std,
            'reference_std': ref_std,
            'std_error': std_err,
            'all_accurate': (mean_err < threshold and median_err < threshold and std_err < threshold),
        }

    def analyze_zipf_law_fit(self, counter, sample_tokens):
        """
        Assess adherence to Zipf's Law using correlation on log-log scale,
        MAPE and chi-square against expected f(r) ~= f(1)/r.
        """
        counter.count_frequencies(sample_tokens)
        sorted_items = counter.get_top_words(len(counter.frequency_map))

        n_points = min(100, len(sorted_items))
        if n_points < 10:
            return {'error': 'Insufficient data points for Zipf analysis (need >= 10 unique words)'}

        ranks = list(range(1, n_points + 1))
        freqs = [freq for _, freq in sorted_items[:n_points]]

        log_ranks = [math.log(r) for r in ranks]
        log_freqs = [math.log(f) if f > 0 else 0 for f in freqs]

        r = self._pearson_corr(log_ranks, log_freqs)
        r_squared = r * r

        expected = [freqs[0] / r for r in ranks]
        mape = self._mape(freqs, expected)
        chi_sq = self._chi_square(freqs, expected)

        if r_squared > 0.95:
            interpretation = 'Excellent fit - strongly follows Zipf\'s Law'
        elif r_squared > 0.85:
            interpretation = 'Good fit - generally follows Zipf\'s Law'
        elif r_squared > 0.70:
            interpretation = 'Moderate fit - loosely follows Zipf\'s Law'
        else:
            interpretation = 'Poor fit - does not follow Zipf\'s Law'

        return {
            'data_points': n_points,
            'correlation_coefficient': r,
            'r_squared': r_squared,
            'mean_absolute_percentage_error': mape,
            'chi_square_statistic': chi_sq,
            'fits_zipf_law': r_squared > 0.85,
            'interpretation': interpretation,
        }

    # --------------------- helpers ---------------------

    def _insertion_sort(self, items):
        arr = items.copy()
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j][1] < key[1]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    def _selection_sort(self, items):
        arr = items.copy()
        n = len(arr)
        for i in range(n):
            max_idx = i
            for j in range(i + 1, n):
                if arr[j][1] > arr[max_idx][1]:
                    max_idx = j
            arr[i], arr[max_idx] = arr[max_idx], arr[i]
        return arr

    def _heapsort(self, items):
        """Manual heapsort implementation for word frequencies (O(n log n))."""
        arr = items.copy()
        n = len(arr)

        def heapify(a, size, idx):
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < size and a[left][1] < a[largest][1]:
                largest = left
            if right < size and a[right][1] < a[largest][1]:
                largest = right
            if largest != idx:
                a[idx], a[largest] = a[largest], a[idx]
                heapify(a, size, largest)

        # Build max heap (sorted by frequency descending)
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)

        # Extract elements from heap
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(arr, i, 0)

        return arr

    def _mergesort(self, items):
        """Manual mergesort implementation for word frequencies (O(n log n), guaranteed)."""
        arr = items.copy()

        def merge(a, l, m, r):
            left = a[l:m+1]
            right = a[m+1:r+1]
            i, j, k = 0, 0, l
            while i < len(left) and j < len(right):
                if left[i][1] > right[j][1]:
                    a[k] = left[i]
                    i += 1
                else:
                    a[k] = right[j]
                    j += 1
                k += 1
            while i < len(left):
                a[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                a[k] = right[j]
                j += 1
                k += 1

        def ms(a, l, r):
            if l < r:
                m = (l + r) // 2
                ms(a, l, m)
                ms(a, m + 1, r)
                merge(a, l, m, r)

        if len(arr) > 0:
            ms(arr, 0, len(arr) - 1)
        return arr

    def _quicksort(self, items):
        arr = items.copy()

        def qs(a, l, r):
            if l >= r:
                return
            pivot = a[(l + r) // 2][1]
            i, j = l, r
            while i <= j:
                while a[i][1] > pivot:
                    i += 1
                while a[j][1] < pivot:
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

    def _pearson_corr(self, xs, ys):
        n = len(xs)
        if n == 0:
            return 0.0
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den_x = sum((x - mean_x) ** 2 for x in xs)
        den_y = sum((y - mean_y) ** 2 for y in ys)
        den = (den_x * den_y) ** 0.5
        return (num / den) if den != 0 else 0.0

    def _mape(self, actual, expected):
        errors = [abs((a - e) / a) * 100 for a, e in zip(actual, expected) if a != 0]
        return (sum(errors) / len(errors)) if errors else 0.0

    def _chi_square(self, observed, expected):
        terms = [((o - e) ** 2) / e for o, e in zip(observed, expected) if e != 0]
        return sum(terms)

    def run_all_benchmarks(self, counter, sample_tokens):
        """
        Run the complete suite: time, space, algorithm comparison, statistics
        validation, and Zipf's Law fit. Returns a dict with all results.
        """
        print('Running benchmarks...')
        print('1. Time Complexity Analysis...')
        time_results = self.measure_time_complexity(counter, sample_tokens)

        print('2. Space Complexity Analysis...')
        space_results = self.measure_space_complexity(counter, sample_tokens)

        print('3. Algorithm Performance Comparison...')
        algo_results = self.compare_sorting_algorithms(counter, sample_tokens)

        print('4. Statistical Accuracy Validation...')
        stats_results = self.validate_statistics_accuracy(counter, sample_tokens)

        print('5. Zipf\'s Law Fit Analysis...')
        zipf_results = self.analyze_zipf_law_fit(counter, sample_tokens)

        print('✓ All benchmarks complete!\n')

        return {
            'time_complexity': time_results,
            'space_complexity': space_results,
            'algorithm_comparison': algo_results,
            'statistical_validation': stats_results,
            'zipf_analysis': zipf_results,
        }
