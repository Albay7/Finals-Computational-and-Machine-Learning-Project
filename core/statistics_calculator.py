"""
Performance Benchmark Module
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
            # Nothing to benchmark; return zeros for the requested sizes
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
            # Prepare test tokens of desired length
            test_tokens = sample_tokens[:min(size, len(sample_tokens))]
            if len(test_tokens) < size and len(test_tokens) > 0:
                repeats = size // len(test_tokens) + 1
                test_tokens = (test_tokens * repeats)[:size]

            # Counting time (O(n))
            start_count = time.perf_counter()
            counter.count_frequencies(test_tokens)
            count_time = time.perf_counter() - start_count

            # Sorting time (O(m^2) over unique words m) using the counter's bubble sort
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

        # Object overhead + keys + values
        hash_map_overhead = sys.getsizeof(counter.frequency_map)
        keys_total = sum(sys.getsizeof(k) for k in counter.frequency_map.keys())
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
        Compare bubble sort (manual O(n^2)) vs built-in Timsort (O(n log n)).
        Returns timing and correctness comparison.
        """
        counter.count_frequencies(sample_tokens)
        items = list(counter.frequency_map.items())

        # Bubble sort (manual)
        start_bubble = time.perf_counter()
        bubble_sorted = counter._bubble_sort_frequencies(items, len(items))
        bubble_time = time.perf_counter() - start_bubble

        # Built-in sort (Timsort)
        start_builtin = time.perf_counter()
        builtin_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        builtin_time = time.perf_counter() - start_builtin

        results_match = (bubble_sorted == builtin_sorted)
        speedup = (bubble_time / builtin_time) if builtin_time > 0 else float('inf')

        return {
            'unique_words': len(items),
            'bubble_sort_ms': bubble_time * 1000.0,
            'builtin_sort_ms': builtin_time * 1000.0,
            'speedup_factor': f"{speedup:.1f}",
            'results_match': results_match,
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

        # Reference mean/median/std (using simple Python operations)
        ref_mean = sum(freq_values) / n

        sorted_vals = sorted(freq_values)
        if n % 2 == 0:
            ref_median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            ref_median = sorted_vals[n // 2]

        ref_variance = sum((x - ref_mean) ** 2 for x in freq_values) / n
        ref_std = ref_variance ** 0.5

        # Extract calculated from counter
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

        # Use natural logs from standard library for stability
        log_ranks = [math.log(r) for r in ranks]
        log_freqs = [math.log(f) if f > 0 else 0 for f in freqs]

        r = self._pearson_corr(log_ranks, log_freqs)
        r_squared = r * r

        # Expected Zipf frequencies scaled to first term
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
"""
Performance Benchmark Module
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
            # Nothing to benchmark; return zeros for the requested sizes
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
            # Prepare test tokens of desired length
            test_tokens = sample_tokens[:min(size, len(sample_tokens))]
            if len(test_tokens) < size and len(test_tokens) > 0:
                repeats = size // len(test_tokens) + 1
                test_tokens = (test_tokens * repeats)[:size]

            # Counting time (O(n))
            start_count = time.perf_counter()
            counter.count_frequencies(test_tokens)
            count_time = time.perf_counter() - start_count

            # Sorting time (O(m^2) over unique words m) using the counter's bubble sort
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

        # Object overhead + keys + values
        hash_map_overhead = sys.getsizeof(counter.frequency_map)
        keys_total = sum(sys.getsizeof(k) for k in counter.frequency_map.keys())
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
        Compare bubble sort (manual O(n^2)) vs built-in Timsort (O(n log n)).
        Returns timing and correctness comparison.
        """
        counter.count_frequencies(sample_tokens)
        items = list(counter.frequency_map.items())

        # Bubble sort (manual)
        start_bubble = time.perf_counter()
        bubble_sorted = counter._bubble_sort_frequencies(items, len(items))
        bubble_time = time.perf_counter() - start_bubble

        # Built-in sort (Timsort)
        start_builtin = time.perf_counter()
        builtin_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        builtin_time = time.perf_counter() - start_builtin

        results_match = (bubble_sorted == builtin_sorted)
        speedup = (bubble_time / builtin_time) if builtin_time > 0 else float('inf')

        return {
            'unique_words': len(items),
            'bubble_sort_ms': bubble_time * 1000.0,
            'builtin_sort_ms': builtin_time * 1000.0,
            'speedup_factor': f"{speedup:.1f}",
            'results_match': results_match,
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

        # Reference mean/median/std (using simple Python operations)
        ref_mean = sum(freq_values) / n

        sorted_vals = sorted(freq_values)
        if n % 2 == 0:
            ref_median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            ref_median = sorted_vals[n // 2]

        ref_variance = sum((x - ref_mean) ** 2 for x in freq_values) / n
        ref_std = ref_variance ** 0.5

        # Extract calculated from counter
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

        # Use natural logs from standard library for stability
        log_ranks = [math.log(r) for r in ranks]
        log_freqs = [math.log(f) if f > 0 else 0 for f in freqs]

        r = self._pearson_corr(log_ranks, log_freqs)
        r_squared = r * r

        # Expected Zipf frequencies scaled to first term
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
"""
Performance Benchmark Module
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

        for size in test_sizes:
            # Prepare test tokens of desired length
            test_tokens = sample_tokens[:min(size, len(sample_tokens))]
            if len(test_tokens) < size and len(test_tokens) > 0:
                repeats = size // len(test_tokens) + 1
                test_tokens = (test_tokens * repeats)[:size]

            # Counting time (O(n))
            start_count = time.perf_counter()
            counter.count_frequencies(test_tokens)
            count_time = time.perf_counter() - start_count

            # Sorting time over unique words (bubble sort)
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

        # Object overhead + keys + values
        hash_map_overhead = sys.getsizeof(counter.frequency_map)
        keys_total = sum(sys.getsizeof(k) for k in counter.frequency_map.keys())
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
        Compare bubble sort (manual O(n^2)) vs built-in Timsort (O(n log n)).
        Returns timing and correctness comparison.
        """
        counter.count_frequencies(sample_tokens)
        items = list(counter.frequency_map.items())

        # Bubble sort (manual)
        start_bubble = time.perf_counter()
        bubble_sorted = counter._bubble_sort_frequencies(items, len(items))
        bubble_time = time.perf_counter() - start_bubble

        # Built-in sort (Timsort)
        start_builtin = time.perf_counter()
        builtin_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        builtin_time = time.perf_counter() - start_builtin

        results_match = (bubble_sorted == builtin_sorted)
        speedup = (bubble_time / builtin_time) if builtin_time > 0 else float('inf')

        return {
            'unique_words': len(items),
            'bubble_sort_ms': bubble_time * 1000.0,
            'builtin_sort_ms': builtin_time * 1000.0,
            'speedup_factor': f"{speedup:.1f}",
            'results_match': results_match,
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

        # Reference mean/median/std (using simple Python operations)
        ref_mean = sum(freq_values) / n

        sorted_vals = sorted(freq_values)
        if n % 2 == 0:
            ref_median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            ref_median = sorted_vals[n // 2]

        ref_variance = sum((x - ref_mean) ** 2 for x in freq_values) / n
        ref_std = ref_variance ** 0.5

        # Extract calculated from counter
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

        # Use natural logs from standard library for stability
        log_ranks = [math.log(r) for r in ranks]
        log_freqs = [math.log(f) if f > 0 else 0 for f in freqs]

        r = self._pearson_corr(log_ranks, log_freqs)
        r_squared = r * r

        # Expected Zipf frequencies scaled to first term
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
"""
Performance Benchmark Module
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

        for size in test_sizes:
            # Prepare test tokens of desired length
            test_tokens = sample_tokens[:min(size, len(sample_tokens))]
            if len(test_tokens) < size and len(test_tokens) > 0:
                repeats = size // len(test_tokens) + 1
                test_tokens = (test_tokens * repeats)[:size]

            # Counting time (O(n))
            start_count = time.perf_counter()
            counter.count_frequencies(test_tokens)
            count_time = time.perf_counter() - start_count

            # Sorting time (O(m^2) over unique words m)
            # Use internal bubble sort via get_top_words over full vocabulary
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

        # Object overhead + keys + values
        hash_map_overhead = sys.getsizeof(counter.frequency_map)
        keys_total = sum(sys.getsizeof(k) for k in counter.frequency_map.keys())
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
        Compare bubble sort (manual O(n^2)) vs built-in Timsort (O(n log n)).
        Returns timing and correctness comparison.
        """
        counter.count_frequencies(sample_tokens)
        items = list(counter.frequency_map.items())

        # Bubble sort (manual)
        start_bubble = time.perf_counter()
        bubble_sorted = counter._bubble_sort_frequencies(items, len(items))
        bubble_time = time.perf_counter() - start_bubble

        # Built-in sort (Timsort)
        start_builtin = time.perf_counter()
        builtin_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        builtin_time = time.perf_counter() - start_builtin

        results_match = (bubble_sorted == builtin_sorted)
        speedup = (bubble_time / builtin_time) if builtin_time > 0 else float('inf')

        return {
            'unique_words': len(items),
            'bubble_sort_ms': bubble_time * 1000.0,
            'builtin_sort_ms': builtin_time * 1000.0,
            'speedup_factor': f"{speedup:.1f}",
            'results_match': results_match,
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

        # Reference mean/median/std (using simple Python operations)
        ref_mean = sum(freq_values) / n

        sorted_vals = sorted(freq_values)
        if n % 2 == 0:
            ref_median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            ref_median = sorted_vals[n // 2]

        ref_variance = sum((x - ref_mean) ** 2 for x in freq_values) / n
        ref_std = ref_variance ** 0.5

        # Extract calculated from counter
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
        MAPE and chi-square against expected f(r) ≈ f(1)/r.
        """
        counter.count_frequencies(sample_tokens)
        sorted_items = counter.get_top_words(len(counter.frequency_map))

        n_points = min(100, len(sorted_items))
        if n_points < 10:
            return {'error': 'Insufficient data points for Zipf analysis (need >= 10 unique words)'}

        ranks = list(range(1, n_points + 1))
        freqs = [freq for _, freq in sorted_items[:n_points]]

        # Use natural logs from standard library for stability
        log_ranks = [math.log(r) for r in ranks]
        log_freqs = [math.log(f) if f > 0 else 0 for f in freqs]

        r = self._pearson_corr(log_ranks, log_freqs)
        r_squared = r * r

        # Expected Zipf frequencies scaled to first term
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

"""
Performance Benchmark Module
Manual implementations for research Results and Discussion
No external libraries (NumPy/SciPy) - pure Python only
"""

import time
"""
Performance Benchmark Module
Manual implementations for research Results and Discussion
No external libraries (NumPy/SciPy) - pure Python only
"""

class PerformanceBenchmark:
    """Research-grade performance analysis with manual implementations"""

    def __init__(self):
        pass
    """Research-grade performance analysis with manual implementations"""

    def measure_time_complexity(self, counter, sample_tokens):
        pass
        Benchmark time complexity: O(n) counting + O(n²) sorting

        Tests on different input sizes to demonstrate algorithmic complexity
        Returns timing data for Results section
        """
        results = []

        # Test different input sizes
        test_sizes = [100, 500, 1000, 5000, 10000]

        for size in test_sizes:
            # Get subset of tokens
            test_tokens = sample_tokens[:min(size, len(sample_tokens))]
            if len(test_tokens) < size:
                # Repeat tokens if needed
                test_tokens = test_tokens * (size // len(test_tokens) + 1)
                test_tokens = test_tokens[:size]

            # Measure counting time (O(n))
            start_count = time.perf_counter()
            frequencies = counter.count_frequencies(test_tokens)
            end_count = time.perf_counter()
            start_count = time.perf_counter()
            frequencies = counter.count_frequencies(test_tokens)
            end_count = time.perf_counter()
            count_time = end_count - start_count
            start_sort = time.perf_counter()
            sorted_freq = counter.bubble_sort_frequencies(frequencies)
            start_sort = time.perf_counter()
            sorted_freq = counter.bubble_sort_frequencies(frequencies)
            end_sort = time.perf_counter()
            sort_time = end_sort - start_sort
            results.append({
                'input_size': size,
                'count_time_ms': count_time * 1000,
                'count_time_ms': count_time * 1000,
                'sort_time_ms': sort_time * 1000,
                'total_time_ms': (count_time + sort_time) * 1000,
                'words_per_second': size / (count_time + sort_time) if count_time + sort_time > 0 else 0

        return results

    def measure_space_complexity(self, counter, sample_tokens):
        """
        Analyze space complexity: hash map storage efficiency

        Measures:
        - Hash map overhead
        - Memory per unique word
        - Compression ratio (unique words / total words)
        """
        # Count frequencies to build hash map
        frequencies = counter.count_frequencies(sample_tokens)

        # Calculate space usage manually
        total_words = len(sample_tokens)
        unique_words = len(frequencies)
            'unique_words': len(counter.frequency_map),
            'hash_map_overhead_bytes': hash_map_overhead,
            'keys_total_bytes': total_key_bytes,
            'values_total_bytes': total_value_bytes,
            'total_bytes': total_bytes,
            'total_kb': round(total_bytes / 1024, 2),
            'total_mb': round(total_bytes / (1024 * 1024), 4),
            'bytes_per_unique_word': round(total_bytes / len(counter.frequency_map), 2) if len(counter.frequency_map) > 0 else 0,
            'compression_ratio': round(len(counter.frequency_map) / counter.total_words, 4) if counter.total_words > 0 else 0
        }

    # ========== ALGORITHM PERFORMANCE COMPARISON ==========

    def compare_sorting_algorithms(self, counter, tokens: List[str]) -> Dict:
        """
        Compare bubble sort vs Python's built-in sort
        Shows educational vs production trade-offs
        """
        counter.count_frequencies(tokens)
        items = list(counter.frequency_map.items())

        # Test bubble sort (manual O(n²))
        start = time.perf_counter()
        bubble_result = counter._bubble_sort_frequencies(items, 20)
        bubble_time = time.perf_counter() - start

        # Test Python's built-in sort (Timsort O(n log n))
        start = time.perf_counter()
        builtin_sorted = sorted(items, key=lambda x: x[1], reverse=True)[:20]
        builtin_time = time.perf_counter() - start

        # Verify results match
        results_match = bubble_result == builtin_sorted

        return {
            'unique_words': len(items),
            'bubble_sort_ms': round(bubble_time * 1000, 4),
            'builtin_sort_ms': round(builtin_time * 1000, 4),
            'speedup_factor': round(bubble_time / builtin_time, 2) if builtin_time > 0 else 0,
            'results_match': results_match,
            'bubble_complexity': 'O(n²)',
            'builtin_complexity': 'O(n log n)'
        }

    # ========== STATISTICAL ACCURACY VALIDATION ==========

    def validate_statistics(self, counter, tokens: List[str]) -> Dict:
        """
        Validate manual statistics against reference calculations
        Tests mean, median, std dev accuracy
        """
        counter.count_frequencies(tokens)
        stats = counter.get_statistics()

        # Manual reference calculations
        frequencies = list(counter.frequency_map.values())

        # Reference mean
        ref_mean = sum(frequencies) / len(frequencies)

        # Reference median (manual)
        sorted_freqs = sorted(frequencies)
        n = len(sorted_freqs)
        if n % 2 == 0:
            ref_median = (sorted_freqs[n//2 - 1] + sorted_freqs[n//2]) / 2
        else:
            ref_median = sorted_freqs[n//2]

        # Reference standard deviation
        ref_variance = sum((f - ref_mean) ** 2 for f in frequencies) / len(frequencies)
        ref_std = ref_variance ** 0.5

        # Calculate errors
        mean_error = abs(stats['mean_frequency'] - ref_mean)
        median_error = abs(stats['median_frequency'] - ref_median)
        std_error = abs(stats['std_deviation'] - ref_std)

        return {
            'calculated_mean': round(stats['mean_frequency'], 6),
            'reference_mean': round(ref_mean, 6),
            'mean_error': round(mean_error, 10),
            'calculated_median': round(stats['median_frequency'], 6),
            'reference_median': round(ref_median, 6),
            'median_error': round(median_error, 10),
            'calculated_std': round(stats['std_deviation'], 6),
            'reference_std': round(ref_std, 6),
            'std_error': round(std_error, 10),
            'all_accurate': mean_error < 0.0001 and median_error < 0.0001 and std_error < 0.0001
        }

    # ========== ZIPF'S LAW FIT ANALYSIS ==========

    def analyze_zipf_law_fit(self, counter, tokens: List[str]) -> Dict:
        """
        Analyze how well data fits Zipf's Law
        Manual correlation coefficient calculation
        """
        counter.count_frequencies(tokens)
        zipf_data = counter.zipf_law_analysis()

        if len(zipf_data) < 2:
            return {'error': 'Insufficient data for Zipf analysis'}

        # Take top 50 words for analysis
        top_n = min(50, len(zipf_data))
        data_subset = zipf_data[:top_n]

        # Extract ranks and frequencies
        ranks = [d['rank'] for d in data_subset]
        frequencies = [d['frequency'] for d in data_subset]

        # Calculate expected Zipf frequencies: f(r) = C/r
        C = frequencies[0]  # Constant = frequency of rank 1
        expected_freqs = [C / r for r in ranks]

        # Manual Pearson correlation coefficient
        correlation = self._calculate_correlation(frequencies, expected_freqs)

        # Manual R-squared
        r_squared = correlation ** 2

        # Calculate mean absolute percentage error
        mape = self._calculate_mape(frequencies, expected_freqs)

        # Manual chi-square goodness of fit
        chi_square = self._calculate_chi_square(frequencies, expected_freqs)

        return {
            'data_points': top_n,
            'correlation_coefficient': round(correlation, 4),
            'r_squared': round(r_squared, 4),
            'mean_absolute_percentage_error': round(mape, 2),
            'chi_square_statistic': round(chi_square, 4),
            'fits_zipf_law': r_squared > 0.85,  # Good fit threshold
            'interpretation': self._interpret_zipf_fit(r_squared)
        }

    # ========== HELPER METHODS ==========

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Manual Pearson correlation coefficient"""
        n = len(x)
        if n == 0:
            return 0.0

        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Calculate correlation
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        denominator = (sum_sq_x * sum_sq_y) ** 0.5

        return numerator / denominator if denominator != 0 else 0.0

    def _calculate_mape(self, actual: List[float], predicted: List[float]) -> float:
        """Manual Mean Absolute Percentage Error"""
        n = len(actual)
        if n == 0:
            return 0.0

        total_percentage_error = sum(
            abs((actual[i] - predicted[i]) / actual[i]) * 100
            for i in range(n) if actual[i] != 0
        )

        return total_percentage_error / n

    def _calculate_chi_square(self, observed: List[float], expected: List[float]) -> float:
        """Manual chi-square statistic"""
        chi_sq = sum(
            ((observed[i] - expected[i]) ** 2) / expected[i]
            for i in range(len(observed)) if expected[i] != 0
        )
        return chi_sq

    def _interpret_zipf_fit(self, r_squared: float) -> str:
        """Interpret R² value for Zipf's Law fit"""
        if r_squared > 0.95:
            return "Excellent fit - Strong adherence to Zipf's Law"
        elif r_squared > 0.85:
            return "Good fit - Follows Zipf's Law pattern"
        elif r_squared > 0.70:
            return "Moderate fit - Partial adherence to Zipf's Law"
        else:
            return "Poor fit - Does not follow Zipf's Law"

    # ========== RUN ALL BENCHMARKS ==========

    def run_all_benchmarks(self, counter, sample_tokens: List[str]) -> Dict:
        """Run complete benchmark suite"""
        print("Running comprehensive benchmarks...")
        print("=" * 60)

        results = {}

        # 1. Time Complexity
        print("\n1. Time Complexity Analysis...")
        test_sizes = [100, 500, 1000, 5000, 10000]
        results['time_complexity'] = self.measure_time_complexity(counter, test_sizes)

        # 2. Space Complexity
        print("2. Space Complexity Analysis...")
        results['space_complexity'] = self.measure_space_complexity(counter, sample_tokens)

        # 3. Algorithm Comparison
        print("3. Algorithm Performance Comparison...")
        results['algorithm_comparison'] = self.compare_sorting_algorithms(counter, sample_tokens)

        # 4. Statistical Validation
        print("4. Statistical Accuracy Validation...")
        results['statistical_validation'] = self.validate_statistics(counter, sample_tokens)

        # 5. Zipf's Law Analysis
        print("5. Zipf's Law Fit Analysis...")
        results['zipf_analysis'] = self.analyze_zipf_law_fit(counter, sample_tokens)

        print("\n" + "=" * 60)
        print("Benchmarks complete!")

        return results
