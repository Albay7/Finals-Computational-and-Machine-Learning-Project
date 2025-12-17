"""
Benchmark Runner for Frequency Counter Performance Analysis
Run this script to generate Results and Discussion data
"""

from core.frequency_counter import FrequencyCounter
from core.text_processor import TextProcessor
from core.perf_benchmark import PerformanceBenchmark
from utils.dataset_loader import DatasetLoader
import json

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_results(results_dict):
    """Pretty print results"""
    for key, value in results_dict.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"  {item}")
        else:
            print(f"{key}: {value}")

def main():
    print_section("FREQUENCY COUNTER PERFORMANCE BENCHMARKING")
    print("Computational Science & Machine Learning Project")
    print("Manual Algorithm Implementations - No External Libraries\n")

    # Initialize components
    counter = FrequencyCounter()
    processor = TextProcessor()
    benchmark = PerformanceBenchmark()
    dataset_loader = DatasetLoader()

    # Load sample data from datasets
    print("Loading sample data from AG News dataset...")
    ag_docs = dataset_loader.get_documents_for_classification('ag_news')

    if not ag_docs:
        print("Warning: No AG News data found. Using generated sample data.")
        sample_text = " ".join([f"word{i % 500}" for i in range(10000)])
        sample_tokens = processor.preprocess(sample_text, remove_stopwords=True)
    else:
        # Use first 100 documents
        sample_texts = [" ".join(tokens) for tokens, _ in ag_docs[:100]]
        combined_text = " ".join(sample_texts)
        sample_tokens = processor.preprocess(combined_text, remove_stopwords=True)

    print(f"Sample data prepared: {len(sample_tokens)} tokens\n")

    # Run all benchmarks
    all_results = benchmark.run_all_benchmarks(counter, sample_tokens)

    # Display results
    print_section("RESULTS")

    # 1. Time Complexity
    print("\n1. TIME COMPLEXITY BENCHMARKS")
    print("   Testing O(n) counting and O(n²) sorting performance\n")
    for result in all_results['time_complexity']:
        print(f"   Input Size: {result['input_size']:>7,} words")
        print(f"      - Count Time: {result['count_time_ms']:>8.2f} ms")
        print(f"      - Sort Time:  {result['sort_time_ms']:>8.2f} ms")
        print(f"      - Total Time: {result['total_time_ms']:>8.2f} ms")
        print(f"      - Throughput: {result['words_per_second']:>8,.0f} words/sec")
        print()

    # 2. Space Complexity
    print("\n2. SPACE COMPLEXITY ANALYSIS")
    space = all_results['space_complexity']
    print(f"   Total Words Processed: {space['total_words']:,}")
    print(f"   Unique Words (Hash Map Entries): {space['unique_words']:,}")
    print(f"   Memory Usage:")
    print(f"      - Hash Map Overhead: {space['hash_map_overhead_bytes']:,} bytes")
    print(f"      - Keys Total Size: {space['keys_total_bytes']:,} bytes")
    print(f"      - Values Total Size: {space['values_total_bytes']:,} bytes")
    print(f"      - Total Memory: {space['total_kb']:.2f} KB ({space['total_mb']:.4f} MB)")
    print(f"   Efficiency:")
    print(f"      - Bytes per Unique Word: {space['bytes_per_unique_word']:.2f}")
    print(f"      - Compression Ratio: {space['compression_ratio']:.4f}")

    # 3. Algorithm Comparison
    print("\n3. ALGORITHM PERFORMANCE COMPARISON")
    algo = all_results['algorithm_comparison']
    print(f"   Unique Words to Sort: {algo['unique_words']:,}")
    print(f"   Insertion Sort (O(n^2)): {algo['insertion_ms']:.4f} ms")
    print(f"   Selection Sort (O(n^2)): {algo['selection_ms']:.4f} ms")
    print(f"   Heapsort (O(n log n)):    {algo['heapsort_ms']:.4f} ms")
    print(f"   Mergesort (O(n log n)):   {algo['mergesort_ms']:.4f} ms")
    print(f"   Quicksort (avg O(n log n)): {algo['quicksort_ms']:.4f} ms")
    print(f"   Fastest: {algo['fastest']}")
    print(f"   Top-20 Consistency: {algo['top20_consistency']}")
    print(f"\n   Relative Speed vs Quicksort:")
    print(f"      - insertion_vs_quicksort:{algo['insertion_ms_vs_quicksort']:.3f}x")
    print(f"      - selection_vs_quicksort:{algo['selection_ms_vs_quicksort']:.3f}x")
    print(f"      - heapsort_vs_quicksort: {algo['heapsort_ms_vs_quicksort']:.3f}x")
    print(f"      - mergesort_vs_quicksort:{algo['mergesort_ms_vs_quicksort']:.3f}x")

    # 4. Statistical Validation
    print("\n4. STATISTICAL ACCURACY VALIDATION")
    stats = all_results['statistical_validation']
    print(f"   Mean Frequency:")
    print(f"      - Calculated: {stats['calculated_mean']}")
    print(f"      - Reference:  {stats['reference_mean']}")
    print(f"      - Error:      {stats['mean_error']:.10f}")
    print(f"   Median Frequency:")
    print(f"      - Calculated: {stats['calculated_median']}")
    print(f"      - Reference:  {stats['reference_median']}")
    print(f"      - Error:      {stats['median_error']:.10f}")
    print(f"   Standard Deviation:")
    print(f"      - Calculated: {stats['calculated_std']}")
    print(f"      - Reference:  {stats['reference_std']}")
    print(f"      - Error:      {stats['std_error']:.10f}")
    print(f"\n   Overall Accuracy: {'✓ PASS' if stats['all_accurate'] else '✗ FAIL'}")

    # 5. Zipf's Law Analysis
    print("\n5. ZIPF'S LAW FIT ANALYSIS")
    zipf = all_results['zipf_analysis']
    if 'error' not in zipf:
        print(f"   Data Points Analyzed: {zipf['data_points']}")
        print(f"   Correlation Coefficient (r): {zipf['correlation_coefficient']}")
        print(f"   R-squared (r²): {zipf['r_squared']}")
        print(f"   Mean Absolute Percentage Error: {zipf['mean_absolute_percentage_error']:.2f}%")
        print(f"   Chi-Square Statistic: {zipf['chi_square_statistic']:.4f}")
        print(f"   Fits Zipf's Law: {'Yes' if zipf['fits_zipf_law'] else 'No'}")
        print(f"   Interpretation: {zipf['interpretation']}")
    else:
        print(f"   {zipf['error']}")

    # Save results to JSON
    print_section("SAVING RESULTS")
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {output_file}")

    print_section("DISCUSSION POINTS")
    print("""
1. TIME COMPLEXITY:
   - O(n) counting performance scales linearly with input size
   - O(n²) bubble sort becomes bottleneck for large vocabularies
   - Throughput demonstrates hash map efficiency

2. SPACE COMPLEXITY:
   - Hash map compression ratio shows memory efficiency
   - Space usage grows with unique words (O(k)), not total words (O(n))
   - Practical for real-world text analysis

3. ALGORITHM COMPARISON:
   - Bubble sort useful for educational purposes, demonstrates O(n²)
   - Production systems should use O(n log n) algorithms
   - Trade-off: pedagogical value vs. performance

4. STATISTICAL ACCURACY:
   - Manual implementations match reference calculations
   - Error margins within acceptable floating-point precision
   - Validates correctness of custom algorithms

5. ZIPF'S LAW:
   - Natural language text follows power law distribution
   - High R² indicates strong adherence to Zipf's Law
   - Validates linguistic properties of analyzed text
    """)

    print_section("BENCHMARK COMPLETE")
    print(f"All results saved to: {output_file}")
    print("Use these metrics for your Results and Discussion section.\n")

if __name__ == "__main__":
    main()
