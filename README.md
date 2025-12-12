# Computational Text Analysis & Machine Learning Platform

A comprehensive text analysis system combining **Computational Science** (manual algorithm implementations for frequency analysis) and **Machine Learning** (classification models for text categorization). The system uses manual implementations of core algorithms alongside industry-standard ML models, with a Streamlit-based web interface.

---

## Table of Contents
- [System Overview](#system-overview)
- [Part 1: Frequency Analysis (Computational Science)](#part-1-frequency-analysis-computational-science)
- [Part 2: Text Classification (Machine Learning)](#part-2-text-classification-machine-learning)
- [Installation & Setup](#installation--setup)
- [Datasets](#datasets)
- [Project Structure](#project-structure)

---

## System Overview

This platform is divided into two major components:

### 1️⃣ **Frequency Analysis** (Computational Science)
Manual implementations of text processing and statistical algorithms without external computation libraries. Demonstrates fundamental CS concepts:
- Hash map-based frequency counting (O(n) time complexity)
- Bubble sort algorithm for ranking
- Statistical calculations (mean, median, standard deviation)
- Zipf's Law analysis
- Lexical diversity metrics

### 2️⃣ **Text Classification** (Machine Learning)
Supervised learning models trained on ~22,000 labeled documents to categorize text into 10 main categories. Combines manual and library-based implementations:
- **Manual Naive Bayes** - From-scratch implementation with Laplace smoothing
- **Random Forest with TF-IDF** - scikit-learn implementation with 300 decision trees
- Bootstrap confidence intervals for model evaluation
- Automatic training on combined datasets

---

## Part 1: Frequency Analysis (Computational Science)

### Purpose
Analyze text composition by counting word frequencies, calculating statistics, and identifying linguistic patterns using manual algorithm implementations.

### Core Algorithms & Computational Methods

#### 1. Text Preprocessing (`core/text_processor.py`)
**Manual Tokenization Algorithm:**
```
Input: Raw text string
1. Convert to lowercase
2. Split on whitespace → tokens
3. For each token:
   - Remove punctuation (check each character manually)
   - Apply lemmatization rules (manual suffix removal)
4. Filter stop words (if enabled)
Output: List of processed tokens
```

**Complexity:** O(n·m) where n = number of words, m = average word length

**Lemmatization Rules (Manual Implementation):**
- "running" → "run" (remove -ing)
- "studies" → "study" (handle -ies → -y)
- "connected" → "connect" (remove -ed)

#### 2. Frequency Counting (`core/frequency_counter.py`)
**Hash Map-Based Algorithm:**
```python
Algorithm: count_frequencies(tokens)
  Initialize: frequency_map = {} (hash map)

  For each token in tokens:
    if token exists in frequency_map:
      frequency_map[token] += 1
    else:
      frequency_map[token] = 1

  Return frequency_map
```

**Time Complexity:** O(n) - single pass through tokens
**Space Complexity:** O(k) - k unique words
**Data Structure:** Python dictionary (hash map with average O(1) lookup)

#### 3. Sorting Algorithm (`core/frequency_counter.py`)
**Bubble Sort Implementation:**
```python
Algorithm: bubble_sort_frequencies(items, limit)
  For i = 0 to n-1:
    For j = 0 to n-i-2:
      if items[j].frequency < items[j+1].frequency:
        swap(items[j], items[j+1])

  Return top 'limit' items
```

**Time Complexity:** O(n²) - nested loops
**Space Complexity:** O(1) - in-place sorting
**Why Bubble Sort?** Educational demonstration of basic sorting; shows algorithm mechanics

#### 4. Statistical Calculations (`core/frequency_counter.py`)
**Manual Formulas (No NumPy/SciPy):**

**Mean (Average Frequency):**
```
mean = Σ(frequencies) / count(unique_words)
```

**Median (Middle Value):**
```
1. Sort frequencies in ascending order
2. If odd count: median = middle_value
3. If even count: median = (middle1 + middle2) / 2
```

**Standard Deviation:**
```
1. Calculate mean (μ)
2. variance = Σ((xi - μ)²) / n
3. std_dev = √variance
```

**Lexical Diversity (Type-Token Ratio):**
```
diversity = unique_words / total_words
Range: [0, 1], higher = more varied vocabulary
```

#### 5. Zipf's Law Analysis
**Concept:** In natural language, word frequency follows power law distribution:
```
frequency(rank) ∝ 1 / rank
```

**Verification:**
1. Rank words by frequency (1 = most frequent)
2. Plot rank vs. frequency on log-log scale
3. Check if relationship is approximately linear (slope ≈ -1)

**Implementation:** Calculate expected vs. actual frequencies for top N words

### Features & Capabilities
✅ Process text from direct input or file upload (TXT, PDF)
✅ Configurable stop word removal
✅ Optional lemmatization
✅ Top N word extraction (configurable)
✅ Interactive visualizations (bar charts, histograms)
✅ Comprehensive statistical summary
✅ Zipf's Law validation charts

---

## Part 2: Text Classification (Machine Learning)

### Purpose
Automatically categorize text documents into predefined categories using supervised machine learning algorithms trained on labeled datasets.

### Categories (10 Main Classes)
The system classifies text into these categories:
1. **Business** - Economics, finance, markets, corporate news
2. **Entertainment** - Movies, music, celebrities, arts
3. **News** - General news, world events
4. **Politics** - Government, elections, policy, political discussions
5. **Science** - Research, discoveries, scientific topics (physics, chemistry, biology)
6. **Sports** - Athletics, games, teams, competitions
7. **Technology** - Computing, software, hardware, tech industry
8. **Spam** - Unwanted messages, advertisements
9. **Not Spam** - Legitimate messages (ham)
10. **Other** - Miscellaneous topics not fitting above categories

### Machine Learning Models

#### Model 1: Naive Bayes Classifier (Manual Implementation)
**File:** `ml_models/naive_bayes.py`

**Algorithm:** Bayesian probability with Laplace smoothing

**Mathematical Foundation:**
```
P(Category|Document) ∝ P(Category) × P(Document|Category)

Where:
- P(Category) = class_doc_count / total_docs (Prior probability)
- P(Document|Category) = Π P(word_i|Category) for all words (Likelihood)
```

**Laplace Smoothing (Add-1):**
```
P(word|Category) = (word_count_in_category + 1) / (total_words_in_category + vocabulary_size)
```
Prevents zero probability for unseen words.

**Training Process:**
1. Count documents per category → Calculate priors
2. Count word frequencies per category → Build vocabulary
3. Store class-conditional word probabilities

**Prediction Process:**
1. Tokenize input text
2. For each category:
   - Start with log(prior)
   - Add log(P(word|category)) for each word
3. Return category with highest score

**Advantages:**
- Fast training and prediction
- Works well with small datasets
- Interpretable probability scores
- Manual implementation demonstrates ML fundamentals

#### Model 2: Random Forest with TF-IDF (scikit-learn)
**File:** `ml_models/sklearn_models.py`

**Algorithm:** Ensemble of 300 decision trees with TF-IDF vectorization

**TF-IDF (Term Frequency-Inverse Document Frequency):**
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

Where:
- TF(word, doc) = frequency of word in document
- IDF(word) = log(total_docs / docs_containing_word)
```

**Purpose:** Weight words by importance; rare words get higher scores than common ones.

**Vectorizer Configuration:**
- **Max Features:** 5000 (top 5000 most important words)
- **N-grams:** (1, 2) - considers single words and word pairs
- **Stop Words:** Removes English stop words
- **Lowercase:** True

**Random Forest Configuration:**
- **Estimators:** 300 decision trees
- **Class Weight:** Balanced (handles imbalanced categories)
- **Max Depth:** None (trees grow fully)
- **Random State:** 42 (reproducible results)

**How It Works:**
1. **Vectorization:** Convert text → 5000-dimensional TF-IDF vector
2. **Ensemble Training:** Train 300 trees on random data subsets
3. **Prediction:** Each tree votes for a category; majority wins
4. **Probability:** Average of tree predictions

**Advantages:**
- Handles high-dimensional data well
- Robust to overfitting (ensemble averaging)
- Captures complex patterns
- Better accuracy than Naive Bayes on large datasets

### Model Training & Evaluation

**Training Dataset Size:** ~22,162 documents
**Training Method:** Automatic on app startup
**Training Time:** 30-60 seconds (combined datasets)

**Evaluation Metrics:**

**1. Accuracy:**
```
accuracy = correct_predictions / total_predictions
```
Overall percentage of correct classifications.

**2. Precision (Per Category):**
```
precision = true_positives / (true_positives + false_positives)
```
Of predicted category X, how many were actually X?

**3. Recall (Per Category):**
```
recall = true_positives / (true_positives + false_negatives)
```
Of actual category X documents, how many did we find?

**4. F1-Score:**
```
F1 = 2 × (precision × recall) / (precision + recall)
```
Harmonic mean balancing precision and recall.

**5. Bootstrap Confidence Intervals:**
- **Method:** Resample predictions 1000 times
- **Output:** 95% confidence interval for each metric
- **Purpose:** Quantify uncertainty in model performance

**Example:** "Accuracy: 85.3% (CI: 83.1%-87.2%)" means we're 95% confident true accuracy is in that range.

### Features & Capabilities
✅ Two classification models (manual + library-based)
✅ Automatic training on 22k+ documents
✅ 10 consolidated main categories
✅ Probability scores for predictions
✅ Model comparison with evaluation metrics
✅ Bootstrap confidence intervals
✅ Real-time text classification

---

## Datasets

The system uses 5 datasets totaling **~22,162 labeled documents** for training:

### 1. Training Data (Built-in)
**File:** `datasets/training_data.json`
**Samples:** ~100
**Categories:** 5 (Technology, Sports, News, Entertainment, Science)
**Content:** Curated examples covering common text classification scenarios
**Format:** JSON with text-label pairs

### 2. AG News Dataset
**File:** `datasets/AG news.csv`
**Samples:** 10,000
**Original Categories:** World, Sports, Business, Sci/Tech → **Normalized to:** News, Sports, Business, Technology
**Content:** News articles from various domains
**Format:** CSV (label, text)
**Source:** AG's corpus of news articles

### 3. BBC News Dataset
**File:** `datasets/BBC News Train.csv`
**Samples:** 1,492
**Original Categories:** business, entertainment, politics, sport, tech → **Normalized to:** Business, Entertainment, Politics, Sports, Technology
**Content:** BBC news articles from 2004-2005
**Format:** CSV (Text, Category)
**Source:** BBC News website

### 4. SMS Spam Dataset
**File:** `datasets/spam.csv`
**Samples:** 5,575
**Original Categories:** spam, ham → **Normalized to:** Spam, Not Spam
**Content:** SMS messages labeled as spam or legitimate
**Format:** CSV (v1=label, v2=text)
**Source:** UCI Machine Learning Repository

### 5. 20 Newsgroups Dataset
**Files:** `datasets/20news-bydate-train/` (subdirectories)
**Samples:** ~5,000 (subset)
**Original Categories:** 20 specific newsgroups → **Normalized to:** Technology, Sports, Science, Politics, Other
**Category Mapping Examples:**
- `comp.graphics`, `comp.sys.*` → Technology
- `rec.sport.baseball`, `rec.motorcycles` → Sports
- `sci.med`, `sci.space` → Science
- `talk.politics.*` → Politics
- `alt.atheism`, `soc.religion.*` → Other

**Content:** Usenet newsgroup posts from 1990s
**Format:** Individual text files in category subdirectories
**Source:** 20 Newsgroups text classification benchmark

### Category Normalization Strategy
To improve usability and accuracy, the system consolidates **30+ specific categories** into **10 main categories**:

```python
CATEGORY_MAPPING = {
    # Map specific categories to main ones
    'comp.graphics' → 'Technology',
    'rec.sport.baseball' → 'Sports',
    'sci.med' → 'Science',
    'talk.politics.guns' → 'Politics',
    # ... (see csv_dataset_parser.py for full mapping)
}
```

**Benefits:**
- Reduces confusion from overly granular categories
- Improves classification accuracy (more training data per category)
- Better matches user expectations

---

## Installation & Setup

### 1. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- `streamlit >= 1.28.0` - Web interface framework
- `plotly >= 5.17.0` - Interactive visualizations
- `PyPDF2 >= 3.0.0` - PDF file processing
- `scikit-learn >= 1.3.0` - Random Forest & TF-IDF (ML component)

### 3. Add Datasets
Place the following files in `datasets/` folder:
- `AG news.csv`
- `BBC News Train.csv`
- `spam.csv`
- `20news-bydate-train/` (directory with subdirectories)
- `training_data.json` (included)

### 4. Run the Application
```bash
streamlit run main.py
```

The application will:
1. Load all datasets (~22k documents)
2. Train both models automatically (30-60 seconds)
3. Open in your browser at `http://localhost:8501`

### First-Time Setup Note
The first run takes 30-60 seconds to train models on all datasets. Subsequent runs use cached models (unless datasets change).

---

## Usage Guide

### 1. Frequency Analysis & Statistics (Computational Science)
**Page:** "📈 Frequency Analysis & Statistics"

**Steps:**
1. Choose input method:
   - **Paste Text:** Directly enter text
   - **Upload File:** TXT or PDF file
2. Configure options:
   - Remove stop words (optional)
   - Apply lemmatization (optional)
   - Set top N words to display
3. Click "🔍 Analyze Text"
4. Explore results in tabs:
   - **📊 Metrics:** Quick overview of key statistics
   - **📈 Top Words:** Bar chart and frequency table of most frequent words
   - **📉 Distribution:** Histogram showing frequency distribution
   - **🔄 Zipf's Law:** Validation of power law distribution pattern
   - **📖 Advanced Stats:** Comprehensive statistical analysis with lexical diversity

**Key Metrics Explained:**
- **Total Words:** Total number of tokens processed
- **Unique Words:** Number of distinct words
- **Type-Token Ratio (TTR):** Vocabulary diversity (0-1), higher = more varied
- **Mean/Median Frequency:** Average and middle word occurrence counts
- **Standard Deviation:** Variability in word frequencies
- **Lexical Diversity:** Classification of vocabulary richness (Low/Medium/High)

**Use Cases:**
- Analyze document vocabulary composition
- Identify key themes in text
- Verify linguistic patterns (Zipf's Law)
- Educational demonstration of computational algorithms
- Compare vocabulary richness across texts

### 2. Text Classification (Machine Learning)
**Page:** "🤖 Text Classification"

**Steps:**
1. Select model:
   - **Naive Bayes** (manual implementation)
   - **Random Forest** (TF-IDF + sklearn)
2. Enter text to classify
3. View prediction with confidence scores
4. Compare models with evaluation metrics:
   - Accuracy, Precision, Recall, F1
   - Bootstrap 95% confidence intervals

**Use Cases:**
- Categorize news articles
- Spam detection
- Topic identification
- Model comparison and evaluation

---

## Project Structure
```
word_frequency_counter/
├── main.py                          # Streamlit app entry point with 2 pages
├── requirements.txt                 # Python dependencies
├── README.md                        # This documentation
│
├── core/                            # COMPUTATIONAL SCIENCE (Manual Implementations)
│   ├── __init__.py
│   ├── text_processor.py           # Manual tokenization, lemmatization, stop word removal
│   ├── frequency_counter.py        # Hash map frequency counting, bubble sort, statistics
│   └── statistics_calculator.py    # Advanced statistical calculations
│
├── ml_models/                       # MACHINE LEARNING (Classification Models)
│   ├── __init__.py
│   ├── naive_bayes.py              # Manual Naive Bayes with Laplace smoothing
│   └── sklearn_models.py           # Random Forest with TF-IDF (scikit-learn)
│
├── utils/                           # Utilities & Data Management
│   ├── __init__.py
│   ├── dataset_loader.py           # Load and combine all datasets
│   ├── csv_dataset_parser.py       # Parse CSV/TXT datasets + category normalization
│   ├── file_handler.py             # PDF/TXT file reading
│   └── visualization.py            # Plotly chart generation
│
└── datasets/                        # Training Data (~22k documents)
    ├── training_data.json          # Built-in examples (100 samples)
    ├── AG news.csv                 # News articles (10,000 samples)
    ├── BBC News Train.csv          # BBC news (1,492 samples)
    ├── spam.csv                    # SMS spam/ham (5,575 samples)
    └── 20news-bydate-train/        # Newsgroups posts (~5,000 samples)
        ├── alt.atheism/
        ├── comp.graphics/
        ├── sci.med/
        └── ... (20 subdirectories)
```

### Page Architecture

**Page 1: Frequency Analysis & Statistics (Combined)**
- **Purpose:** Unified computational science interface combining word frequency analysis with advanced statistics
- **Sections:**
  - Input & Processing Options
  - 📊 Metrics Tab - Quick statistics overview
  - 📈 Top Words Tab - Bar chart and frequency table
  - 📉 Distribution Tab - Frequency distribution histogram
  - 🔄 Zipf's Law Tab - Power law analysis
  - 📖 Advanced Stats Tab - Comprehensive statistical analysis with lexical diversity
- **Algorithms:** O(n) frequency counting, O(n²) bubble sort, manual statistics
- **Key Function:** `page_frequency_analysis()` - Combines previous Frequency Analysis and Advanced Statistics pages

**Page 2: Text Classification**
- **Purpose:** Machine learning model comparison and evaluation
- **Features:** Model selection, prediction with confidence scores, evaluation metrics with bootstrap CIs
- **Models:** Naive Bayes (manual) + Random Forest (TF-IDF + scikit-learn)
- **Key Function:** `page_text_classification()` - Pre-trained on 22k+ documents, instant classification

### Key Files Explained

#### `main.py`
- **Purpose:** Streamlit web application entry point
- **Pages:** 3 main pages (Frequency Analysis, Text Classification, Advanced Statistics)
- **Key Function:** `load_models_and_data()` - Pre-trains both ML models on startup using `@st.cache_resource` decorator

#### `core/frequency_counter.py`
- **Purpose:** Computational science implementations
- **Algorithms:** Hash map frequency counting (O(n)), bubble sort (O(n²)), manual statistics
- **Key Methods:**
  - `count_frequencies()` - O(n) word counting
  - `get_top_words()` - Sorted ranking with bubble sort
  - `get_statistics()` - Mean, median, std dev calculations

#### `ml_models/naive_bayes.py`
- **Purpose:** Manual ML implementation for educational purposes
- **Algorithm:** Bayesian probability with Laplace smoothing
- **Key Methods:**
  - `train()` - Build class priors and word probabilities
  - `predict()` - Classify text using Bayes theorem

#### `ml_models/sklearn_models.py`
- **Purpose:** Production-grade ML model using scikit-learn
- **Components:** TF-IDF vectorizer (5000 features) + Random Forest (300 trees)
- **Key Methods:**
  - `fit()` - Train on text-label pairs
  - `predict_proba()` - Return probability distribution

#### `utils/csv_dataset_parser.py`
- **Purpose:** Parse diverse dataset formats and normalize categories
- **Key Feature:** `CATEGORY_MAPPING` - Consolidates 30+ categories to 10 main ones
- **Parsers:** `parse_ag_news()`, `parse_bbc_news()`, `parse_spam()`, `parse_20_newsgroups()`

#### `utils/dataset_loader.py`
- **Purpose:** Centralized dataset management
- **Key Functions:**
  - `get_all_documents_combined()` - Returns ~22k (tokens, label) tuples
  - `get_all_texts_and_labels_combined()` - For Random Forest training
  - `get_all_categories_combined()` - Returns 10 main categories

---

## Technical Details

### Performance Characteristics

**Frequency Analysis:**
- **Time Complexity:** O(n·m + k²) where n=words, m=avg word length, k=unique words
- **Space Complexity:** O(k) for frequency hash map
- **Bottleneck:** Bubble sort for ranking (O(k²))

**Text Classification:**
- **Training Time:** 30-60 seconds for ~22k documents
- **Prediction Time:**
  - Naive Bayes: <10ms per document
  - Random Forest: <50ms per document (TF-IDF transformation + tree voting)
- **Model Size:**
  - Naive Bayes: ~2-5 MB (word probability tables)
  - Random Forest: ~50-100 MB (300 decision trees)

### Computational Trade-offs

**Bubble Sort vs. Built-in Sort:**
- Bubble sort used for educational demonstration (shows algorithm mechanics)
- For production: Python's Timsort (O(n log n)) would be faster
- Current implementation: O(k²) where k ≈ 1000-5000 unique words

**Manual Naive Bayes vs. scikit-learn:**
- Manual version demonstrates Bayesian probability concepts
- scikit-learn version (not used) would have optimizations
- Performance difference negligible for this use case

**TF-IDF Dimensionality:**
- 5000 features captures most important words
- Trade-off: Higher features = more accuracy but slower training/prediction
- Current setting balances performance and accuracy

### Why Two Models?

**Naive Bayes (Educational):**
- Simple, interpretable algorithm
- Demonstrates core ML concepts
- Shows manual implementation approach
- Fast training and prediction

**Random Forest (Practical):**
- Higher accuracy on complex patterns
- Industry-standard approach
- Handles high-dimensional TF-IDF vectors well
- Better generalization on unseen text

**Model Comparison Results:**
- Random Forest typically achieves 3-7% higher accuracy
- Naive Bayes excels with limited training data
- Both provide confidence scores for predictions

---

## Educational Value

### Computational Science Concepts Demonstrated
✅ **Hash Maps:** Efficient key-value storage for frequency counting
✅ **Sorting Algorithms:** Bubble sort implementation and complexity analysis
✅ **Algorithm Complexity:** O(n), O(n²) time complexity analysis
✅ **Statistical Calculations:** Manual mean, median, standard deviation
✅ **Text Processing:** Tokenization, lemmatization without libraries

### Machine Learning Concepts Demonstrated
✅ **Supervised Learning:** Training on labeled data
✅ **Bayesian Probability:** Naive Bayes theorem application
✅ **Feature Engineering:** TF-IDF vectorization
✅ **Ensemble Methods:** Random Forest decision tree voting
✅ **Model Evaluation:** Accuracy, precision, recall, F1 scores
✅ **Confidence Intervals:** Bootstrap resampling for uncertainty quantification

### Learning Outcomes
Students/users will understand:
1. How text is processed computationally (tokenization → frequency counting)
2. Time/space complexity trade-offs in algorithm design
3. Difference between manual and library-based implementations
4. How ML models learn patterns from labeled data
5. How to evaluate and compare model performance
6. Practical applications of NLP and text classification

---

## Limitations & Future Enhancements

### Current Limitations
- Bubble sort inefficient for large vocabularies (use O(n log n) sort in production)
- English-only text processing (lemmatization rules)
- No GPU acceleration for ML training
- Models retrain on each app restart (no persistence)
- Category normalization hardcoded (not configurable)

### Planned Enhancements
🔄 **Performance:**
- Implement quicksort or mergesort for better O(n log n) complexity
- Add model save/load for persistence across sessions
- GPU acceleration for Random Forest training

🔄 **Features:**
- Multi-language support (Spanish, French, etc.)
- Additional file formats (DOCX, EPUB)
- Sentiment analysis (positive/negative/neutral)
- Named entity recognition (people, places, organizations)
- Custom category training interface

🔄 **ML Models:**
- Support Vector Machines (SVM)
- Deep learning models (LSTM, Transformers)
- Active learning for dataset expansion
- Model explainability (LIME, SHAP)

🔄 **Datasets:**
- Configurable category mapping
- Dataset versioning and tracking
- Automatic dataset download from Kaggle API
- Support for custom dataset formats

---

## Requirements

**Python Version:** 3.8+

**Core Dependencies:**
```
streamlit >= 1.28.0      # Web UI framework
plotly >= 5.17.0         # Interactive visualizations
PyPDF2 >= 3.0.0          # PDF processing
scikit-learn >= 1.3.0    # ML models (Random Forest, TF-IDF)
```

**System Requirements:**
- **RAM:** 2 GB minimum (4 GB recommended for large datasets)
- **Storage:** 500 MB for datasets + models
- **CPU:** Multi-core recommended for Random Forest training

---

## License & Usage

This is an **educational project** demonstrating computational science and machine learning concepts.

**Free to use for:**
✅ Learning and educational purposes
✅ Academic projects and assignments
✅ Teaching computational algorithms and ML
✅ Personal experimentation and modification

**Attribution:** If you use this project in academic work, please cite appropriately.

---

## Acknowledgments

**Datasets:**
- AG News Corpus
- BBC News Dataset (2004-2005)
- SMS Spam Collection (UCI ML Repository)
- 20 Newsgroups (Classic text classification benchmark)

**Technologies:**
- Streamlit (UI framework)
- Plotly (Visualizations)
- scikit-learn (ML library)

---

## Support & Contact

For questions, issues, or contributions:
- Review code documentation in source files
- Check inline comments for algorithm explanations
- Experiment with different parameters and datasets

**Happy analyzing! 🚀📊**
