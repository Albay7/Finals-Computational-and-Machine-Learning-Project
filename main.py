import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Dict
from core.text_processor import TextProcessor
from core.frequency_counter import FrequencyCounter
from ml_models.naive_bayes import NaiveBayesTextClassifier
from collections import Counter
try:
    from ml_models.sklearn_models import RandomForestText, AVAILABLE as SKLEARN_AVAILABLE
except ImportError:
    RandomForestText = None
    SKLEARN_AVAILABLE = False
from utils.file_handler import FileHandler
from utils.dataset_loader import DatasetLoader
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Word Frequency Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header { font-size: 3em; color: #1f77b4; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 1.5em; border-radius: 10px; }
    .stat-number { font-size: 2em; font-weight: bold; color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# Initialize dataset loader and pre-trained models
@st.cache_resource
def load_models_and_data():
    """Load datasets and pre-train baseline and main models on app startup"""
    dataset_loader = DatasetLoader()

    # Prepare combined documents (tokens, label)
    documents = dataset_loader.get_all_documents_combined()

    # Build category frequency profiles for explainability
    category_profiles: Dict[str, Dict] = {}
    overall_counter = Counter()
    for tokens, label in documents:
        cat_counter = category_profiles.setdefault(label, Counter())
        cat_counter.update(tokens)
        overall_counter.update(tokens)

    category_profiles = {
        label: {
            "counter": counter,
            "total": sum(counter.values())
        }
        for label, counter in category_profiles.items()
    }
    overall_profile = {"counter": overall_counter, "total": sum(overall_counter.values())}

    # Train Naive Bayes on ALL datasets combined
    nb_classifier = NaiveBayesTextClassifier()
    if documents:
        nb_classifier.train(documents)
        print(f"Trained Naive Bayes on {len(documents)} documents from all datasets")

    # Train Random Forest on ALL datasets combined (if available)
    rf_classifier = None
    if SKLEARN_AVAILABLE and RandomForestText is not None:
        try:
            texts, labels = dataset_loader.get_all_texts_and_labels_combined()
            if texts and labels:
                rf_classifier = RandomForestText()
                rf_classifier.fit(texts, labels)
                print(f"Trained Random Forest on {len(texts)} samples from all datasets")
        except Exception as e:
            print(f"Random Forest training failed: {e}")
            rf_classifier = None

    return dataset_loader, nb_classifier, rf_classifier, category_profiles, overall_profile

# Load models on startup
dataset_loader, pre_trained_nb, pre_trained_rf, category_profiles, overall_profile = load_models_and_data()

def main():
    st.markdown('<h1 class="main-header">Final Project for Computational Science and Machine Learning</h1>', unsafe_allow_html=True)
    st.write("Advanced text analysis with machine learning capabilities")

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Select Feature", [
            "📈 Frequency Analysis",
            "🤖 Text Classification"
        ])

    # Initialize session state
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    if 'tokens' not in st.session_state:
        st.session_state.tokens = []

    # Page routing
    if page == "📈 Frequency Analysis":
        page_frequency_analysis()
    elif page == "🤖 Text Classification":
        page_text_classification()

def page_frequency_analysis():
    st.header("📈 Frequency Analysis & Statistics")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Input method selection
        input_method = st.radio("Select input method:", ["Text Input", "Upload File"], horizontal=True)

        text_input = ""
        if input_method == "Text Input":
            text_input = st.text_area(
                "Paste your text here:",
                height=300,
                placeholder="Enter text for analysis..."
            )
        else:
            uploaded_file = st.file_uploader("Upload text file", type=['txt', 'pdf'])
            if uploaded_file:
                file_handler = FileHandler()
                text_input = file_handler.read_file(uploaded_file)
                st.success("File uploaded successfully!")

    with col2:
        st.subheader("Processing Options")
        remove_stopwords = st.checkbox("Remove Stop Words", value=True)
        top_n = st.slider("Top N Words", 5, 50, 20)
        apply_lemmatization = st.checkbox("Apply Lemmatization", value=False)

    if st.button("🔍 Analyze Text", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text!")
            return

        # Process text
        processor = TextProcessor()
        tokens = processor.preprocess(text_input, remove_stopwords)

        if apply_lemmatization:
            tokens = [processor.lemmatize_manual(token) for token in tokens]

        # Count frequencies
        counter = FrequencyCounter()
        frequency_map = counter.count_frequencies(tokens)
        top_words = counter.get_top_words(top_n)
        stats = counter.get_statistics()

        # Display results
        st.success(f"Analysis complete! Processed {stats['total_words']} tokens")

        st.divider()

        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Metrics",
            "📈 Top Words",
            "📉 Distribution",
            "🔄 Zipf's Law",
            "📖 Advanced Stats"
        ])

        # Tab 1: Quick Metrics
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Words", stats['total_words'])
            with col2:
                st.metric("Unique Words", stats['unique_words'])
            with col3:
                st.metric("Type-Token Ratio", f"{stats['type_token_ratio']:.3f}")
            with col4:
                st.metric("Mean Frequency", f"{stats['mean_frequency']:.2f}")

        # Tab 2: Bar Chart & Frequency Table
        with tab2:
            st.subheader("Top Words by Frequency")
            words, freqs = zip(*top_words)
            fig = go.Figure(data=[
                go.Bar(x=list(words), y=list(freqs), marker_color='#1f77b4')
            ])
            fig.update_layout(
                title="Top Words by Frequency",
                xaxis_title="Words",
                yaxis_title="Frequency",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Frequency Table")
            # Create frequency table
            freq_data = []
            for rank, (word, freq) in enumerate(top_words, 1):
                freq_data.append({
                    "Rank": rank,
                    "Word": word,
                    "Frequency": freq,
                    "Percentage": f"{(freq/stats['total_words']*100):.2f}%"
                })

            st.dataframe(freq_data, use_container_width=True, hide_index=True)

        # Tab 3: Distribution
        with tab3:
            st.subheader("Frequency Distribution")
            # Histogram of frequency distribution
            all_freqs = [f for _, f in frequency_map.items()]
            fig = go.Figure(data=[
                go.Histogram(x=all_freqs, nbinsx=30, marker_color='#ff7f0e')
            ])
            fig.update_layout(
                title="Frequency Distribution",
                xaxis_title="Frequency",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Tab 4: Zipf's Law
        with tab4:
            st.subheader("Zipf's Law Analysis")
            # Zipf's Law visualization
            zipf_data = counter.zipf_law_analysis()

            if len(zipf_data) > 0:
                ranks = [d['rank'] for d in zipf_data[:top_n]]
                freqs = [d['frequency'] for d in zipf_data[:top_n]]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ranks, y=freqs, mode='lines+markers',
                    name='Actual', marker_color='#1f77b4'
                ))
                fig.add_trace(go.Scatter(
                    x=ranks, y=[1/r for r in ranks],
                    mode='lines', name='Zipf Expected',
                    line=dict(dash='dash', color='#d62728')
                ))

                fig.update_layout(
                    title="Zipf's Law Analysis",
                    xaxis_title="Rank",
                    yaxis_title="Frequency",
                    height=400,
                    yaxis_type="log",
                    xaxis_type="log"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Zipf's Law** states that word frequency follows a power law distribution. "
                         "The dashed line shows the expected distribution if Zipf's Law holds perfectly.")

        # Tab 5: Advanced Statistics
        with tab5:
            st.subheader("Comprehensive Statistical Analysis")
            
            # Detailed metrics in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Word Count", stats['total_words'])
                st.metric("Unique Words", stats['unique_words'])
                st.metric("Maximum Frequency", stats['max_frequency'])

            with col2:
                st.metric("Mean Frequency", f"{stats['mean_frequency']:.2f}")
                st.metric("Median Frequency", f"{stats['median_frequency']:.2f}")
                st.metric("Min Frequency", stats['min_frequency'])

            with col3:
                st.metric("Std Deviation", f"{stats['std_deviation']:.2f}")
                st.metric("Type-Token Ratio", f"{stats['type_token_ratio']:.4f}")
                st.metric("Vocabulary Richness", f"{(stats['type_token_ratio']*100):.2f}%")

            st.divider()

            # Lexical diversity explanation
            st.subheader("📖 Lexical Diversity Analysis")
            ttr = stats['type_token_ratio']

            if ttr < 0.3:
                diversity_level = "Low (Repetitive)"
                diversity_color = "🔴"
                explanation = "Your text uses a limited vocabulary with many word repetitions. This can indicate specialized content or narrow focus."
            elif ttr < 0.6:
                diversity_level = "Medium (Standard)"
                diversity_color = "🟡"
                explanation = "Your text shows a balanced vocabulary. This is typical for most written content."
            else:
                diversity_level = "High (Varied)"
                diversity_color = "🟢"
                explanation = "Your text uses a rich and diverse vocabulary. This suggests sophisticated or varied content."

            st.write(f"{diversity_color} **Diversity Level**: {diversity_level}")
            st.write(f"Type-Token Ratio: **{ttr:.4f}** ({ttr*100:.2f}% unique words)")
            st.write(f"**Interpretation**: {explanation}")

            st.divider()

            # Statistical summary
            st.subheader("📊 Statistical Summary")
            st.write(f"""
            **Text Composition:**
            - Total tokens processed: **{stats['total_words']}**
            - Unique tokens: **{stats['unique_words']}**
            - Vocabulary coverage: **{(stats['unique_words']/stats['total_words']*100):.2f}%**

            **Frequency Statistics:**
            - Most frequent word appears: **{stats['max_frequency']}** times
            - Least frequent word appears: **{stats['min_frequency']}** time(s)
            - Average word frequency: **{stats['mean_frequency']:.2f}** occurrences
            - Median frequency: **{stats['median_frequency']:.2f}** (middle value)
            - Standard deviation: **{stats['std_deviation']:.2f}** (variability in word frequencies)

            **Linguistic Insights:**
            - **Type-Token Ratio (TTR)**: {ttr:.4f} - Measures vocabulary diversity (0-1 range)
            - **Vocabulary Richness**: {(ttr*100):.2f}% - Higher values indicate more diverse vocabulary
            """)

def page_text_classification():
    st.header("🤖 Text Classification")

    processor = TextProcessor()

    # Show unified model info
    dataset_info = dataset_loader.get_dataset_info()
    all_categories = dataset_loader.get_all_categories_combined()
    total_samples = sum(info.get('samples', 0) for info in dataset_info.values())

    st.info(f"✅ Models trained on {total_samples:,} samples from {len(dataset_info)} datasets")

    st.subheader("📊 Model Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", f"{total_samples:,}")
    with col2:
        st.metric("Categories", len(all_categories))
    with col3:
        st.metric("Datasets Used", len(dataset_info))

    with st.expander("📋 View All Categories"):
        st.write(", ".join(all_categories))

    st.divider()

    model_options = ["Naive Bayes (Baseline)"]
    if pre_trained_rf is not None:
        model_options.append("Random Forest (Main)")
    model_choice = st.selectbox(
        "Choose model:",
        model_options
    )

    # Classification
    st.subheader("Classify Text")
    test_text = st.text_area(
        "Enter text to classify:",
        height=150,
        placeholder="Type or paste text here..."
    )

    if st.button("🔍 Classify Text", use_container_width=True):
        if not test_text.strip():
            st.warning("Please enter some text!")
        else:
            from collections import Counter  # ensure availability in this scope
            # Preprocess once for explanations
            tokens_processed = processor.preprocess(test_text)

            if model_choice.startswith("Naive Bayes"):
                predicted_class, scores = pre_trained_nb.predict(tokens_processed)
            elif pre_trained_rf is not None:
                predicted_class = pre_trained_rf.predict(test_text)
                scores = pre_trained_rf.predict_proba(test_text)
            else:
                st.warning("Random Forest is unavailable. Install scikit-learn in the venv and restart.")
                return

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction Result")
                st.success(f"**{predicted_class}** 🎯")

            with col2:
                st.subheader("Confidence Scores")

                # Create confidence chart
                categories_for_chart = list(scores.keys())
                max_score = max(scores.values()) if scores else 1.0
                min_score = min(scores.values()) if scores else 0.0
                normalized_scores = [
                    (scores[cat] - min_score) / (max_score - min_score) * 100
                    if max_score != min_score else 50
                    for cat in categories_for_chart
                ]

                fig = go.Figure(data=[
                    go.Bar(x=categories_for_chart, y=normalized_scores, marker_color='#2ca02c')
                ])
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Explainability section
            st.subheader("Why this prediction?")

            cat_profile = category_profiles.get(predicted_class)
            if cat_profile:
                cat_counter = cat_profile["counter"]
                cat_total = cat_profile["total"] or 1
                overall_counter = overall_profile["counter"]
                overall_total = overall_profile["total"] or 1

                input_counter = Counter(tokens_processed)

                evidence = []
                for word, count in input_counter.most_common(20):
                    cat_freq = cat_counter.get(word, 0)
                    other_freq = overall_counter.get(word, 0) - cat_freq
                    other_total = max(overall_total - cat_total, 1)

                    cat_rel = cat_freq / cat_total
                    other_rel = other_freq / other_total
                    lift = (cat_rel + 1e-9) / (other_rel + 1e-9)

                    evidence.append({
                        "Word": word,
                        "Count in text": count,
                        f"Freq in {predicted_class}%": round(cat_rel * 100, 3),
                        "Freq in others%": round(other_rel * 100, 3),
                        "Lift": round(lift, 2)
                    })

                # Sort by lift then by count
                evidence = sorted(evidence, key=lambda x: (x["Lift"], x["Count in text"]), reverse=True)
                top_evidence = evidence[:8]

                st.write("These words are much more common in the predicted category than in others (high lift):")
                st.dataframe(top_evidence, use_container_width=True, hide_index=True)
            else:
                st.write("No category profile available to explain this prediction.")

    st.divider()
    st.subheader("📑 Evaluate Model Performance")

    col_eval1, col_eval2 = st.columns([1,2])
    with col_eval1:
        eval_model_options = ["Naive Bayes (Baseline)"]
        if pre_trained_rf is not None:
            eval_model_options.append("Random Forest (Main)")
        eval_model_choice = st.selectbox(
            "Model to evaluate:",
            eval_model_options,
            key="eval_model_choice"
        )
        n_boot = st.slider("Bootstrap samples", 100, 1000, 200, step=50)
        sample_size = st.slider("Evaluation samples", 1000, 22000, 5000, step=1000, 
                                help="Use fewer samples for faster evaluation")

    with col_eval2:
        pass

    if st.button("📊 Run Evaluation", use_container_width=True):
        # Prepare combined dataset
        docs = dataset_loader.get_all_documents_combined()
        
        # Sample subset for faster evaluation
        import random
        if len(docs) > sample_size:
            docs = random.sample(docs, sample_size)
        
        texts = [" ".join(tokens) for tokens, label in docs]
        labels = [label for tokens, label in docs]

        if not texts:
            st.warning("No datasets found for evaluation.")
            return

        st.info(f"Running evaluation on {len(texts):,} samples...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Compute predictions once (20% of progress)
        status_text.text("Step 1/2: Computing predictions...")
        if eval_model_choice.startswith("Naive Bayes"):
            y_pred = []
            for i, t in enumerate(texts):
                tokens = processor.preprocess(t)
                p, s = pre_trained_nb.predict(tokens)
                y_pred.append(p)
                if i % 100 == 0:
                    progress_bar.progress(min(0.2, i / len(texts) * 0.2))
        elif pre_trained_rf is not None:
            y_pred = []
            for i, t in enumerate(texts):
                y_pred.append(pre_trained_rf.predict(t))
                if i % 100 == 0:
                    progress_bar.progress(min(0.2, i / len(texts) * 0.2))
        else:
            st.warning("Random Forest unavailable. Install scikit-learn and restart.")
            return
        
        progress_bar.progress(0.2)
        status_text.text("Step 2/2: Computing bootstrap confidence intervals...")

        # Compute metrics
        import math
        from collections import Counter

        classes = sorted(set(labels))

        def compute_metrics(y_true, y_pred):
            total = len(y_true)
            correct = sum(1 for a,b in zip(y_true, y_pred) if a==b)
            acc = correct/total if total else 0.0

            # per-class
            precision_list = []
            recall_list = []
            f1_list = []
            for c in classes:
                tp = sum(1 for a,b in zip(y_true, y_pred) if a==c and b==c)
                fp = sum(1 for a,b in zip(y_true, y_pred) if a!=c and b==c)
                fn = sum(1 for a,b in zip(y_true, y_pred) if a==c and b!=c)
                prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
                rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
                f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)

            # macro averages
            precision = sum(precision_list)/len(classes) if classes else 0.0
            recall = sum(recall_list)/len(classes) if classes else 0.0
            f1 = sum(f1_list)/len(classes) if classes else 0.0
            return acc, precision, recall, f1

        acc, precision, recall, f1 = compute_metrics(labels, y_pred)

        # Bootstrap CIs (now only resampling indices, not re-predicting)
        metrics_boot = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        idx = list(range(len(labels)))
        for boot_i in range(n_boot):
            sample_idx = [random.choice(idx) for __ in idx]
            y_true_s = [labels[i] for i in sample_idx]
            y_pred_s = [y_pred[i] for i in sample_idx]
            a, p, r, f = compute_metrics(y_true_s, y_pred_s)
            metrics_boot["accuracy"].append(a)
            metrics_boot["precision"].append(p)
            metrics_boot["recall"].append(r)
            metrics_boot["f1"].append(f)
            
            # Update progress bar (remaining 80%)
            if boot_i % 10 == 0:
                progress_bar.progress(0.2 + (boot_i / n_boot * 0.8))
        
        progress_bar.progress(1.0)
        status_text.text("✅ Evaluation complete!")
        
        # Clear progress indicators after a moment
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        def ci95(values):
            vals = sorted(values)
            low_idx = max(0, int(0.025*len(vals))-1)
            high_idx = min(len(vals)-1, int(0.975*len(vals))-1)
            return vals[low_idx], vals[high_idx]

        acc_ci = ci95(metrics_boot["accuracy"]) if metrics_boot["accuracy"] else (acc, acc)
        prec_ci = ci95(metrics_boot["precision"]) if metrics_boot["precision"] else (precision, precision)
        rec_ci = ci95(metrics_boot["recall"]) if metrics_boot["recall"] else (recall, recall)
        f1_ci = ci95(metrics_boot["f1"]) if metrics_boot["f1"] else (f1, f1)

        # Display metrics table
        table_rows = [
            {"Metric": "Accuracy", "Value": f"{acc:.3f}", "95% CI": f"[{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]"},
            {"Metric": "Precision (macro)", "Value": f"{precision:.3f}", "95% CI": f"[{prec_ci[0]:.3f}, {prec_ci[1]:.3f}]"},
            {"Metric": "Recall (macro)", "Value": f"{recall:.3f}", "95% CI": f"[{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]"},
            {"Metric": "F1 (macro)", "Value": f"{f1:.3f}", "95% CI": f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"},
        ]

        st.table(table_rows)


if __name__ == "__main__":
    main()
