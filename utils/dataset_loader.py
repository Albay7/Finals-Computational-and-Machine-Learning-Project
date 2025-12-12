import json
import os
from typing import List, Tuple, Dict
from pathlib import Path
from utils.csv_dataset_parser import CSVDatasetParser

class DatasetLoader:
    """Load and manage datasets for training models"""

    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)
        self.datasets = {}
        self.csv_parser = CSVDatasetParser(base_path)
        self.load_all_datasets()

    def load_all_datasets(self) -> None:
        """Load all available datasets"""
        # Load built-in training data
        self._load_builtin_dataset()

        # Check for Kaggle datasets
        self._check_kaggle_datasets()

        # Load CSV/TXT datasets
        self._load_csv_datasets()

    def _load_builtin_dataset(self) -> None:
        """Load built-in training dataset"""
        training_file = self.base_path / "training_data.json"

        if training_file.exists():
            try:
                with open(training_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.datasets['builtin'] = {
                        'name': 'Built-in Dataset',
                        'samples': data.get('data', []),
                        'categories': data.get('metadata', {}).get('categories', []),
                        'size': len(data.get('data', []))
                    }
            except Exception as e:
                print(f"Error loading built-in dataset: {e}")

    def _check_kaggle_datasets(self) -> None:
        """Check for downloaded Kaggle datasets"""
        kaggle_path = self.base_path / "kaggle"

        if not kaggle_path.exists():
            return

        # Check for BBC News dataset
        bbc_file = kaggle_path / "bbc_news.json"
        if bbc_file.exists():
            self._load_kaggle_dataset('bbc', bbc_file, 'BBC News Dataset')

        # Check for SMS Spam dataset
        spam_file = kaggle_path / "sms_spam.json"
        if spam_file.exists():
            self._load_kaggle_dataset('spam', spam_file, 'SMS Spam Dataset')

        # Check for 20 Newsgroups dataset
        newsgroups_file = kaggle_path / "newsgroups.json"
        if newsgroups_file.exists():
            self._load_kaggle_dataset('newsgroups', newsgroups_file, '20 Newsgroups Dataset')

    def _load_kaggle_dataset(self, key: str, filepath: Path, name: str) -> None:
        """Load a Kaggle dataset"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.datasets[key] = {
                    'name': name,
                    'samples': data.get('data', []),
                    'categories': data.get('categories', []),
                    'size': len(data.get('data', []))
                }
        except Exception as e:
            print(f"Error loading {name}: {e}")

    def get_documents_for_classification(self, dataset_name: str = 'builtin') -> List[Tuple[List[str], str]]:
        """Get documents formatted for classification training"""
        if dataset_name not in self.datasets:
            dataset_name = 'builtin'

        dataset = self.datasets.get(dataset_name, {})
        samples = dataset.get('samples', [])

        # Import here to avoid circular imports
        from core.text_processor import TextProcessor
        processor = TextProcessor()

        documents = []
        for sample in samples:
            text = sample.get('text', '')
            category = sample.get('category', '')
            if text and category:
                tokens = processor.preprocess(text, remove_stopwords=True)
                documents.append((tokens, category))

        return documents

    def get_documents_for_clustering(self, dataset_name: str = 'builtin', limit: int = None) -> List[str]:
        """Get documents for clustering"""
        if dataset_name not in self.datasets:
            dataset_name = 'builtin'

        dataset = self.datasets.get(dataset_name, {})
        samples = dataset.get('samples', [])

        documents = [sample.get('text', '') for sample in samples if sample.get('text')]

        if limit:
            documents = documents[:limit]

        return documents

    def get_dataset_info(self) -> Dict:
        """Get information about all available datasets"""
        info = {}
        for key, dataset in self.datasets.items():
            info[key] = {
                'name': dataset.get('name'),
                'samples': dataset.get('size'),
                'categories': dataset.get('categories', [])
            }
        return info

    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset keys"""
        return list(self.datasets.keys())

    def get_categories(self, dataset_name: str = 'builtin') -> List[str]:
        """Get categories for a dataset"""
        if dataset_name not in self.datasets:
            dataset_name = 'builtin'

        return self.datasets.get(dataset_name, {}).get('categories', [])

    def _load_csv_datasets(self) -> None:
        """Load datasets from CSV and TXT files"""
        available = self.csv_parser.get_all_available_datasets()

        dataset_map = {
            'ag_news': ('ag_news', 'AG News Dataset', 10000),  # Limit to 10k for performance
            'bbc': ('bbc_news', 'BBC News Dataset', None),
            'spam': ('spam', 'SMS Spam Dataset', None),
            'newsgroups': ('newsgroups', '20 Newsgroups Dataset', 5000)
        }

        for key, (dataset_key, name, limit) in dataset_map.items():
            if available.get(key, False):
                try:
                    documents, categories = self.csv_parser.get_dataset(key, sample_limit=limit)
                    if documents:
                        self.datasets[dataset_key] = {
                            'name': name,
                            'samples': [{'text': text, 'category': cat} for text, cat in documents],
                            'categories': categories,
                            'size': len(documents)
                        }
                        print(f"Loaded {name}: {len(documents)} samples, {len(categories)} categories")
                except Exception as e:
                    print(f"Error loading {name}: {e}")

    def get_raw_texts_and_labels(self, dataset_name: str = 'builtin') -> Tuple[List[str], List[str]]:
        """Get raw texts and labels (for Random Forest training)"""
        if dataset_name not in self.datasets:
            dataset_name = 'builtin'

        dataset = self.datasets.get(dataset_name, {})
        samples = dataset.get('samples', [])

        texts = [sample.get('text', '') for sample in samples if sample.get('text')]
        labels = [sample.get('category', '') for sample in samples if sample.get('category')]

        return texts, labels

    def get_all_documents_combined(self) -> List[Tuple[List[str], str]]:
        """Get all documents from all datasets combined (for unified training)"""
        from core.text_processor import TextProcessor
        processor = TextProcessor()

        all_documents = []
        for dataset_key in self.datasets.keys():
            samples = self.datasets[dataset_key].get('samples', [])
            for sample in samples:
                text = sample.get('text', '')
                category = sample.get('category', '')
                if text and category:
                    tokens = processor.preprocess(text, remove_stopwords=True)
                    all_documents.append((tokens, category))

        return all_documents

    def get_all_texts_and_labels_combined(self) -> Tuple[List[str], List[str]]:
        """Get all raw texts and labels combined (for Random Forest training)"""
        all_texts = []
        all_labels = []

        for dataset_key in self.datasets.keys():
            samples = self.datasets[dataset_key].get('samples', [])
            for sample in samples:
                text = sample.get('text', '')
                category = sample.get('category', '')
                if text and category:
                    all_texts.append(text)
                    all_labels.append(category)

        return all_texts, all_labels

    def get_all_categories_combined(self) -> List[str]:
        """Get all unique categories across all datasets"""
        all_categories = set()
        for dataset_key in self.datasets.keys():
            categories = self.datasets[dataset_key].get('categories', [])
            all_categories.update(categories)
        return sorted(list(all_categories))
