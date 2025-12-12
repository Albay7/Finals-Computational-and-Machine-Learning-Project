"""Parse CSV and TXT datasets into training format"""
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict

class CSVDatasetParser:
    """Parse various dataset formats (CSV, TXT) for training"""

    # Category mapping: consolidate specific categories into main ones
    CATEGORY_MAPPING = {
        # AG News & BBC categories
        'World': 'News',
        'world': 'News',
        'News': 'News',
        'Business': 'Business',
        'business': 'Business',
        'Sci/Tech': 'Technology',
        'Science': 'Science',
        'Sports': 'Sports',
        'sport': 'Sports',
        'sports': 'Sports',
        'Technology': 'Technology',
        'tech': 'Technology',
        'Entertainment': 'Entertainment',
        'entertainment': 'Entertainment',
        'Politics': 'Politics',
        'politics': 'Politics',

        # Spam dataset
        'spam': 'Spam',
        'ham': 'Not Spam',

        # 20 Newsgroups - map to main categories
        'alt.atheism': 'Other',
        'comp.graphics': 'Technology',
        'comp.os.ms-windows.misc': 'Technology',
        'comp.sys.ibm.pc.hardware': 'Technology',
        'comp.sys.mac.hardware': 'Technology',
        'comp.windows.x': 'Technology',
        'misc.forsale': 'Other',
        'rec.autos': 'Sports',
        'rec.motorcycles': 'Sports',
        'rec.sport.baseball': 'Sports',
        'rec.sport.hockey': 'Sports',
        'sci.crypt': 'Science',
        'sci.electronics': 'Science',
        'sci.med': 'Science',
        'sci.space': 'Science',
        'soc.religion.christian': 'Other',
        'talk.politics.guns': 'Politics',
        'talk.politics.mideast': 'Politics',
        'talk.politics.misc': 'Politics',
        'talk.religion.misc': 'Other'
    }

    def __init__(self, datasets_path: str = "datasets"):
        self.datasets_path = Path(datasets_path)

    def normalize_category(self, category: str) -> str:
        """Normalize category to main categories"""
        return self.CATEGORY_MAPPING.get(category, category)

    def parse_ag_news(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Parse AG News CSV dataset"""
        ag_file = self.datasets_path / "AG news.csv"
        if not ag_file.exists():
            return [], []

        documents = []
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        categories = set()

        try:
            with open(ag_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get('text', '').strip()
                    label_id = int(row.get('label', 0))
                    category = label_map.get(label_id, "Business")
                    normalized_category = self.normalize_category(category)

                    if text:
                        documents.append((text, normalized_category))
                        categories.add(normalized_category)
        except Exception as e:
            print(f"Error parsing AG News: {e}")

        return documents, sorted(list(categories))

    def parse_bbc_news(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Parse BBC News Train CSV dataset"""
        bbc_file = self.datasets_path / "BBC News Train.csv"
        if not bbc_file.exists():
            return [], []

        documents = []
        categories = set()

        try:
            with open(bbc_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get('Text', '').strip()
                    category = row.get('Category', '').strip()
                    normalized_category = self.normalize_category(category)

                    if text and category:
                        documents.append((text, normalized_category))
                        categories.add(normalized_category)
        except Exception as e:
            print(f"Error parsing BBC News: {e}")

        return documents, sorted(list(categories))

    def parse_spam(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Parse spam CSV dataset"""
        spam_file = self.datasets_path / "spam.csv"
        if not spam_file.exists():
            return [], []

        documents = []
        categories = set()

        try:
            with open(spam_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        category = row[0].strip()
                        text = row[1].strip()
                        normalized_category = self.normalize_category(category)

                        if text and category:
                            documents.append((text, normalized_category))
                            categories.add(normalized_category)
        except Exception as e:
            print(f"Error parsing spam dataset: {e}")

        return documents, sorted(list(categories))

    def parse_20_newsgroups(self, limit_per_category: int = 500) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Parse 20 Newsgroups txt files"""
        newsgroups_dir = self.datasets_path / "20 Newsgroups"
        if not newsgroups_dir.exists():
            return [], []

        documents = []
        categories = set()

        # Get all txt files in the directory
        txt_files = list(newsgroups_dir.glob("*.txt"))

        for txt_file in txt_files:
            # Extract category name from filename (e.g., "alt.atheism.txt" -> "alt.atheism")
            category = txt_file.stem
            normalized_category = self.normalize_category(category)
            categories.add(normalized_category)

            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Split content by common newsgroup separators
                    # Assuming each message starts with "From: "
                    messages = content.split('\nFrom: ')

                    count = 0
                    for msg in messages:
                        if count >= limit_per_category:
                            break

                        # Extract text after headers (simple approach: take everything after first blank line)
                        parts = msg.split('\n\n', 1)
                        if len(parts) > 1:
                            text = parts[1].strip()
                            if len(text) > 50:  # Minimum text length
                                documents.append((text, normalized_category))
                                count += 1
            except Exception as e:
                print(f"Error parsing {txt_file}: {e}")

        return documents, sorted(list(categories))

    def get_dataset(self, dataset_name: str, sample_limit: int = None) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Get a specific dataset by name"""
        parsers = {
            'ag_news': self.parse_ag_news,
            'bbc': self.parse_bbc_news,
            'spam': self.parse_spam,
            'newsgroups': self.parse_20_newsgroups
        }

        parser = parsers.get(dataset_name)
        if not parser:
            return [], []

        documents, categories = parser()

        # Apply sample limit if specified
        if sample_limit and documents:
            documents = documents[:sample_limit]

        return documents, categories

    def get_all_available_datasets(self) -> Dict[str, bool]:
        """Check which datasets are available"""
        available = {}

        available['ag_news'] = (self.datasets_path / "AG news.csv").exists()
        available['bbc'] = (self.datasets_path / "BBC News Train.csv").exists()
        available['spam'] = (self.datasets_path / "spam.csv").exists()
        available['newsgroups'] = (self.datasets_path / "20 Newsgroups").exists()

        return available
