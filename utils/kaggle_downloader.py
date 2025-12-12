"""
Kaggle Dataset Downloader
Download and convert Kaggle datasets for use with the Word Frequency Counter

SETUP INSTRUCTIONS:
1. Install kaggle: pip install kaggle
2. Download your Kaggle API key from https://www.kaggle.com/settings/account
3. Place kaggle.json in C:/Users/YourUsername/.kaggle/
4. Run this script: python utils/kaggle_downloader.py
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict

def check_kaggle_setup() -> bool:
    """Check if Kaggle API is properly configured"""
    try:
        import kaggle
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"

        if not kaggle_json.exists():
            print("❌ Kaggle API key not found!")
            print(f"   Download from: https://www.kaggle.com/settings/account")
            print(f"   Place kaggle.json in: {kaggle_dir}")
            return False

        return True
    except ImportError:
        print("❌ kaggle package not installed!")
        print("   Install it: pip install kaggle")
        return False

def download_bbc_news() -> bool:
    """Download BBC News dataset"""
    print("\n📥 Downloading BBC News Dataset...")
    try:
        import kaggle

        dataset_name = "sauravjoshi619/bbc-news-classification"
        download_path = "datasets/kaggle/bbc_raw"

        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True
        )

        # Convert to our format
        bbc_path = Path(download_path)
        samples = []

        if (bbc_path / "BBC News Train.csv").exists():
            import csv
            with open(bbc_path / "BBC News Train.csv", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append({
                        "text": row.get("ArticleText", ""),
                        "category": row.get("Category", "")
                    })

        # Save in our format
        output_path = Path("datasets/kaggle/bbc_news.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": "BBC News Classification",
                "categories": list(set(s["category"] for s in samples)),
                "data": samples
            }, f, indent=2)

        print(f"✅ BBC News dataset saved to {output_path}")
        print(f"   Total samples: {len(samples)}")
        return True

    except Exception as e:
        print(f"❌ Error downloading BBC News: {e}")
        return False

def download_sms_spam() -> bool:
    """Download SMS Spam Classification dataset"""
    print("\n📥 Downloading SMS Spam Dataset...")
    try:
        import kaggle

        dataset_name = "uciml/sms-spam-collection-dataset"
        download_path = "datasets/kaggle/spam_raw"

        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True
        )

        # Convert to our format
        spam_path = Path(download_path)
        samples = []

        if (spam_path / "spam.csv").exists():
            import csv
            with open(spam_path / "spam.csv", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("v2", "") or row.get("Message", "")
                    label = row.get("v1", "") or row.get("Label", "")
                    samples.append({
                        "text": text,
                        "category": label
                    })

        # Save in our format
        output_path = Path("datasets/kaggle/sms_spam.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": "SMS Spam Classification",
                "categories": ["spam", "ham"],
                "data": samples
            }, f, indent=2)

        print(f"✅ SMS Spam dataset saved to {output_path}")
        print(f"   Total samples: {len(samples)}")
        return True

    except Exception as e:
        print(f"❌ Error downloading SMS Spam: {e}")
        return False

def download_20newsgroups() -> bool:
    """Download 20 Newsgroups dataset"""
    print("\n📥 Downloading 20 Newsgroups Dataset...")
    try:
        import kaggle

        dataset_name = "uciml/twenty-newsgroups-text-classification"
        download_path = "datasets/kaggle/newsgroups_raw"

        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True
        )

        # Convert to our format
        newsgroups_path = Path(download_path)
        samples = []

        if (newsgroups_path / "20_newsgroups.csv").exists():
            import csv
            with open(newsgroups_path / "20_newsgroups.csv", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append({
                        "text": row.get("text", ""),
                        "category": row.get("target", "")
                    })

        # Save in our format
        output_path = Path("datasets/kaggle/newsgroups.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": "20 Newsgroups",
                "categories": list(set(s["category"] for s in samples)),
                "data": samples
            }, f, indent=2)

        print(f"✅ 20 Newsgroups dataset saved to {output_path}")
        print(f"   Total samples: {len(samples)}")
        return True

    except Exception as e:
        print(f"❌ Error downloading 20 Newsgroups: {e}")
        return False

def main():
    print("=" * 60)
    print("🎯 Kaggle Dataset Downloader")
    print("=" * 60)

    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\n⚠️  Setup Instructions:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Get API key: https://www.kaggle.com/settings/account")
        print("3. Place kaggle.json in ~/.kaggle/")
        sys.exit(1)

    print("\n✅ Kaggle setup is valid!")

    # Menu
    print("\n📚 Available Datasets:")
    print("1. BBC News Classification (5 categories, ~2000 samples)")
    print("2. SMS Spam Detection (2 categories, ~5500 samples)")
    print("3. 20 Newsgroups (20 categories, ~20000 samples)")
    print("4. Download All")
    print("5. Exit")

    choice = input("\nSelect dataset(s) to download (1-5): ").strip()

    results = {}

    if choice == '1':
        results['BBC'] = download_bbc_news()
    elif choice == '2':
        results['SMS'] = download_sms_spam()
    elif choice == '3':
        results['20NG'] = download_20newsgroups()
    elif choice == '4':
        print("\n🔄 Downloading all datasets...")
        results['BBC'] = download_bbc_news()
        results['SMS'] = download_sms_spam()
        results['20NG'] = download_20newsgroups()
    elif choice == '5':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice!")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {name}: {status}")
    print("=" * 60)

    print("\n✨ Datasets are ready to use!")
    print("   Start the app: streamlit run main.py")
    print("   The app will automatically detect and use the datasets.")

if __name__ == "__main__":
    main()
