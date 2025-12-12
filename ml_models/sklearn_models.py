try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    RandomForestClassifier = None
    TfidfVectorizer = None

# Flag to signal availability of sklearn
AVAILABLE = RandomForestClassifier is not None and TfidfVectorizer is not None

class RandomForestText:
    def __init__(self, n_estimators: int = 300, max_depth=None, class_weight="balanced"):
        if not AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
        self._fitted = False
        self.classes_ = None

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        self._fitted = True

    def predict(self, text: str):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def predict_proba(self, text: str):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if not hasattr(self.model, "predict_proba"):
            return {c: 0.0 for c in self.classes_}
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        return {c: float(p) for c, p in zip(self.classes_, probs)}
