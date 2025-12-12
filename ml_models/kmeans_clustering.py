from typing import List, Dict, Tuple
import math

class KMeansClustering:
    """Manual K-Means clustering for text documents"""

    def __init__(self, k: int = 3, max_iterations: int = 100, seed: int = 42):
        self.k = k
        self.max_iterations = max_iterations
        self.seed = seed
        self.centroids = []
        self.assignments = []

    def _tfidf_vectorize(self, documents: List[List[str]]) -> List[Dict[str, float]]:
        """
        Create TF-IDF vectors manually
        Returns document vectors as dictionaries (word: score)
        """
        # Calculate document frequency
        doc_freq = {}
        total_docs = len(documents)

        for doc in documents:
            words_seen = set()
            for word in doc:
                if word not in words_seen:
                    doc_freq[word] = doc_freq.get(word, 0) + 1
                    words_seen.add(word)

        # Calculate IDF and TF-IDF for each document
        vectors = []
        for doc in documents:
            vector = {}
            term_freq = {}

            # Calculate term frequency
            for word in doc:
                term_freq[word] = term_freq.get(word, 0) + 1

            # Calculate TF-IDF
            doc_length = len(doc)
            for word, freq in term_freq.items():
                tf = freq / doc_length if doc_length > 0 else 0
                idf = math.log(total_docs / doc_freq[word]) if doc_freq[word] > 0 else 0
                vector[word] = tf * idf

            vectors.append(vector)

        return vectors

    def _euclidean_distance(self, vec1: Dict, vec2: Dict) -> float:
        """Calculate Euclidean distance between two sparse vectors"""
        all_keys = set(vec1.keys()) | set(vec2.keys())

        sum_squared = 0
        for key in all_keys:
            diff = vec1.get(key, 0) - vec2.get(key, 0)
            sum_squared += diff ** 2

        return math.sqrt(sum_squared)

    def _initialize_centroids(self, vectors: List[Dict]) -> None:
        """Initialize centroids randomly"""
        random_indices = self._manual_random_sample(len(vectors), self.k, self.seed)
        self.centroids = [vectors[i] for i in random_indices]

    def _manual_random_sample(self, n: int, k: int, seed: int) -> List[int]:
        """Manual random sampling without numpy"""
        # Simple pseudo-random based on seed
        indices = []
        used = set()
        rand_value = seed

        while len(indices) < k and len(indices) < n:
            rand_value = (rand_value * 1103515245 + 12345) % (2**31)
            idx = (rand_value // 65536) % n

            if idx not in used:
                indices.append(idx)
                used.add(idx)

        return indices

    def fit(self, documents: List[List[str]]) -> None:
        """Fit K-Means clustering"""
        # Vectorize documents
        vectors = self._tfidf_vectorize(documents)

        # Initialize centroids
        self._initialize_centroids(vectors)

        for iteration in range(self.max_iterations):
            # Assign documents to nearest centroid
            self.assignments = []
            for vector in vectors:
                min_distance = float('inf')
                nearest_centroid = 0

                for i, centroid in enumerate(self.centroids):
                    dist = self._euclidean_distance(vector, centroid)
                    if dist < min_distance:
                        min_distance = dist
                        nearest_centroid = i

                self.assignments.append(nearest_centroid)

            # Update centroids
            new_centroids = []
            for i in range(self.k):
                cluster_vectors = [
                    vectors[j] for j in range(len(vectors))
                    if self.assignments[j] == i
                ]

                if cluster_vectors:
                    # Calculate mean vector
                    new_centroid = self._calculate_mean_vector(cluster_vectors)
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(self.centroids[i])

            # Check for convergence
            if self._centroids_converged(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def _calculate_mean_vector(self, vectors: List[Dict]) -> Dict:
        """Calculate mean of vectors"""
        if not vectors:
            return {}

        all_keys = set()
        for vec in vectors:
            all_keys.update(vec.keys())

        mean_vector = {}
        for key in all_keys:
            total = sum(vec.get(key, 0) for vec in vectors)
            mean_vector[key] = total / len(vectors)

        return mean_vector

    def _centroids_converged(self, old: List[Dict], new: List[Dict]) -> bool:
        """Check if centroids have converged"""
        threshold = 0.001
        for old_c, new_c in zip(old, new):
            if self._euclidean_distance(old_c, new_c) > threshold:
                return False
        return True
