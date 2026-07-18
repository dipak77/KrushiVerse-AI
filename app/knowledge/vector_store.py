import math
import re
from collections import Counter

class SimpleVectorStore:
    """In-memory Vector Index for agricultural documents with Cosine Similarity."""

    def __init__(self):
        self.documents = []
        self.doc_vectors = []
        self.vocabulary = set()
        self.idf = {}

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r'\w+', text.lower())
        return words

    def build_index(self, docs: list[dict]):
        self.documents = docs
        num_docs = len(docs)
        if num_docs == 0:
            return

        # Build vocabulary
        doc_tokens = [self._tokenize(doc["content"] + " " + doc["title"]) for doc in docs]
        for tokens in doc_tokens:
            self.vocabulary.update(tokens)

        # Calculate IDF
        self.idf = {}
        for term in self.vocabulary:
            doc_count = sum(1 for tokens in doc_tokens if term in set(tokens))
            self.idf[term] = math.log((1 + num_docs) / (1 + doc_count)) + 1.0

        # Build TF-IDF vectors
        self.doc_vectors = []
        for tokens in doc_tokens:
            vec = self._vectorize(tokens)
            self.doc_vectors.append(vec)

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total_terms = len(tokens) if tokens else 1
        vector = {}
        for term, count in tf.items():
            if term in self.idf:
                vector[term] = (count / total_terms) * self.idf[term]
        return vector

    def _cosine_similarity(self, vec1: dict[str, float], vec2: dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([val ** 2 for val in vec1.values()])
        sum2 = sum([val ** 2 for val in vec2.values()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        return float(numerator) / denominator

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = self._tokenize(query)
        query_vec = self._vectorize(query_tokens)

        results = []
        for i, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_similarity(query_vec, doc_vec)
            if score > 0.001:
                results.append((score, self.documents[i]))

        results.sort(key=lambda x: x[0], reverse=True)
        return [{"score": float(score), "doc": doc} for score, doc in results[:top_k]]
