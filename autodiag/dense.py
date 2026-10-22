"""Dense (embedding) retrieval and reranking backends.

- LSAEncoder: latent semantic analysis (TF-IDF -> truncated SVD). No model download,
  so it always runs, including in CI.
- SentenceTransformerEncoder / CrossEncoderReranker: neural models from Hugging Face.
  Optional: install requirements-dense.txt (pulls in PyTorch and downloads the models).
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LSAEncoder:
    """Embeds text into a low-rank 'concept' space learned from the corpus itself.

    Word and character n-grams are combined so that morphological variants
    ("misfire" / "misfiring") land close together.
    """

    def __init__(self, n_components=48, seed=0):
        self.words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.n_components, self.seed = n_components, seed

    def _features(self, texts, fit=False):
        from scipy.sparse import hstack
        if fit:
            return hstack([self.words.fit_transform(texts), self.chars.fit_transform(texts)]).tocsr()
        return hstack([self.words.transform(texts), self.chars.transform(texts)]).tocsr()

    def fit(self, texts):
        x = self._features(texts, fit=True)
        k = min(self.n_components, min(x.shape) - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(x)
        return self

    def encode(self, texts):
        return normalize(self.svd.transform(self._features(texts)))


def sentence_transformers_available():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class SentenceTransformerEncoder:
    def __init__(self, model=ST_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def fit(self, texts):
        return self

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly; slower but more precise than bi-encoders."""

    def __init__(self, model=CROSS_ENCODER_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def rerank(self, query, chunks):
        scores = self.model.predict([(query, f"{c.title}. {c.text}") for c in chunks])
        return [c for _, c in sorted(zip(scores, chunks), key=lambda x: -x[0])]
