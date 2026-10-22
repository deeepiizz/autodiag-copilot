"""Knowledge-base chunking and hybrid retrieval (BM25 + TF-IDF, fused with RRF)."""
import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


@dataclass
class Chunk:
    id: str          # e.g. "misfire_diagnosis#random-misfire-p0300" or "dtc:P0171"
    title: str
    text: str


# Workshop slang and abbreviations mapped to the terms used in the manuals.
GLOSSARY = {
    "lambda": "o2 oxygen sensor",
    "o2": "oxygen",
    "crank": "crankshaft ckp",
    "cam": "camshaft cmp",
    "cat": "catalyst catalytic converter",
    "mil": "check engine lamp",
    "light": "lamp",
    "maf": "mass air flow",
    "ect": "engine coolant temperature",
    "iat": "intake air temperature",
    "tcm": "transmission control module",
    "ecm": "engine control module",
    "pcm": "engine control module ecm",
    "abs": "anti-lock brake wheel speed",
    "hv": "high voltage",
    "ev": "electric high voltage battery",
    "evap": "evaporative emission",
    "misfiring": "misfire",
    "drive": "driven driving",
}


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def expand_query(query):
    extra = [GLOSSARY[t] for t in tokenize(query) if t in GLOSSARY]
    return " ".join([query, *extra])


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_chunks(kb_dir=KB_DIR):
    """One chunk per '## ' section of each manual, plus one per DTC row."""
    chunks = []
    for path in sorted((kb_dir / "manuals").glob("*.md")):
        doc_title, section, lines = path.stem, None, []

        def flush():
            body = " ".join(l.strip() for l in lines if l.strip() and not l.startswith(">"))
            if section and body:
                chunks.append(Chunk(f"{path.stem}#{_slug(section)}", f"{doc_title} - {section}", body))

        for line in path.read_text().splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                section, lines = line[3:].strip(), []
            else:
                lines.append(line)
        flush()

    with open(kb_dir / "dtc_codes.csv", newline="") as f:
        for row in csv.DictReader(f):
            text = (f"{row['code']} {row['description']}. System: {row['system']}. "
                    f"Severity: {row['severity']}. Common causes: {row['common_causes']}.")
            chunks.append(Chunk(f"dtc:{row['code']}", f"DTC {row['code']}", text))
    return chunks


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.lengths = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = self.lengths.mean()
        self.tf = [{} for _ in self.docs]
        df = {}
        for i, toks in enumerate(self.docs):
            for t in toks:
                self.tf[i][t] = self.tf[i].get(t, 0) + 1
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: np.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.docs))
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.tf], dtype=float)
            denom = tf + self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
            out += self.idf[t] * tf * (self.k1 + 1) / denom
        return out


class Retriever:
    """Sparse, dense and hybrid retrieval over the knowledge base.

    mode:
        "bm25", "tfidf"            sparse lexical retrieval
        "lsa", "st"                dense: LSA concepts or sentence-transformer embeddings
        "hybrid"                   BM25 + TF-IDF fused with reciprocal-rank fusion (RRF)
        "hybrid+lsa", "hybrid+st"  BM25 + TF-IDF + a dense retriever, fused with RRF
    expand: add glossary synonyms to the query first.
    rerank: re-order the top `rerank_depth` candidates with a cross-encoder (needs the dense extras).
    """

    COMPONENTS = {"bm25": ["bm25"], "tfidf": ["tfidf"], "lsa": ["lsa"], "st": ["st"],
                  "hybrid": ["bm25", "tfidf"], "hybrid+lsa": ["bm25", "tfidf", "lsa"],
                  "hybrid+st": ["bm25", "tfidf", "st"]}

    def __init__(self, chunks=None, mode="hybrid", expand=True, rerank=False, rerank_depth=20, rrf_k=60):
        if mode not in self.COMPONENTS:
            raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(self.COMPONENTS)}")
        self.chunks = chunks or load_chunks()
        self.mode, self.expand, self.rrf_k, self.rerank_depth = mode, expand, rrf_k, rerank_depth
        texts = [f"{c.title} {c.text}" for c in self.chunks]
        parts = self.COMPONENTS[mode]
        if "bm25" in parts:
            self.bm25 = BM25(texts)
        if "tfidf" in parts:
            self.tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.tfidf.fit_transform(texts)
        if "lsa" in parts or "st" in parts:
            from . import dense
            self.encoder = (dense.LSAEncoder() if "lsa" in parts else dense.SentenceTransformerEncoder()).fit(texts)
            self.embeddings = self.encoder.encode(texts)
        self.reranker = None
        if rerank:
            from .dense import CrossEncoderReranker
            self.reranker = CrossEncoderReranker()

    def _ranked(self, scores):
        return list(np.argsort(-scores, kind="stable"))

    def _scores(self, part, query):
        if part == "bm25":
            return self.bm25.scores(query)
        if part == "tfidf":
            return (self.matrix @ self.tfidf.transform([query]).T).toarray().ravel()
        return self.embeddings @ self.encoder.encode([query])[0]

    def search(self, query, k=4, include_dtc=True):
        original = query
        if self.expand:
            query = expand_query(query)
        rankings = [self._ranked(self._scores(part, query)) for part in self.COMPONENTS[self.mode]]
        if len(rankings) == 1:
            order = rankings[0]
        else:
            fused = np.zeros(len(self.chunks))
            for ranking in rankings:
                for rank, idx in enumerate(ranking):
                    fused[idx] += 1.0 / (self.rrf_k + rank + 1)
            order = self._ranked(fused)
        hits = [self.chunks[i] for i in order]
        if not include_dtc:
            hits = [c for c in hits if not c.id.startswith("dtc:")]
        if self.reranker:
            # The cross-encoder reads natural language, so it gets the unexpanded query.
            hits = self.reranker.rerank(original, hits[:self.rerank_depth]) + hits[self.rerank_depth:]
        return hits[:k]


def _st_available():
    from .dense import sentence_transformers_available
    return sentence_transformers_available()


# Retrieval modes and whether each can run in this environment.
MODES = {"bm25": lambda: True, "tfidf": lambda: True, "lsa": lambda: True, "hybrid": lambda: True,
         "hybrid+lsa": lambda: True, "st": _st_available, "hybrid+st": _st_available}


@lru_cache
def default_retriever():
    return Retriever()
