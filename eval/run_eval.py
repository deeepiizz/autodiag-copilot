"""Retrieval evaluation: Hit@1, Hit@3 and MRR for BM25, TF-IDF and hybrid retrieval.

    python eval/run_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autodiag.retriever import Retriever, load_chunks  # noqa: E402

K = 5


def evaluate(mode, expand, questions, chunks):
    r = Retriever(chunks, mode=mode, expand=expand)
    hit1 = hit3 = rr = 0.0
    misses = []
    for item in questions:
        ids = [c.id for c in r.search(item["q"], k=K)]
        ranks = [i for i, cid in enumerate(ids) if cid in item["relevant"]]
        if ranks:
            hit1 += ranks[0] == 0
            hit3 += ranks[0] < 3
            rr += 1 / (ranks[0] + 1)
        else:
            misses.append(item["q"])
    n = len(questions)
    return {"mode": mode + ("+glossary" if expand else ""), "hit@1": hit1 / n, "hit@3": hit3 / n, f"mrr@{K}": rr / n}, misses


def main():
    path = Path(__file__).with_name("questions.jsonl")
    questions = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    chunks = load_chunks()
    print(f"{len(questions)} questions, {len(chunks)} chunks\n")
    print(f"{'mode':<16} {'hit@1':>6} {'hit@3':>6} {'mrr@5':>6}")
    results = []
    for expand in (False, True):
        for mode in ("bm25", "tfidf", "hybrid"):
            res, misses = evaluate(mode, expand, questions, chunks)
            results.append(res)
            print(f"{res['mode']:<16} {res['hit@1']:>6.2f} {res['hit@3']:>6.2f} {res[f'mrr@{K}']:>6.2f}")
            for q in misses:
                print(f"    miss: {q}")
    Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
