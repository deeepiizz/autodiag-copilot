"""Retrieval evaluation: Hit@1, Hit@3 and MRR@5 per retriever, on dev and held-out splits.

dev.jsonl was used while building the retriever (e.g. to write the glossary);
heldout.jsonl was written before any tuning and is the honest number.

    python eval/run_eval.py                    # print the table, write results.json
    python eval/run_eval.py --check            # CI: fail if any metric drops below baseline
    python eval/run_eval.py --update-baseline  # accept the current numbers as the new baseline
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autodiag.retriever import MODES, Retriever, load_chunks  # noqa: E402

K = 5
TOLERANCE = 0.02
EVAL_DIR = Path(__file__).resolve().parent
BASELINE = EVAL_DIR / "baseline.json"


def load_split(name):
    return [json.loads(l) for l in (EVAL_DIR / f"{name}.jsonl").read_text().splitlines() if l.strip()]


def evaluate(retriever, questions):
    hit1 = hit3 = rr = 0.0
    misses = []
    for item in questions:
        ids = [c.id for c in retriever.search(item["q"], k=K)]
        ranks = [i for i, cid in enumerate(ids) if cid in item["relevant"]]
        if ranks:
            hit1 += ranks[0] == 0
            hit3 += ranks[0] < 3
            rr += 1 / (ranks[0] + 1)
        else:
            misses.append(item["q"])
    n = len(questions)
    return {"hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3), f"mrr@{K}": round(rr / n, 3)}, misses


def configurations():
    """(name, Retriever kwargs) for every configuration that can run in this environment."""
    for mode, available in MODES.items():
        if not available():
            print(f"(skipping {mode}: optional dependencies not installed)")
            continue
        for expand in (False, True):
            yield mode + ("+glossary" if expand else ""), {"mode": mode, "expand": expand}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--show-misses", action="store_true")
    args = p.parse_args()

    chunks = load_chunks()
    splits = {"dev": load_split("dev"), "heldout": load_split("heldout")}
    print(f"{len(chunks)} chunks; " + ", ".join(f"{k}: {len(v)} questions" for k, v in splits.items()) + "\n")
    print(f"{'retriever':<22}" + "".join(f"{s + ' ' + m:>15}" for s in splits for m in ("hit@1", "hit@3", "mrr@5")))

    results = {}
    for name, kwargs in configurations():
        retriever = Retriever(chunks, **kwargs)
        row = {}
        for split, questions in splits.items():
            metrics, misses = evaluate(retriever, questions)
            row[split] = metrics
            if args.show_misses:
                row[split + "_misses"] = misses
        results[name] = row
        print(f"{name:<22}" + "".join(f"{row[s][m]:>15.2f}" for s in splits for m in ("hit@1", "hit@3", f"mrr@{K}")))
        if args.show_misses:
            for split in splits:
                for q in row[split + "_misses"]:
                    print(f"    {split} miss: {q}")

    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    if args.update_baseline:
        BASELINE.write_text(json.dumps({k: {s: v[s] for s in splits} for k, v in results.items()}, indent=2) + "\n")
        print(f"\nWrote {BASELINE.name}")
    if args.check:
        baseline = json.loads(BASELINE.read_text())
        failures = [f"{name} {split} {metric}: {results[name][split][metric]} < {value}"
                    for name, splits_ in baseline.items() if name in results
                    for split, metrics in splits_.items() for metric, value in metrics.items()
                    if results[name][split][metric] < value - TOLERANCE]
        if failures:
            print("\nRegression against baseline:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("\nNo regressions against baseline.")


if __name__ == "__main__":
    main()
