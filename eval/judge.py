"""End-to-end answer evaluation.

Deterministic metrics (always, no API key needed):
  - citation_validity: share of cited sources that exist in the knowledge base
  - dtc_recall: share of expected DTCs mentioned in the answer
  - source_recall: share of expected manual sections cited
  - warning_correct / refusal_correct: safety behaviour matches expectations

LLM-as-judge (with --judge, needs Anthropic credentials):
  Claude lists every factual claim in the answer and marks whether the tool evidence
  supports it. faithfulness = supported claims / all claims, computed here rather than
  asked for as a score, so it is auditable.

    python eval/judge.py                     # offline answers, deterministic metrics
    python eval/judge.py --mode claude --judge
    python eval/judge.py --check             # CI: fail on regression vs baseline
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autodiag import agent  # noqa: E402
from autodiag.retriever import load_chunks  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
BASELINE = EVAL_DIR / "answer_baseline.json"
JUDGE_MODEL = "claude-opus-5-5"
TOLERANCE = 0.02

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "citation_supports_claim": {"type": "string", "enum": ["yes", "no", "no_citation"]},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "supported", "citation_supports_claim", "reason"],
                "additionalProperties": False,
            },
        },
        "answers_the_question": {"type": "boolean"},
    },
    "required": ["claims", "answers_the_question"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """You are grading an automotive diagnostic assistant for faithfulness.

Below are the evidence the assistant retrieved with its tools, the technician's question, and the assistant's answer.

List every factual or diagnostic claim the answer makes (skip greetings, headings and pure formatting). For each claim:
- supported: true only if the evidence states or directly implies it. General workshop knowledge that is not in the evidence counts as unsupported, even if it is true.
- citation_supports_claim: "yes" if the claim carries a [source] citation and that source's evidence supports it, "no" if it is cited but the cited source does not support it, "no_citation" if it has none.
- reason: one short sentence.

Also say whether the answer addresses what the technician asked.

<evidence>
{evidence}
</evidence>

<question>
{question}
</question>

<answer>
{answer}
</answer>"""


def valid_sources():
    return {c.id for c in load_chunks()}


def deterministic_metrics(case, ans, sources_in_kb):
    m = {}
    if "expect_refusal" in case:
        m["refusal_correct"] = float((ans.mode == "refused") == case["expect_refusal"])
        if case["expect_refusal"]:
            return m
    cited = ans.sources
    known = [s for s in cited if s in sources_in_kb or re.fullmatch(r"log:DEMO-\d{3}", s)]
    m["citation_validity"] = len(known) / len(cited) if cited else 1.0
    if case.get("expect_dtcs"):
        m["dtc_recall"] = sum(c in ans.text for c in case["expect_dtcs"]) / len(case["expect_dtcs"])
    if case.get("expect_sources"):
        m["source_recall"] = sum(s in cited for s in case["expect_sources"]) / len(case["expect_sources"])
    m["warning_correct"] = float(bool(ans.warnings) == case.get("expect_warning", False))
    return m


def judge(case, ans, client):
    """Ask Claude to grade one answer; returns the parsed judgement plus faithfulness."""
    evidence = "\n\n".join(f"[{e['tool']} {json.dumps(e['input'])}]\n{e['output']}" for e in ans.evidence)
    response = client.beta.messages.create(
        model=JUDGE_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            evidence=evidence or "(no evidence retrieved)", question=case["q"], answer=ans.text)}],
    )
    if response.stop_reason == "refusal":
        return {"error": "judge refused"}
    verdict = json.loads(next(b.text for b in response.content if b.type == "text"))
    claims = verdict["claims"]
    cited = [c for c in claims if c["citation_supports_claim"] != "no_citation"]
    verdict["faithfulness"] = sum(c["supported"] for c in claims) / len(claims) if claims else 1.0
    verdict["citation_precision"] = (sum(c["citation_supports_claim"] == "yes" for c in cited) / len(cited)
                                     if cited else None)
    return verdict


def summarise(rows):
    keys = sorted({k for r in rows for k in r["metrics"]})
    return {k: round(sum(r["metrics"][k] for r in rows if k in r["metrics"]) /
                     sum(k in r["metrics"] for r in rows), 3) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["offline", "claude"], default="offline")
    p.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness (needs credentials)")
    p.add_argument("--check", action="store_true")
    p.add_argument("--update-baseline", action="store_true")
    args = p.parse_args()

    cases = [json.loads(l) for l in (EVAL_DIR / "answers.jsonl").read_text().splitlines() if l.strip()]
    sources_in_kb = valid_sources()
    client = None
    if args.judge:
        import anthropic
        client = anthropic.Anthropic()

    rows = []
    for case in cases:
        ans = agent.ask(case["q"], use_claude=args.mode == "claude")
        metrics = deterministic_metrics(case, ans, sources_in_kb)
        row = {"id": case["id"], "mode": ans.mode, "metrics": metrics, "answer": ans.text, "sources": ans.sources}
        if client and ans.mode != "refused":
            verdict = judge(case, ans, client)
            row["judge"] = verdict
            if "faithfulness" in verdict:
                metrics["faithfulness"] = verdict["faithfulness"]
                if verdict["citation_precision"] is not None:
                    metrics["citation_precision"] = verdict["citation_precision"]
                metrics["answers_question"] = float(verdict["answers_the_question"])
        rows.append(row)
        print(f"{case['id']:<18} " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(metrics.items())))

    summary = summarise(rows)
    print("\nMEAN  " + "  ".join(f"{k}={v:.3f}" for k, v in summary.items()))
    out = EVAL_DIR / f"answer_results_{args.mode}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")

    if args.update_baseline:
        baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        baseline[args.mode] = summary
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")
    if args.check:
        expected = json.loads(BASELINE.read_text()).get(args.mode, {})
        failures = [f"{k}: {summary.get(k)} < {v}" for k, v in expected.items()
                    if k in summary and summary[k] < v - TOLERANCE]
        if failures:
            print("Regression against baseline:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("No regressions against baseline.")


if __name__ == "__main__":
    main()
