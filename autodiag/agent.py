"""Diagnostic assistant.

- With Anthropic credentials: Claude runs an agentic loop over the diagnostic tools
  and writes a grounded, cited answer.
- Without: a deterministic offline pipeline calls the same tools and assembles an
  extractive answer, so the project runs for anyone who clones it.

Either way the Answer records every tool result as `evidence`, which the
evaluation harness uses to judge faithfulness.
"""
import functools
import json
import os
import re
from dataclasses import dataclass, field

from . import guardrails
from .tools import (DTC_RE, TOOLS, analyze_can_log, get_vehicle_snapshot, lookup_dtc,
                    search_service_manual)

MODEL = "claude-opus-5-5"
VEHICLE_RE = re.compile(r"\bDEMO-\d{3}\b", re.IGNORECASE)
CITATION_RE = re.compile(r"\[([a-z0-9_]+#[a-z0-9-]+|dtc:[A-Z0-9]+|log:DEMO-\d{3})\]")

SYSTEM_PROMPT = """You are AutoDiag, a diagnostic assistant for automotive technicians.

Use the tools to gather evidence before answering. When a vehicle ID is given, read its snapshot (DTCs come from a UDS read) and analyse its CAN log for anomalies; relate anomalies in time to each other (e.g. a voltage dip and a module going silent). Look up every DTC, decode raw CAN or UDS data the technician pastes, and search the service manual (include_dtc=false) for test procedures.

Ground every technical claim in tool output and cite sources in square brackets using the `source` field, e.g. [fuel_trim_lean_rich#diagnosing-a-lean-code-p0171-p0174] or [dtc:P0171]. Cite CAN log findings as [log:DEMO-002]. If the knowledge base does not cover something, say so rather than guessing.

Structure the answer as: Summary, Evidence (codes and live data), Likely causes (most to least likely), Recommended tests in order, and Safety notes when relevant. Keep it concise and practical."""


@dataclass
class Answer:
    text: str
    mode: str                          # "claude", "offline" or "refused"
    sources: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    evidence: list = field(default_factory=list)   # [{"tool", "input", "output"}]


def has_credentials():
    return any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))


def _recording(fn, evidence):
    """Wrap a tool so each call's input and output is appended to `evidence`."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        evidence.append({"tool": fn.__name__, "input": kwargs or list(args), "output": out})
        return out
    return wrapper


def _gather_context(question, evidence):
    """Collect vehicle data, log anomalies, DTC info and manual passages for the question."""
    snapshot_fn, analyze_fn, dtc_fn, search_fn = (
        _recording(f, evidence) for f in (get_vehicle_snapshot, analyze_can_log, lookup_dtc, search_service_manual))
    vehicle_ids = list(dict.fromkeys(v.upper() for v in VEHICLE_RE.findall(question)))
    snapshots = [json.loads(snapshot_fn(vehicle_id=v)) for v in vehicle_ids]
    reports = [json.loads(analyze_fn(vehicle_id=s["vehicle_id"])) for s in snapshots if "error" not in s]
    codes = {c.upper() for c in DTC_RE.findall(question)}
    for s in snapshots:
        codes.update(s.get("active_dtcs", []))
    dtcs = [json.loads(dtc_fn(code=c)) for c in sorted(codes)]

    # Multi-query retrieval: one search for the question and symptoms, one per DTC,
    # fused with reciprocal-rank fusion so no single query dominates.
    queries = [" ".join([question] + [s.get("symptoms", "") for s in snapshots])]
    queries += [f"{d['code']} {d['description']}" for d in dtcs if d["found"]]
    fused, by_source, top_per_query = {}, {}, []
    for q in queries:
        # DTC facts come from lookup_dtc; passages are reserved for manual procedures.
        hits = json.loads(search_fn(query=q, k=8, include_dtc=False))
        for rank, h in enumerate(hits):
            fused[h["source"]] = fused.get(h["source"], 0) + 1 / (60 + rank + 1)
            by_source[h["source"]] = h
        if hits:
            top_per_query.append(hits[0]["source"])
    # Each sub-query's best hit first (coverage), then strong fused matches only:
    # a passage must score at least half the best fused score to fill a slot.
    best = max(fused.values(), default=0)
    strong = [src for src in sorted(fused, key=fused.get, reverse=True) if fused[src] >= best / 2]
    order = list(dict.fromkeys(top_per_query + strong))
    passages = [by_source[src] for src in order[:max(4, len(set(top_per_query)))]]
    return snapshots, reports, dtcs, passages


def _describe_report(r):
    lines = []
    for p in r["plausibility_violations"]:
        lines.append(f"- {p['description']} ({p['start']}-{p['end']} s)")
    for g in r["message_gaps"]:
        lines.append(f"- {g['message']} ({g['id']}) missing for {g['duration_s']} s at {g['start']} s "
                     f"(normal period {g['normal_period_s']} s)")
    by_signal = {}
    for s in r["spikes"]:
        by_signal.setdefault(s["signal"], []).append(s)
    for name, events in by_signal.items():
        lines.append(f"- {len(events)} abnormal {name} excursion(s), most extreme {events[0]['extreme']}")
    return lines or ["- No anomalies detected"]


def answer_offline(question):
    evidence = []
    snapshots, reports, dtcs, passages = _gather_context(question, evidence)
    report_by_vid = {r["vehicle_id"]: r for r in reports}
    lines = []
    for s in snapshots:
        if "error" in s:
            lines.append(f"**Vehicle:** {s['error']}")
            continue
        lines.append(f"**Vehicle {s['vehicle_id']}:** {s['description']}. Symptoms: {s['symptoms']}")
        for ecu, result in s["dtcs_by_ecu"].items():
            codes = ", ".join(d["dtc"] for d in result) if isinstance(result, list) else result
            lines.append(f"- UDS DTC read, {ecu}: {codes or 'no DTCs'}")
        for frame in s["latest_signals"]:
            vals = ", ".join(f"{k}={v['value']} {v['unit']}" for k, v in frame["signals"].items())
            lines.append(f"- CAN {frame['id']} {frame['message']}: {vals}")
        if s["vehicle_id"] in report_by_vid:
            r = report_by_vid[s["vehicle_id"]]
            lines.append(f"\n**CAN log analysis ({r['duration_s']} s)** [log:{s['vehicle_id']}]")
            lines += _describe_report(r)

    if dtcs:
        lines.append("\n**Trouble codes**")
        for d in dtcs:
            if d["found"]:
                lines.append(f"- **{d['code']}** {d['description']} (severity: {d['severity']}). "
                             f"Common causes: {'; '.join(d['common_causes'])} [dtc:{d['code']}]")
            else:
                lines.append(f"- **{d['code']}**: {d['message']}")

    lines.append("\n**Relevant procedures**")
    for p in passages:
        lines.append(f"- *{p['title']}*: {p['text']} [{p['source']}]")

    severities = [d.get("severity", "") for d in dtcs]
    warnings = guardrails.safety_warnings(" ".join([question] + [d.get("description", "") for d in dtcs]), severities)
    text = "\n".join(lines)
    return Answer(text, "offline", list(dict.fromkeys(CITATION_RE.findall(text))), warnings, evidence)


def answer_with_claude(question):
    import anthropic
    from anthropic import beta_tool

    evidence = []
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        tools=[beta_tool(_recording(fn, evidence)) for fn in TOOLS],
        messages=[{"role": "user", "content": question}],
        # On a safety-classifier refusal, retry on a fallback model chosen by the API.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        max_iterations=12,
    )
    final = None
    for message in runner:
        final = message

    if final is None or final.stop_reason == "refusal":
        return Answer("The model declined to answer this request.", "refused", evidence=evidence)
    text = "\n".join(b.text for b in final.content if b.type == "text")
    warnings = guardrails.safety_warnings(question + " " + text)
    return Answer(text, "claude", list(dict.fromkeys(CITATION_RE.findall(text))), warnings, evidence)


def ask(question, use_claude=None):
    if guardrails.is_tampering(question):
        return Answer(guardrails.TAMPERING_REFUSAL, "refused")
    if use_claude is None:
        use_claude = has_credentials()
    return answer_with_claude(question) if use_claude else answer_offline(question)
