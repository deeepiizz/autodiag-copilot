"""Diagnostic assistant.

- With Anthropic credentials: Claude runs an agentic loop over the diagnostic tools
  and writes a grounded, cited answer.
- Without: a deterministic offline pipeline calls the same tools and assembles an
  extractive answer, so the project runs for anyone who clones it.
"""
import json
import os
import re
from dataclasses import dataclass, field

from . import guardrails
from .tools import DTC_RE, TOOLS, get_vehicle_snapshot, lookup_dtc, search_service_manual

MODEL = "claude-opus-5-5"
VEHICLE_RE = re.compile(r"\bDEMO-\d{3}\b", re.IGNORECASE)

SYSTEM_PROMPT = """You are AutoDiag, a diagnostic assistant for automotive technicians.

Use the tools to gather evidence before answering: read the vehicle snapshot when a vehicle ID is given, look up every DTC, decode CAN data when relevant, and search the service manual for test procedures.

Ground every technical claim in tool output and cite sources in square brackets using the `source` field, e.g. [fuel_trim_lean_rich#diagnosing-a-lean-code-p0171-p0174] or [dtc:P0171]. If the knowledge base does not cover something, say so rather than guessing.

Structure the answer as: Summary, Evidence (codes and live data), Likely causes (most to least likely), Recommended tests in order, and Safety notes when relevant. Keep it concise and practical."""


@dataclass
class Answer:
    text: str
    mode: str                          # "claude", "offline" or "refused"
    sources: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def has_credentials():
    return any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))


def _gather_context(question):
    """Collect vehicle data, DTC info and manual passages relevant to the question."""
    vehicle_ids = [v.upper() for v in VEHICLE_RE.findall(question)]
    snapshots = [json.loads(get_vehicle_snapshot(v)) for v in vehicle_ids]
    codes = {c.upper() for c in DTC_RE.findall(question)}
    for s in snapshots:
        codes.update(s.get("active_dtcs", []))
    dtcs = [json.loads(lookup_dtc(c)) for c in sorted(codes)]

    # Expand the search query with code descriptions and symptoms for better recall.
    query_parts = [question] + [d["description"] for d in dtcs if d["found"]] + [s.get("symptoms", "") for s in snapshots]
    passages = json.loads(search_service_manual(" ".join(query_parts), k=4))
    return snapshots, dtcs, passages


def answer_offline(question):
    snapshots, dtcs, passages = _gather_context(question)
    lines = []
    for s in snapshots:
        if "error" in s:
            lines.append(f"**Vehicle:** {s['error']}")
            continue
        lines.append(f"**Vehicle {s['vehicle_id']}:** {s['description']}. Symptoms: {s['symptoms']}")
        for frame in s["can_log_decoded"]:
            vals = ", ".join(f"{k}={v['value']} {v['unit']}" for k, v in frame["signals"].items())
            lines.append(f"- CAN {frame['id']} {frame['message']}: {vals}")

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
    sources = [f"dtc:{d['code']}" for d in dtcs if d["found"]] + [p["source"] for p in passages]
    return Answer("\n".join(lines), "offline", sources, warnings)


def answer_with_claude(question):
    import anthropic
    from anthropic import beta_tool

    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        tools=[beta_tool(fn) for fn in TOOLS],
        messages=[{"role": "user", "content": question}],
        # On a safety-classifier refusal, retry on a fallback model chosen by the API.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        max_iterations=10,
    )
    final = None
    sources = []
    for message in runner:
        final = message
        for block in message.content:
            if block.type == "tool_use" and block.name == "lookup_dtc":
                sources.append(f"dtc:{block.input.get('code', '').upper()}")

    if final is None or final.stop_reason == "refusal":
        return Answer("The model declined to answer this request.", "refused")
    text = "\n".join(b.text for b in final.content if b.type == "text")
    sources += re.findall(r"\[([a-z0-9_]+#[a-z0-9-]+|dtc:[A-Z0-9]+)\]", text)
    warnings = guardrails.safety_warnings(question + " " + text)
    return Answer(text, "claude", list(dict.fromkeys(sources)), warnings)


def ask(question, use_claude=None):
    if guardrails.is_tampering(question):
        return Answer(guardrails.TAMPERING_REFUSAL, "refused")
    if use_claude is None:
        use_claude = has_credentials()
    return answer_with_claude(question) if use_claude else answer_offline(question)
