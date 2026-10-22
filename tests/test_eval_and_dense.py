import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autodiag import agent, dense
from autodiag.retriever import MODES, Retriever, load_chunks

ROOT = Path(__file__).resolve().parent.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


judge_mod = load_module("judge", "eval/judge.py")


# --- Dense retrieval ------------------------------------------------------------

def test_lsa_embeddings_are_normalised_and_deterministic():
    texts = [f"{c.title} {c.text}" for c in load_chunks()]
    a = dense.LSAEncoder().fit(texts).encode(texts[:5])
    b = dense.LSAEncoder().fit(texts).encode(texts[:5])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a, b)


@pytest.mark.parametrize("mode", [m for m, ok in MODES.items() if ok()])
def test_every_available_mode_finds_obvious_passage(mode):
    hits = Retriever(mode=mode).search("CAN bus termination resistance 60 ohm", k=5)
    assert "can_bus_diagnostics#termination-check" in [h.id for h in hits]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        Retriever(mode="magic")


def test_include_dtc_false_returns_only_manual_sections():
    hits = Retriever().search("P0171 System Too Lean (Bank 1)", k=5, include_dtc=False)
    assert len(hits) == 5 and not any(h.id.startswith("dtc:") for h in hits)


def test_reranker_reorders_candidates():
    r = Retriever(mode="bm25")
    target = "charging_system#battery-test"
    # Stand-in for the cross-encoder: puts the target passage first.
    r.reranker = SimpleNamespace(rerank=lambda q, cs: sorted(cs, key=lambda c: c.id != target))
    assert r.search("battery", k=1)[0].id == target


# --- Answer evaluation ----------------------------------------------------------

def test_deterministic_metrics_flag_hallucinated_citation():
    ans = agent.Answer("P0420 means ... [dtc:P0420] [made_up#section]", "offline",
                       sources=["dtc:P0420", "made_up#section"])
    case = {"q": "", "expect_dtcs": ["P0420"], "expect_sources": ["catalyst_o2_sensors#catalyst-efficiency-p0420"]}
    m = judge_mod.deterministic_metrics(case, ans, judge_mod.valid_sources())
    assert m["citation_validity"] == 0.5 and m["dtc_recall"] == 1.0 and m["source_recall"] == 0.0


def test_refusal_metric():
    case = {"q": "", "expect_refusal": True}
    refused = agent.Answer("no", "refused")
    assert judge_mod.deterministic_metrics(case, refused, set()) == {"refusal_correct": 1.0}


class FakeClient:
    """Mimics client.beta.messages.create and records the request."""

    def __init__(self, verdict, stop_reason="end_turn"):
        self.verdict, self.stop_reason, self.request = verdict, stop_reason, None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.verdict))])


def test_judge_computes_faithfulness_from_claims():
    verdict = {"answers_the_question": True, "claims": [
        {"claim": "a", "supported": True, "citation_supports_claim": "yes", "reason": ""},
        {"claim": "b", "supported": True, "citation_supports_claim": "no_citation", "reason": ""},
        {"claim": "c", "supported": False, "citation_supports_claim": "no", "reason": ""},
        {"claim": "d", "supported": True, "citation_supports_claim": "yes", "reason": ""},
    ]}
    client = FakeClient(verdict)
    ans = agent.ask("What does P0420 mean?", use_claude=False)
    out = judge_mod.judge({"q": "What does P0420 mean?"}, ans, client)
    assert out["faithfulness"] == 0.75
    assert out["citation_precision"] == pytest.approx(2 / 3)
    # The judge sees the recorded tool evidence and requests schema-constrained JSON.
    prompt = client.request["messages"][0]["content"]
    assert "lookup_dtc" in prompt and "Catalyst" in prompt
    assert client.request["output_config"]["format"]["type"] == "json_schema"


def test_judge_handles_refusal():
    out = judge_mod.judge({"q": "x"}, agent.Answer("t", "offline"), FakeClient({}, stop_reason="refusal"))
    assert "error" in out


def test_offline_answer_records_evidence():
    ans = agent.ask("DEMO-002 ABS light", use_claude=False)
    tools_used = {e["tool"] for e in ans.evidence}
    assert {"get_vehicle_snapshot", "analyze_can_log", "lookup_dtc", "search_service_manual"} <= tools_used
