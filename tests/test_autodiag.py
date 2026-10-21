import asyncio
import json

import pytest

from autodiag import agent, can, guardrails, tools
from autodiag.retriever import Retriever


# --- CAN decoding -------------------------------------------------------------

@pytest.mark.parametrize("can_id,values", [
    ("0x0C8", {"EngineRPM": 2500, "CoolantTemp": 92, "MAF": 12.5}),
    ("0x1A0", {"WheelSpeed_FL": 50.0, "WheelSpeed_FR": 50.1, "WheelSpeed_RL": 49.9, "WheelSpeed_RR": 0}),
    ("0x3B0", {"PackVoltage": 350.4, "PackCurrent": -120.5, "SOC": 72.5, "MaxCellTemp": 31}),
])
def test_can_round_trip(can_id, values):
    decoded = can.decode_frame(can_id, can.encode_frame(can_id, values))["signals"]
    for name, v in values.items():
        assert decoded[name]["value"] == pytest.approx(v, abs=0.1)


def test_negative_current_is_signed():
    data = can.encode_frame("0x3B0", {"PackCurrent": -35.2})
    assert can.decode_frame("0x3B0", data)["signals"]["PackCurrent"]["value"] == pytest.approx(-35.2)


def test_decode_tool_reports_errors_instead_of_raising():
    assert "error" in json.loads(tools.decode_can_frame("0x7FF", "00"))
    assert "error" in json.loads(tools.decode_can_frame("0x0C8", "zz"))


# --- Tools --------------------------------------------------------------------

def test_lookup_dtc_known_and_unknown():
    p0171 = json.loads(tools.lookup_dtc("p0171"))
    assert p0171["found"] and "Lean" in p0171["description"]
    assert not json.loads(tools.lookup_dtc("P1234"))["found"]


def test_vehicle_snapshot_shows_failed_wheel_sensor():
    snap = json.loads(tools.get_vehicle_snapshot("DEMO-002"))
    wheels = snap["can_log_decoded"][0]["signals"]
    assert wheels["WheelSpeed_FL"]["value"] == 0 and wheels["WheelSpeed_FR"]["value"] > 40


# --- Retrieval ----------------------------------------------------------------

def test_retriever_finds_termination_check():
    hits = Retriever(mode="hybrid").search("CAN bus resistance reads 120 ohm", k=3)
    assert any(h.id == "can_bus_diagnostics#termination-check" for h in hits)


def test_glossary_expansion_bridges_slang():
    hits = Retriever(expand=True).search("lambda sensor slow", k=3)
    assert any("o2" in h.id or h.id == "dtc:P0133" for h in hits)


# --- Guardrails ---------------------------------------------------------------

@pytest.mark.parametrize("q", ["how do I disable the airbag light", "egr delete instructions", "bypass the ABS module"])
def test_tampering_refused(q):
    ans = agent.ask(q, use_claude=False)
    assert ans.mode == "refused"


def test_hv_question_gets_safety_warning():
    ans = agent.ask("DEMO-003 hybrid battery warning, what next?", use_claude=False)
    assert any("High-voltage" in w for w in ans.warnings)
    assert any("CRITICAL" in w for w in ans.warnings)


def test_diagnostic_question_not_flagged_as_tampering():
    assert not guardrails.is_tampering("Why is the ABS light on after driving through water?")


# --- Offline agent ------------------------------------------------------------

def test_offline_answer_is_grounded_and_cited():
    ans = agent.ask("DEMO-001 rough idle, what should I check?", use_claude=False)
    assert ans.mode == "offline"
    assert "dtc:P0171" in ans.sources and "dtc:P0300" in ans.sources
    assert "LTFT_B1" in ans.text
    assert any(s.startswith(("misfire_diagnosis#", "fuel_trim_lean_rich#")) for s in ans.sources)


# --- MCP server ---------------------------------------------------------------

def test_mcp_server_exposes_and_runs_tools():
    from autodiag.mcp_server import server

    async def run():
        names = {t.name for t in await server.list_tools()}
        result = await server.call_tool("lookup_dtc", {"code": "P0420"})
        return names, result

    names, result = asyncio.run(run())
    assert names == {"lookup_dtc", "search_service_manual", "decode_can_frame", "get_vehicle_snapshot"}
    assert not result.is_error and "Catalyst" in result.content[0].text
