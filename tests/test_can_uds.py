import csv
import importlib.util
import json
from pathlib import Path

import pytest

from autodiag import anomaly, can, tools, uds

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "knowledge_base"
VEHICLES = json.loads((KB / "vehicles.json").read_text())["vehicles"]


def frames_for(vid):
    return can.parse_candump((KB / VEHICLES[vid]["can_log_file"]).read_text())


# --- DTC encoding ---------------------------------------------------------------

def test_every_dtc_in_knowledge_base_round_trips():
    with open(KB / "dtc_codes.csv", newline="") as f:
        codes = [row["code"] for row in csv.DictReader(f)]
    for code in codes:
        assert uds.decode_dtc(uds.encode_dtc(code, 0x31)) == (code, 0x31)


def test_known_dtc_byte_layout():
    # P0171: system P (00), first digit 0, then 1 / 71 -> 0x01 0x71
    assert uds.encode_dtc("P0171") == bytes([0x01, 0x71, 0x00])
    # U0121: system U (11) -> 0xC1 0x21
    assert uds.encode_dtc("U0121") == bytes([0xC1, 0x21, 0x00])


def test_status_flags():
    assert uds.decode_status(0x09) == ["testFailed", "confirmedDTC"]
    assert "warningIndicatorRequested" in uds.decode_status(0xAF)


# --- ISO-TP ---------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 7, 8, 20, 64])
def test_isotp_round_trip(n):
    payload = bytes(range(n))
    frames = [(0.0, 0x7E8, f) for f in uds.isotp_segment(payload)]
    assert all(len(f[2]) == 8 for f in frames)
    assert uds.isotp_reassemble(frames)[0].payload == payload


def test_isotp_drops_out_of_sequence_message():
    frames = uds.isotp_segment(bytes(range(20)))
    bad = [frames[0], frames[2], frames[1]]          # consecutive frames swapped
    assert uds.isotp_reassemble([(0.0, 0x7E8, f) for f in bad]) == []


# --- UDS ------------------------------------------------------------------------

def test_read_dtc_response_parses():
    payload = uds.read_dtc_response([("P0420", 0x00, 0x2F), ("C0035", 0x31, 0xAF)])
    parsed = uds.parse_uds(payload)
    assert [d["dtc"] for d in parsed["dtcs"]] == ["P0420", "C0035"]
    assert parsed["dtcs"][1]["failure_type"] == "0x31"


def test_negative_response():
    parsed = uds.parse_uds(bytes([0x7F, 0x19, 0x31]))
    assert parsed["type"] == "negative_response" and parsed["nrc_name"] == "requestOutOfRange"


@pytest.mark.parametrize("vid", list(VEHICLES))
def test_uds_read_from_log_matches_ground_truth(vid):
    snap = json.loads(tools.get_vehicle_snapshot(vid))
    assert snap["active_dtcs"] == sorted(VEHICLES[vid]["active_dtcs"])


def test_unresponsive_abs_module_is_reported():
    assert uds.read_dtcs_from_frames(frames_for("DEMO-004"))["ABS"]["result"] == "no_response"


# --- candump --------------------------------------------------------------------

def test_candump_round_trip():
    frames = frames_for("DEMO-002")[:50]
    assert can.parse_candump(can.format_candump(frames)) == frames


def test_candump_rejects_malformed_line():
    with pytest.raises(ValueError, match="line 2"):
        can.parse_candump("(1.000000) can0 1A0#00\nnot a frame\n")


def test_logs_are_reproducible(tmp_path):
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_demo_logs.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.main(tmp_path)
    for vid in VEHICLES:
        assert (tmp_path / f"{vid}.log").read_text() == (KB / "logs" / f"{vid}.log").read_text()


# --- Anomaly detection ----------------------------------------------------------

def test_detects_all_injected_misfires_without_false_positives():
    report = anomaly.analyze(frames_for("DEMO-001"))
    rpm = [s for s in report["spikes"] if s["signal"] == "EngineRPM"]
    assert len(rpm) == 9                                   # 9 dips injected by the generator
    assert all(s["extreme"] < 700 for s in rpm)            # all are drops below normal idle
    assert {s["signal"] for s in report["spikes"]} == {"EngineRPM"}


def test_wheel_speed_dropouts_found_by_plausibility_rule():
    report = anomaly.analyze(frames_for("DEMO-002"))
    drops = [p for p in report["plausibility_violations"] if p["rule"] == "wheel_speed_dropout"]
    assert len(drops) == 3


def test_abs_module_dropping_off_bus_is_a_message_gap():
    gaps = json.loads(tools.analyze_can_log("DEMO-004"))["message_gaps"]
    assert len(gaps) == 1 and gaps[0]["message"] == "WheelSpeeds"
    assert 9 <= gaps[0]["start"] <= 10 and gaps[0]["duration_s"] > 3


def test_hv_battery_rules():
    rules = {p["rule"] for p in anomaly.analyze(frames_for("DEMO-003"))["plausibility_violations"]}
    assert rules == {"battery_overtemp", "cell_imbalance"}
