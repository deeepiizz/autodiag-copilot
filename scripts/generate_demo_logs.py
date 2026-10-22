"""Generate the simulated candump logs in knowledge_base/logs/ (seeded, reproducible).

Each log has 30 s of 10 Hz broadcast traffic with a fault pattern injected,
followed by a UDS ReadDTCInformation session per ECU over ISO-TP.

    python scripts/generate_demo_logs.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autodiag import uds  # noqa: E402
from autodiag.can import Frame, format_candump, load_db  # noqa: E402

T0, DURATION, RATE = 1760000000.0, 30.0, 10
LOG_DIR = ROOT / "knowledge_base" / "logs"
STATUS_CONFIRMED = 0x2F      # testFailed, thisCycle, pending, confirmed, sinceLastClear
STATUS_MIL = 0xAF            # ... plus warningIndicatorRequested


def broadcast(rng, msg_name, fn, skip=None):
    """Frames for one message at RATE Hz; fn(t, rng) -> signal dict; skip(t) drops frames."""
    msg = load_db().get_message_by_name(msg_name)
    frames = []
    for i in range(int(DURATION * RATE)):
        t = i / RATE + rng.uniform(0, 0.002)          # small scheduling jitter
        if skip and skip(t):
            continue
        frames.append(Frame(T0 + t, msg.frame_id, msg.encode(fn(t, rng), strict=False)))
    return frames


def dtc_session(t, ecu, dtcs, respond=True):
    """Tester request, flow control and ECU response for a 0x19 0x02 DTC read."""
    req_id = next(r for r, (name, _) in uds.DIAG_ADDRESSES.items() if name == ecu)
    resp_id = uds.DIAG_ADDRESSES[req_id][1]
    frames = [Frame(T0 + t, req_id, f) for f in uds.isotp_segment(uds.read_dtc_request(0xFF))]
    if not respond:
        return frames
    resp = uds.isotp_segment(uds.read_dtc_response(dtcs))
    frames.append(Frame(T0 + t + 0.012, resp_id, resp[0]))
    if len(resp) > 1:
        frames.append(Frame(T0 + t + 0.015, req_id, uds.FLOW_CONTROL_CTS))
        frames += [Frame(T0 + t + 0.02 + 0.002 * i, resp_id, f) for i, f in enumerate(resp[1:])]
    return frames


def body(v12):
    return lambda t, rng: {"BatteryVoltage12V": v12(t) + rng.normal(0, 0.03), "AmbientTemp": 18}


def demo_001(rng):
    misfire_times = rng.uniform(2, 28, 9)
    def engine(t, rng):
        dip = 140 if np.any(np.abs(t - misfire_times) < 0.06) else 0
        return {"EngineRPM": 780 + rng.normal(0, 8) - dip, "CoolantTemp": 88, "ThrottlePos": 0,
                "MAF": 2.1 + rng.normal(0, 0.04)}
    def trim(t, rng):
        return {"STFT_B1": 6 + rng.normal(0, 1.5), "LTFT_B1": 18, "STFT_B2": 0, "LTFT_B2": 0}
    return (broadcast(rng, "EngineData", engine) + broadcast(rng, "FuelTrim", trim)
            + broadcast(rng, "BodyStatus", body(lambda t: 14.1))
            + dtc_session(DURATION + 0.5, "ECM", [("P0171", 0x00, STATUS_CONFIRMED), ("P0300", 0x00, STATUS_MIL)]))


def demo_002(rng):
    dropouts = [(6.2, 7.0), (14.5, 15.8), (22.1, 22.6)]
    def wheels(t, rng):
        v = 42 + 3 * np.sin(t / 5)
        fl = 0.0 if any(a <= t <= b for a, b in dropouts) else v + rng.normal(0, 0.1)
        return {"WheelSpeed_FL": fl, "WheelSpeed_FR": v + rng.normal(0, 0.1),
                "WheelSpeed_RL": v + rng.normal(0, 0.1), "WheelSpeed_RR": v + rng.normal(0, 0.1)}
    return (broadcast(rng, "WheelSpeeds", wheels) + broadcast(rng, "BodyStatus", body(lambda t: 14.0))
            + dtc_session(DURATION + 0.5, "ABS", [("C0035", 0x31, STATUS_MIL)])
            + dtc_session(DURATION + 1.0, "ECM", []))


def demo_003(rng):
    def battery(t, rng):
        current = -35 + 25 * np.sin(t / 2) + rng.normal(0, 2)
        return {"PackVoltage": 201.6 + 0.08 * current + rng.normal(0, 0.2), "PackCurrent": current,
                "SOC": 41, "MaxCellTemp": 50 + t / 3, "CellVoltageSpread": 0.42 + rng.normal(0, 0.01)}
    return (broadcast(rng, "BatteryStatus", battery) + broadcast(rng, "BodyStatus", body(lambda t: 13.9))
            + dtc_session(DURATION + 0.5, "BMS", [("P0A80", 0x00, STATUS_MIL)]))


def demo_004(rng):
    def v12(t):
        return 10.7 if (9 <= t <= 11.5 or 20 <= t <= 21) else 11.9
    abs_offline = lambda t: 9.2 <= t <= 12.4     # ABS module browns out with the voltage dips
    wheels = lambda t, rng: {"WheelSpeed_FL": 0, "WheelSpeed_FR": 0, "WheelSpeed_RL": 0, "WheelSpeed_RR": 0}
    engine = lambda t, rng: {"EngineRPM": 720 + rng.normal(0, 6), "CoolantTemp": 90, "ThrottlePos": 0, "MAF": 2.6}
    return (broadcast(rng, "EngineData", engine) + broadcast(rng, "WheelSpeeds", wheels, skip=abs_offline)
            + broadcast(rng, "BodyStatus", body(v12))
            + dtc_session(DURATION + 0.5, "ECM", [("U0121", 0x00, STATUS_CONFIRMED), ("U0001", 0x00, STATUS_CONFIRMED),
                                                  ("P0562", 0x00, STATUS_CONFIRMED)])
            + dtc_session(DURATION + 1.0, "ABS", [], respond=False))


SCENARIOS = {"DEMO-001": demo_001, "DEMO-002": demo_002, "DEMO-003": demo_003, "DEMO-004": demo_004}


def main(out_dir=LOG_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (vid, fn) in enumerate(SCENARIOS.items()):
        frames = sorted(fn(np.random.default_rng(i)), key=lambda f: f.timestamp)
        (out_dir / f"{vid}.log").write_text(format_candump(frames))
        print(f"{vid}: {len(frames)} frames")


if __name__ == "__main__":
    main()
