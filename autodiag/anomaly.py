"""Signal-level anomaly detection on decoded CAN logs.

Three complementary detectors:
  1. Robust spikes: rolling median / MAD z-score on each signal.
  2. Message timing: gaps much longer than a message's normal period (a module dropping off the bus).
  3. Physical plausibility rules: e.g. one wheel reads 0 while the others are moving.
"""
from collections import defaultdict

import numpy as np

from .can import load_db

# (description, predicate over a dict of signal values at one instant)
PLAUSIBILITY_RULES = {
    "wheel_speed_dropout": (
        "One wheel speed reads ~0 while the other wheels are moving",
        ["WheelSpeed_FL", "WheelSpeed_FR", "WheelSpeed_RL", "WheelSpeed_RR"],
        lambda v: min(v) < 1.0 and np.median(v) > 10.0,
    ),
    "low_system_voltage": ("12 V system below 11.8 V", ["BatteryVoltage12V"], lambda v: v[0] < 11.8),
    "battery_overtemp": ("HV battery max cell temperature above 55 degC", ["MaxCellTemp"], lambda v: v[0] > 55),
    "cell_imbalance": ("HV battery cell voltage spread above 0.3 V", ["CellVoltageSpread"], lambda v: v[0] > 0.3),
    "high_fuel_trim": ("Long-term fuel trim beyond +/-10 %", ["LTFT_B1"], lambda v: abs(v[0]) > 10),
}


def decode_series(frames, db=None):
    """frames -> {signal: (timestamps, values)} and {can_id: timestamps}."""
    db = db or load_db()
    known = {m.frame_id: m for m in db.messages}
    series = defaultdict(lambda: ([], []))
    arrivals = defaultdict(list)
    for f in frames:
        msg = known.get(f.can_id)
        if msg is None:
            continue
        arrivals[f.can_id].append(f.timestamp)
        for name, value in msg.decode(f.data.ljust(msg.length, b"\x00"), decode_choices=False).items():
            series[name][0].append(f.timestamp)
            series[name][1].append(float(value))
    return {k: (np.array(t), np.array(v)) for k, (t, v) in series.items()}, dict(arrivals)


def _merge_events(times, max_gap):
    """Group sorted timestamps into [start, end] ranges."""
    if len(times) == 0:
        return []
    ranges, start, prev = [], times[0], times[0]
    for t in times[1:]:
        if t - prev > max_gap:
            ranges.append((start, prev))
            start = t
        prev = t
    ranges.append((start, prev))
    return ranges


def robust_spikes(t, v, resolution=1e-3, window=21, threshold=6.0):
    """Indices where |v - rolling median| / (1.4826 * rolling MAD) exceeds threshold.

    `resolution` is the signal's DBC scale factor: quantised signals often have
    MAD = 0, so the noise scale is never allowed below one quantisation step.
    """
    if len(v) < window:
        return np.array([], dtype=int)
    half = window // 2
    padded = np.pad(v, half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    med = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - med[:, None]), axis=1)
    # Local MAD can collapse in short or padded windows, so floor it with the
    # signal's global noise level as well as its resolution.
    global_mad = np.median(np.abs(v - np.median(v)))
    scale = np.maximum(1.4826 * np.maximum(mad, global_mad), abs(resolution))
    return np.flatnonzero(np.abs(v - med) / scale > threshold)


def message_gaps(arrivals, factor=5.0):
    """Gaps longer than `factor` x the median period for each message ID."""
    gaps = []
    db = load_db()
    for can_id, ts in arrivals.items():
        ts = np.sort(np.array(ts))
        if len(ts) < 5:
            continue
        d = np.diff(ts)
        period = np.median(d)
        for i in np.flatnonzero(d > factor * period):
            gaps.append({"message": db.get_message_by_frame_id(can_id).name, "id": f"0x{can_id:03X}",
                         "start": round(float(ts[i]), 3), "end": round(float(ts[i + 1]), 3),
                         "duration_s": round(float(d[i]), 3), "normal_period_s": round(float(period), 3)})
    return gaps


def plausibility(series):
    events = []
    for rule, (desc, names, pred) in PLAUSIBILITY_RULES.items():
        if not all(n in series for n in names):
            continue
        t = series[names[0]][0]
        n = min(len(series[x][1]) for x in names)
        hits = [t[i] for i in range(n) if pred([series[x][1][i] for x in names])]
        for start, end in _merge_events(np.array(hits), max_gap=0.5):
            events.append({"rule": rule, "description": desc, "start": round(float(start), 3),
                           "end": round(float(end), 3)})
    return events


def analyze(frames):
    """Full report: per-signal stats, spikes, message gaps and plausibility violations."""
    series, arrivals = decode_series(frames)
    resolution = {s.name: s.scale for m in load_db().messages for s in m.signals}
    stats, spikes = {}, []
    for name, (t, v) in sorted(series.items()):
        stats[name] = {"min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
                       "mean": round(float(v.mean()), 3), "samples": int(len(v))}
        idx = robust_spikes(t, v, resolution.get(name, 1e-3))
        for start, end in _merge_events(t[idx], max_gap=0.5):
            in_range = (t >= start) & (t <= end)
            spikes.append({"signal": name, "start": round(float(start), 3), "end": round(float(end), 3),
                           "extreme": round(float(v[in_range][np.argmax(np.abs(v[in_range] - np.median(v)))]), 3)})
    duration = max((t.max() for t, _ in series.values()), default=0) - min((t.min() for t, _ in series.values()), default=0)
    return {"duration_s": round(float(duration), 2), "signals": stats, "spikes": spikes,
            "message_gaps": message_gaps(arrivals), "plausibility_violations": plausibility(series)}
