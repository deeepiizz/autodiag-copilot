"""Decode and encode CAN frames with a simple DBC-style signal database."""
import json
from functools import lru_cache
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "can_signals.json"


@lru_cache
def load_db(path=DB_PATH):
    return json.loads(Path(path).read_text())["messages"]


def _normalise_id(can_id):
    value = int(can_id, 16) if isinstance(can_id, str) else int(can_id)
    return f"0x{value:03X}"


def _raw_to_int(data: bytes):
    return int.from_bytes(data.ljust(8, b"\x00"), "little")


def decode_frame(can_id, data_hex, db=None):
    """Decode one frame into {signal: {"value", "unit"}}.

    Raises KeyError for an unknown message ID and ValueError for bad hex.
    """
    db = db or load_db()
    key = _normalise_id(can_id)
    if key not in db:
        raise KeyError(f"Unknown CAN ID {key}. Known IDs: {', '.join(sorted(db))}")
    data = bytes.fromhex(data_hex.replace(" ", ""))
    if len(data) > 8:
        raise ValueError("Classic CAN frames carry at most 8 data bytes")

    raw = _raw_to_int(data)
    msg = db[key]
    signals = {}
    for sig in msg["signals"]:
        value = (raw >> sig["start_bit"]) & ((1 << sig["length"]) - 1)
        if sig["signed"] and value & (1 << (sig["length"] - 1)):
            value -= 1 << sig["length"]
        signals[sig["name"]] = {"value": round(value * sig["factor"] + sig["offset"], 3), "unit": sig["unit"]}
    return {"id": key, "message": msg["name"], "sender": msg["sender"], "signals": signals}


def encode_frame(can_id, values, db=None):
    """Inverse of decode_frame; used to build test data and simulated vehicles."""
    db = db or load_db()
    msg = db[_normalise_id(can_id)]
    raw = 0
    for sig in msg["signals"]:
        physical = values.get(sig["name"], sig["offset"])
        value = round((physical - sig["offset"]) / sig["factor"])
        value &= (1 << sig["length"]) - 1  # two's complement for signed values
        raw |= value << sig["start_bit"]
    n_bytes = max((s["start_bit"] + s["length"] + 7) // 8 for s in msg["signals"])
    return raw.to_bytes(8, "little")[:n_bytes].hex().upper()
