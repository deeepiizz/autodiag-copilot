"""CAN frame decoding with a real DBC database (via cantools) and candump log parsing."""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cantools

DBC_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "demo_vehicle.dbc"

# candump -L format: "(1700000000.123456) can0 1A0#0000861072107C10"
CANDUMP_RE = re.compile(r"^\((?P<ts>\d+\.\d+)\)\s+(?P<iface>\S+)\s+(?P<id>[0-9A-Fa-f]{3,8})#(?P<data>[0-9A-Fa-f]*)\s*$")


@dataclass
class Frame:
    timestamp: float
    can_id: int
    data: bytes


@lru_cache
def load_db(path=DBC_PATH):
    return cantools.database.load_file(str(path))


def _normalise_id(can_id):
    return int(can_id, 16) if isinstance(can_id, str) else int(can_id)


def decode_frame(can_id, data_hex, db=None):
    """Decode one frame into {signal: {"value", "unit"}}.

    Raises KeyError for an ID not in the DBC and ValueError for bad hex.
    """
    db = db or load_db()
    frame_id = _normalise_id(can_id)
    try:
        msg = db.get_message_by_frame_id(frame_id)
    except KeyError:
        known = ", ".join(f"0x{m.frame_id:03X}" for m in db.messages)
        raise KeyError(f"Unknown CAN ID 0x{frame_id:03X}. Known IDs: {known}") from None
    data = bytes.fromhex(data_hex.replace(" ", "")) if isinstance(data_hex, str) else bytes(data_hex)
    if len(data) > 8:
        raise ValueError("Classic CAN frames carry at most 8 data bytes")
    values = msg.decode(data.ljust(msg.length, b"\x00"), decode_choices=False)
    signals = {s.name: {"value": round(float(values[s.name]), 3), "unit": s.unit or ""} for s in msg.signals}
    return {"id": f"0x{frame_id:03X}", "message": msg.name, "sender": (msg.senders or ["?"])[0], "signals": signals}


def encode_frame(can_id, values, db=None):
    """Inverse of decode_frame; signals not given are set to their raw zero value."""
    db = db or load_db()
    msg = db.get_message_by_frame_id(_normalise_id(can_id))
    full = {s.name: values.get(s.name, s.offset) for s in msg.signals}
    return msg.encode(full, strict=False).hex().upper()


def parse_candump(text):
    """Parse candump -L log text into Frames, skipping blank and comment lines."""
    frames = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = CANDUMP_RE.match(line.strip())
        if not m:
            raise ValueError(f"line {lineno}: not candump -L format: {line!r}")
        frames.append(Frame(float(m["ts"]), int(m["id"], 16), bytes.fromhex(m["data"])))
    return frames


def format_candump(frames, iface="can0"):
    return "\n".join(f"({f.timestamp:.6f}) {iface} {f.can_id:03X}#{f.data.hex().upper()}" for f in frames) + "\n"
