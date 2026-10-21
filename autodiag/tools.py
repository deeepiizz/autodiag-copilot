"""Diagnostic tools shared by the MCP server, the Claude agent and the offline mode.

Each tool returns a JSON string so it can be passed straight back to an LLM.
"""
import csv
import json
import re
from functools import lru_cache

from . import can
from .retriever import KB_DIR, default_retriever

DTC_RE = re.compile(r"\b[PCBU][0-9A-F]{4}\b", re.IGNORECASE)


@lru_cache
def _dtc_table():
    with open(KB_DIR / "dtc_codes.csv", newline="") as f:
        return {row["code"]: row for row in csv.DictReader(f)}


@lru_cache
def _vehicles():
    return json.loads((KB_DIR / "vehicles.json").read_text())["vehicles"]


def lookup_dtc(code: str) -> str:
    """Look up an OBD-II diagnostic trouble code (e.g. P0171, C0035, U0121).

    Returns the description, affected system, severity and common causes.
    """
    code = code.strip().upper()
    row = _dtc_table().get(code)
    if row is None:
        return json.dumps({"code": code, "found": False,
                           "message": "Code not in the knowledge base. It may be manufacturer-specific."})
    return json.dumps({
        "code": code, "found": True, "description": row["description"], "system": row["system"],
        "severity": row["severity"], "common_causes": row["common_causes"].split("; "),
        "source": f"dtc:{code}",
    })


def search_service_manual(query: str, k: int = 4) -> str:
    """Search the service manuals and DTC reference for passages relevant to `query`.

    Returns the top-k passages with source IDs that should be cited in answers.
    """
    hits = default_retriever().search(query, k=max(1, min(int(k), 8)))
    return json.dumps([{"source": c.id, "title": c.title, "text": c.text} for c in hits])


def decode_can_frame(can_id: str, data_hex: str) -> str:
    """Decode a raw CAN frame into physical signal values using the vehicle signal database.

    Args:
        can_id: 11-bit identifier in hex, e.g. "0x1A0".
        data_hex: Payload bytes in hex, e.g. "0000861072107C10".
    """
    try:
        return json.dumps(can.decode_frame(can_id, data_hex))
    except (KeyError, ValueError) as e:
        return json.dumps({"error": str(e)})


def get_vehicle_snapshot(vehicle_id: str) -> str:
    """Read a (simulated) vehicle's active DTCs, freeze-frame data and decoded CAN log.

    Args:
        vehicle_id: e.g. "DEMO-001". Call with "list" to see available vehicles.
    """
    vehicles = _vehicles()
    if vehicle_id.strip().lower() == "list":
        return json.dumps({vid: v["description"] for vid, v in vehicles.items()})
    v = vehicles.get(vehicle_id.strip().upper())
    if v is None:
        return json.dumps({"error": f"Unknown vehicle {vehicle_id}. Available: {', '.join(vehicles)}"})
    decoded = [can.decode_frame(f["id"], f["data"]) for f in v["can_log"]]
    return json.dumps({**{k: v[k] for k in ("description", "symptoms", "active_dtcs", "freeze_frame")},
                       "vehicle_id": vehicle_id.upper(), "can_log_decoded": decoded})


TOOLS = [lookup_dtc, search_service_manual, decode_can_frame, get_vehicle_snapshot]
