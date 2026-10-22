"""Diagnostic tools shared by the MCP server, the Claude agent and the offline mode.

Each tool returns a JSON string so it can be passed straight back to an LLM.
"""
import csv
import json
import re
from functools import lru_cache

from . import anomaly, can, uds
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


def search_service_manual(query: str, k: int = 4, include_dtc: bool = True) -> str:
    """Search the service manuals and DTC reference for passages relevant to `query`.

    Returns the top-k passages with source IDs that should be cited in answers.

    Args:
        query: What to look for, e.g. "lean code at idle vacuum leak".
        k: Number of passages to return (1-8).
        include_dtc: False to search only service-manual procedures, not DTC reference rows.
    """
    hits = default_retriever().search(query, k=max(1, min(int(k), 8)), include_dtc=include_dtc)
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


@lru_cache
def _frames(vehicle_id):
    v = _vehicles()[vehicle_id]
    return tuple(can.parse_candump((KB_DIR / v["can_log_file"]).read_text()))


def _resolve_vehicle(vehicle_id):
    vid = vehicle_id.strip().upper()
    if vid not in _vehicles():
        raise KeyError(f"Unknown vehicle {vehicle_id}. Available: {', '.join(_vehicles())}")
    return vid


def _uds_dtcs(vid):
    """DTCs read over UDS from the vehicle's log, keyed by ECU."""
    out = {}
    for ecu, session in uds.read_dtcs_from_frames(_frames(vid)).items():
        result = session["result"]
        if result == "no_response":
            out[ecu] = "no_response"
        elif result.get("type") == "negative_response":
            out[ecu] = f"negative_response ({result['nrc_name']})"
        else:
            out[ecu] = [{"dtc": d["dtc"], "status": d["status"], "flags": d["flags"]} for d in result.get("dtcs", [])]
    return out


def get_vehicle_snapshot(vehicle_id: str) -> str:
    """Read a (simulated) vehicle's description, symptoms, DTCs (read over UDS) and latest CAN signal values.

    Args:
        vehicle_id: e.g. "DEMO-001". Call with "list" to see available vehicles.
    """
    if vehicle_id.strip().lower() == "list":
        return json.dumps({vid: v["description"] for vid, v in _vehicles().items()})
    try:
        vid = _resolve_vehicle(vehicle_id)
    except KeyError as e:
        return json.dumps({"error": str(e)})
    v = _vehicles()[vid]
    latest = {}
    for f in _frames(vid):   # frames are time-ordered, so the last one per ID wins
        try:
            latest[f.can_id] = can.decode_frame(f.can_id, f.data)
        except KeyError:
            pass                                  # diagnostic (UDS) frames are not in the DBC
    dtcs_by_ecu = _uds_dtcs(vid)
    active = sorted({d["dtc"] for r in dtcs_by_ecu.values() if isinstance(r, list) for d in r})
    return json.dumps({"vehicle_id": vid, "description": v["description"], "symptoms": v["symptoms"],
                       "active_dtcs": active, "dtcs_by_ecu": dtcs_by_ecu, "freeze_frame": v["freeze_frame"],
                       "latest_signals": list(latest.values())})


def analyze_can_log(vehicle_id: str) -> str:
    """Analyse a vehicle's recorded CAN log for anomalies: signal spikes (robust z-score),
    missing messages (module dropping off the bus) and physically implausible values
    (e.g. one wheel reading zero while the others move). Times are seconds from log start.

    Args:
        vehicle_id: e.g. "DEMO-002".
    """
    try:
        vid = _resolve_vehicle(vehicle_id)
    except KeyError as e:
        return json.dumps({"error": str(e)})
    frames = _frames(vid)
    report = anomaly.analyze(frames)
    t0 = frames[0].timestamp
    for key in ("spikes", "message_gaps", "plausibility_violations"):
        for event in report[key]:
            event["start"] = round(event["start"] - t0, 2)
            event["end"] = round(event["end"] - t0, 2)
    return json.dumps({"vehicle_id": vid, **report})


def decode_uds_message(payload_hex: str) -> str:
    """Decode a UDS (ISO 14229) diagnostic payload, e.g. a 0x59 ReadDTCInformation response
    or a 0x7F negative response, into services, DTC codes and status flags.

    Args:
        payload_hex: Reassembled UDS payload in hex (without ISO-TP headers), e.g. "5902FF0171002F".
    """
    try:
        return json.dumps(uds.parse_uds(bytes.fromhex(payload_hex.replace(" ", ""))))
    except ValueError as e:
        return json.dumps({"error": str(e)})


TOOLS = [lookup_dtc, search_service_manual, decode_can_frame, get_vehicle_snapshot, analyze_can_log,
         decode_uds_message]
