"""ISO-TP (ISO 15765-2) reassembly and UDS (ISO 14229) diagnostic message decoding.

Covers what a DTC read needs: single/first/consecutive/flow-control frames,
service 0x19 ReadDTCInformation, and negative responses.
"""
from dataclasses import dataclass

# Diagnostic request -> response IDs (11-bit physical addressing) for the demo vehicle.
DIAG_ADDRESSES = {0x7E0: ("ECM", 0x7E8), 0x7E1: ("TCM", 0x7E9), 0x760: ("ABS", 0x768), 0x7E4: ("BMS", 0x7EC)}
RESPONSE_TO_ECU = {resp: name for name, resp in DIAG_ADDRESSES.values()}
REQUEST_TO_ECU = {req: name for req, (name, _) in DIAG_ADDRESSES.items()}

SERVICES = {0x10: "DiagnosticSessionControl", 0x11: "ECUReset", 0x14: "ClearDiagnosticInformation",
            0x19: "ReadDTCInformation", 0x22: "ReadDataByIdentifier", 0x3E: "TesterPresent"}

NRC = {0x10: "generalReject", 0x11: "serviceNotSupported", 0x12: "subFunctionNotSupported",
       0x13: "incorrectMessageLengthOrInvalidFormat", 0x22: "conditionsNotCorrect",
       0x31: "requestOutOfRange", 0x33: "securityAccessDenied", 0x78: "requestCorrectlyReceivedResponsePending"}

# DTC status byte bits (ISO 14229-1 D.2).
STATUS_BITS = ["testFailed", "testFailedThisOperationCycle", "pendingDTC", "confirmedDTC",
               "testNotCompletedSinceLastClear", "testFailedSinceLastClear",
               "testNotCompletedThisOperationCycle", "warningIndicatorRequested"]

SYSTEM_LETTERS = "PCBU"


# --- DTC encoding (SAE J2012 two-byte format + failure-type byte) ----------------

def decode_dtc(b: bytes):
    """3-byte UDS DTC -> ("P0171", failure_type_byte)."""
    hi, lo, ftb = b[0], b[1], b[2]
    code = f"{SYSTEM_LETTERS[hi >> 6]}{(hi >> 4) & 0x3}{hi & 0xF:X}{lo:02X}"
    return code, ftb


def encode_dtc(code: str, ftb: int = 0) -> bytes:
    letter, digits = code[0].upper(), code[1:]
    hi = (SYSTEM_LETTERS.index(letter) << 6) | (int(digits[0]) << 4) | int(digits[1], 16)
    return bytes([hi, int(digits[2:], 16), ftb])


def decode_status(status: int):
    return [name for bit, name in enumerate(STATUS_BITS) if status & (1 << bit)]


# --- ISO-TP ---------------------------------------------------------------------

def isotp_segment(payload: bytes, pad=0xAA):
    """Split a payload into ISO-TP CAN frames (8-byte classic CAN)."""
    if len(payload) <= 7:
        return [bytes([len(payload)]) + payload + bytes([pad] * (7 - len(payload)))]
    frames = [bytes([0x10 | (len(payload) >> 8), len(payload) & 0xFF]) + payload[:6]]
    rest, seq = payload[6:], 1
    while rest:
        chunk, rest = rest[:7], rest[7:]
        frames.append(bytes([0x20 | (seq & 0xF)]) + chunk + bytes([pad] * (7 - len(chunk))))
        seq += 1
    return frames


FLOW_CONTROL_CTS = bytes([0x30, 0x00, 0x00]) + bytes([0xAA] * 5)


@dataclass
class IsoTpMessage:
    can_id: int
    timestamp: float
    payload: bytes


def isotp_reassemble(frames):
    """Reassemble ISO-TP messages per CAN ID from (timestamp, can_id, data) frames.

    Flow-control frames are skipped. Out-of-sequence frames drop the partial message.
    """
    partial = {}   # can_id -> [timestamp, expected_len, buffer, next_seq]
    messages = []
    for ts, can_id, data in frames:
        pci = data[0] >> 4
        if pci == 0x0:
            n = data[0] & 0xF
            messages.append(IsoTpMessage(can_id, ts, bytes(data[1:1 + n])))
        elif pci == 0x1:
            n = ((data[0] & 0xF) << 8) | data[1]
            partial[can_id] = [ts, n, bytearray(data[2:8]), 1]
        elif pci == 0x2 and can_id in partial:
            start, n, buf, seq = partial[can_id]
            if data[0] & 0xF != seq & 0xF:
                del partial[can_id]
                continue
            buf += data[1:8]
            partial[can_id][3] += 1
            if len(buf) >= n:
                messages.append(IsoTpMessage(can_id, start, bytes(buf[:n])))
                del partial[can_id]
    return messages


# --- UDS ------------------------------------------------------------------------

def read_dtc_request(status_mask=0xFF):
    """0x19 0x02 reportDTCByStatusMask."""
    return bytes([0x19, 0x02, status_mask])


def read_dtc_response(dtcs, availability_mask=0xFF):
    """Build a positive 0x59 0x02 response from [(code, ftb, status), ...]."""
    out = bytearray([0x59, 0x02, availability_mask])
    for code, ftb, status in dtcs:
        out += encode_dtc(code, ftb) + bytes([status])
    return bytes(out)


def parse_uds(payload: bytes):
    """Decode a UDS request or response payload into a readable dict."""
    if not payload:
        return {"type": "empty"}
    sid = payload[0]
    if sid == 0x7F and len(payload) >= 3:
        return {"type": "negative_response", "service": SERVICES.get(payload[1], hex(payload[1])),
                "nrc": f"0x{payload[2]:02X}", "nrc_name": NRC.get(payload[2], "unknown")}
    if sid == 0x59 and len(payload) >= 3 and payload[1] == 0x02:
        records = []
        for i in range(3, len(payload) - 3, 4):
            code, ftb = decode_dtc(payload[i:i + 3])
            status = payload[i + 3]
            records.append({"dtc": code, "failure_type": f"0x{ftb:02X}", "status": f"0x{status:02X}",
                            "flags": decode_status(status)})
        return {"type": "positive_response", "service": "ReadDTCInformation",
                "subfunction": "reportDTCByStatusMask", "dtcs": records}
    if sid in SERVICES:
        return {"type": "request", "service": SERVICES[sid], "data": payload[1:].hex().upper()}
    if sid - 0x40 in SERVICES:
        return {"type": "positive_response", "service": SERVICES[sid - 0x40], "data": payload[1:].hex().upper()}
    return {"type": "unknown", "data": payload.hex().upper()}


def read_dtcs_from_frames(frames, timeout=1.0):
    """Find DTC-read sessions in a frame log.

    Returns one entry per ECU that was asked: its DTCs, a negative response,
    or "no_response" if nothing came back within `timeout` seconds.
    """
    diag = [(f.timestamp, f.can_id, f.data) for f in frames
            if f.can_id in REQUEST_TO_ECU or f.can_id in RESPONSE_TO_ECU]
    msgs = isotp_reassemble(diag)
    results = {}
    for m in msgs:
        if m.can_id in REQUEST_TO_ECU and m.payload[:2] == b"\x19\x02":
            results[REQUEST_TO_ECU[m.can_id]] = {"requested_at": m.timestamp, "result": "no_response"}
    for m in msgs:
        ecu = RESPONSE_TO_ECU.get(m.can_id)
        if ecu in results and m.timestamp - results[ecu]["requested_at"] <= timeout:
            parsed = parse_uds(m.payload)
            if parsed["type"] != "request":
                results[ecu]["result"] = parsed
    return results
