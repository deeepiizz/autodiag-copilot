"""Command-line interface.

    python -m autodiag ask "DEMO-001 has a rough idle, what should I check?"
    python -m autodiag ask "What does P0420 mean?" --offline
    python -m autodiag decode 0x1A0 0000861072107C10
    python -m autodiag analyze DEMO-002            # or a path to any candump -L log
    python -m autodiag uds 5902FF0171002F
    python -m autodiag vehicles
"""
import argparse
import json
from pathlib import Path

from . import agent, anomaly, can, uds
from .tools import analyze_can_log, decode_can_frame, decode_uds_message, get_vehicle_snapshot


def _print(obj):
    print(json.dumps(json.loads(obj) if isinstance(obj, str) else obj, indent=2))


def main():
    p = argparse.ArgumentParser(prog="autodiag", description="Automotive diagnostic assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="ask a diagnostic question")
    a.add_argument("question")
    a.add_argument("--offline", action="store_true", help="skip Claude even if credentials are set")
    d = sub.add_parser("decode", help="decode a raw CAN frame")
    d.add_argument("can_id")
    d.add_argument("data_hex")
    an = sub.add_parser("analyze", help="anomaly report + UDS DTCs for a vehicle ID or candump log file")
    an.add_argument("target")
    u = sub.add_parser("uds", help="decode a UDS payload")
    u.add_argument("payload_hex")
    sub.add_parser("vehicles", help="list simulated vehicles")
    args = p.parse_args()

    if args.cmd == "ask":
        ans = agent.ask(args.question, use_claude=False if args.offline else None)
        if ans.warnings:
            print("SAFETY:\n" + "\n".join(f"  ! {w}" for w in ans.warnings) + "\n")
        print(ans.text)
        print(f"\n[mode: {ans.mode}]")
    elif args.cmd == "decode":
        _print(decode_can_frame(args.can_id, args.data_hex))
    elif args.cmd == "analyze":
        path = Path(args.target)
        if path.is_file():
            frames = can.parse_candump(path.read_text())
            _print({**anomaly.analyze(frames), "uds_dtc_reads": uds.read_dtcs_from_frames(frames)})
        else:
            _print(analyze_can_log(args.target))
    elif args.cmd == "uds":
        _print(decode_uds_message(args.payload_hex))
    else:
        _print(get_vehicle_snapshot("list"))


if __name__ == "__main__":
    main()
