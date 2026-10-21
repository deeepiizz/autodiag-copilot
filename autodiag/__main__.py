"""Command-line interface.

    python -m autodiag ask "DEMO-001 has a rough idle, what should I check?"
    python -m autodiag ask "What does P0420 mean?" --offline
    python -m autodiag decode 0x1A0 0000861072107C10
    python -m autodiag vehicles
"""
import argparse
import json

from . import agent
from .tools import decode_can_frame, get_vehicle_snapshot


def main():
    p = argparse.ArgumentParser(prog="autodiag", description="Automotive diagnostic assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="ask a diagnostic question")
    a.add_argument("question")
    a.add_argument("--offline", action="store_true", help="skip Claude even if credentials are set")
    d = sub.add_parser("decode", help="decode a raw CAN frame")
    d.add_argument("can_id")
    d.add_argument("data_hex")
    sub.add_parser("vehicles", help="list simulated vehicles")
    args = p.parse_args()

    if args.cmd == "ask":
        ans = agent.ask(args.question, use_claude=False if args.offline else None)
        if ans.warnings:
            print("SAFETY:\n" + "\n".join(f"  ! {w}" for w in ans.warnings) + "\n")
        print(ans.text)
        print(f"\n[mode: {ans.mode}]")
    elif args.cmd == "decode":
        print(json.dumps(json.loads(decode_can_frame(args.can_id, args.data_hex)), indent=2))
    else:
        print(json.dumps(json.loads(get_vehicle_snapshot("list")), indent=2))


if __name__ == "__main__":
    main()
