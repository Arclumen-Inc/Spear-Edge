#!/usr/bin/env python3
"""
Fake Remote ID backend decoder for end-to-end validation.

This script emulates a successful RID decode and is intended for local
integration testing of:
- capture -> decode artifact -> protocol panel -> fusion behavior

CLI contract (matches wrapper expectations):
  --iq-path --center-freq-hz --sample-rate-sps --output-json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEAR fake RID backend")
    parser.add_argument("--iq-path", required=True)
    parser.add_argument("--center-freq-hz", required=True, type=int)
    parser.add_argument("--sample-rate-sps", required=True, type=int)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    # Deterministic but plausible fake telemetry
    payload = {
        "protocol": "remote_id",
        "status": "decoded_verified",
        "confidence": 0.99,
        "validation": {"crc_pass": True, "frame_count": 4},
        "decoded_fields": {
            "uas_id": "RID-STUB-001",
            "operator_id": "OP-STUB-ALPHA",
            "drone_position": {
                "lat": 32.776700,
                "lon": -96.797000,
                "altitude_m": 121.0,
            },
            "takeoff_position": {
                "lat": 32.775900,
                "lon": -96.798200,
                "altitude_m": 118.0,
            },
            "operator_position": {
                "lat": 32.776100,
                "lon": -96.797600,
                "altitude_m": 120.0,
            },
            "speed_mps": 9.8,
            "heading_deg": 134.0,
        },
        "evidence": {
            "reason": "stub_backend_decode",
            "iq_path": args.iq_path,
            "center_freq_hz": int(args.center_freq_hz),
            "sample_rate_sps": int(args.sample_rate_sps),
        },
        "generated_at_ts": time.time(),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
