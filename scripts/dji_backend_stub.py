#!/usr/bin/env python3
"""
Fake DJI DroneID backend decoder for end-to-end validation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEAR fake DJI backend")
    parser.add_argument("--iq-path", required=True)
    parser.add_argument("--center-freq-hz", required=True, type=int)
    parser.add_argument("--sample-rate-sps", required=True, type=int)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = {
        "protocol": "dji_droneid",
        "status": "decoded_verified",
        "confidence": 0.97,
        "validation": {"crc_pass": True, "frame_count": 3},
        "decoded_fields": {
            "uas_id": "DJI-STUB-001",
            "model": "DJI-MAVIC-STUB",
            "drone_position": {
                "lat": 32.777200,
                "lon": -96.796400,
                "altitude_m": 82.0,
            },
            "takeoff_position": {
                "lat": 32.776700,
                "lon": -96.797000,
                "altitude_m": 80.0,
            },
            "operator_position": {
                "lat": 32.776900,
                "lon": -96.796800,
                "altitude_m": 81.0,
            },
            "speed_mps": 6.2,
            "heading_deg": 214.0,
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
