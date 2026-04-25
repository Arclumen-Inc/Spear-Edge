#!/usr/bin/env python3
"""
RID decode wrapper for SPEAR-Edge Phase 1B.

This script is designed to be called by CaptureManager via SPEAR_RID_DECODER_CMD.
It guarantees a JSON output artifact with a stable schema, even when no backend
decoder is installed yet.

Optional backend delegation:
  - Set SPEAR_RID_BACKEND_CMD to a command that accepts:
      --iq-path, --center-freq-hz, --sample-rate-sps, --output-json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


def _base_payload(reason: str, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "protocol": "remote_id",
        "status": "no_decode",
        "confidence": 0.0,
        "validation": {"crc_pass": False, "frame_count": 0},
        "decoded_fields": {},
        "evidence": {
            "reason": reason,
            "iq_path": args.iq_path,
            "center_freq_hz": int(args.center_freq_hz),
            "sample_rate_sps": int(args.sample_rate_sps),
        },
        "generated_at_ts": time.time(),
    }


def _normalize_payload(raw: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Canonical contract keys
    protocol = str(raw.get("protocol", "remote_id") or "remote_id")
    decoded_fields = dict(raw.get("decoded_fields", {}) or {})
    validation = dict(raw.get("validation", {}) or {})
    evidence = dict(raw.get("evidence", {}) or {})

    # Accept a few common alternate field names from external tools
    if not decoded_fields:
        decoded_fields = dict(raw.get("fields", {}) or raw.get("data", {}) or {})

    if "uas_id" not in decoded_fields:
        for k in ("uasId", "ua_id", "id", "serial"):
            if k in decoded_fields and decoded_fields[k]:
                decoded_fields["uas_id"] = str(decoded_fields[k])
                break

    # Normalize lat/lon aliases
    if "latitude" not in decoded_fields and "lat" in decoded_fields:
        decoded_fields["latitude"] = decoded_fields.get("lat")
    if "longitude" not in decoded_fields and "lon" in decoded_fields:
        decoded_fields["longitude"] = decoded_fields.get("lon")

    # Normalize structured position blocks if backend emits flat keys.
    if "drone_position" not in decoded_fields:
        d_lat = decoded_fields.get("latitude")
        d_lon = decoded_fields.get("longitude")
        d_alt = decoded_fields.get("altitude_m")
        if d_lat is not None and d_lon is not None:
            decoded_fields["drone_position"] = {"lat": d_lat, "lon": d_lon, "altitude_m": d_alt}

    # Common aliases for takeoff/operator positions.
    if "takeoff_position" not in decoded_fields:
        t_lat = decoded_fields.get("takeoff_lat") or decoded_fields.get("home_lat")
        t_lon = decoded_fields.get("takeoff_lon") or decoded_fields.get("home_lon")
        t_alt = decoded_fields.get("takeoff_altitude_m") or decoded_fields.get("home_altitude_m")
        if t_lat is not None and t_lon is not None:
            decoded_fields["takeoff_position"] = {"lat": t_lat, "lon": t_lon, "altitude_m": t_alt}

    if "operator_position" not in decoded_fields:
        o_lat = decoded_fields.get("operator_lat")
        o_lon = decoded_fields.get("operator_lon")
        o_alt = decoded_fields.get("operator_altitude_m")
        if o_lat is not None and o_lon is not None:
            decoded_fields["operator_position"] = {"lat": o_lat, "lon": o_lon, "altitude_m": o_alt}

    # Validation fallback from top-level
    crc_pass = bool(validation.get("crc_pass", raw.get("crc_pass", False)))
    frame_count = int(validation.get("frame_count", raw.get("frame_count", 0) or 0))
    confidence = float(raw.get("confidence", 0.0) or 0.0)

    status = str(raw.get("status", "")).lower().strip()
    if status not in {"decoded_verified", "decoded_partial", "no_decode", "decode_error"}:
        if crc_pass and frame_count > 0:
            status = "decoded_verified"
        elif decoded_fields:
            status = "decoded_partial"
        else:
            status = "no_decode"

    if status == "decoded_verified" and confidence <= 0.0:
        confidence = 0.99
    elif status == "decoded_partial" and confidence <= 0.0:
        confidence = 0.5
    elif status in {"no_decode", "decode_error"} and confidence < 0.0:
        confidence = 0.0

    out = {
        "protocol": protocol,
        "status": status,
        "confidence": confidence,
        "validation": {
            "crc_pass": crc_pass,
            "frame_count": frame_count,
        },
        "decoded_fields": decoded_fields,
        "evidence": {
            **evidence,
            "iq_path": args.iq_path,
            "center_freq_hz": int(args.center_freq_hz),
            "sample_rate_sps": int(args.sample_rate_sps),
        },
        "generated_at_ts": float(raw.get("generated_at_ts", time.time())),
    }
    return out


def _run_backend(args: argparse.Namespace, backend_cmd: str) -> Dict[str, Any]:
    cmd = shlex.split(backend_cmd) + [
        "--iq-path",
        args.iq_path,
        "--center-freq-hz",
        str(int(args.center_freq_hz)),
        "--sample-rate-sps",
        str(int(args.sample_rate_sps)),
        "--output-json",
        args.output_json,
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as e:
        payload = _base_payload("backend_invocation_failed", args)
        payload["status"] = "decode_error"
        payload["evidence"]["error"] = str(e)
        payload["evidence"]["backend_cmd"] = cmd
        return payload

    if proc.returncode != 0:
        payload = _base_payload("backend_nonzero_exit", args)
        payload["status"] = "decode_error"
        payload["evidence"]["returncode"] = int(proc.returncode)
        payload["evidence"]["stderr"] = (proc.stderr or "")[-4000:]
        payload["evidence"]["stdout"] = (proc.stdout or "")[-1000:]
        return payload

    # Prefer backend-written JSON file
    out_path = Path(args.output_json)
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            return _normalize_payload(payload, args)
        except Exception as e:
            payload = _base_payload("backend_output_unreadable", args)
            payload["status"] = "decode_error"
            payload["evidence"]["error"] = str(e)
            return payload

    # Fallback: parse backend stdout as JSON
    stdout_txt = (proc.stdout or "").strip()
    if stdout_txt:
        try:
            payload = json.loads(stdout_txt)
            return _normalize_payload(payload, args)
        except Exception:
            pass

    return _base_payload("backend_no_output", args)


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEAR RID decoder wrapper")
    parser.add_argument("--iq-path", required=True)
    parser.add_argument("--center-freq-hz", required=True, type=int)
    parser.add_argument("--sample-rate-sps", required=True, type=int)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend_cmd = os.getenv("SPEAR_RID_BACKEND_CMD", "").strip()
    if backend_cmd:
        payload = _run_backend(args, backend_cmd)
    else:
        payload = _base_payload("backend_not_configured", args)

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Also emit JSON on stdout for compatibility/inspection
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
