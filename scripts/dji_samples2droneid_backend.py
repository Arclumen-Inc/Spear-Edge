#!/usr/bin/env python3
"""
SPEAR backend: decode DJI DroneID (Occusync 2.0) from a bladeRF-style IQ capture
using the external `samples2djidroneid` tool (native binary or Docker).

IQ format from SPEAR captures: interleaved float32 I/Q per sample (same layout as
GNU Radio complex float32 / fc32), stored as numpy complex64 in `samples.iq`.

Upstream reference: https://github.com/anarkiwi/samples2djidroneid

Environment:
  SPEAR_DJI_SAMPLES2_CMD
      Native decoder argv0 (default: samples2djidroneid). Invoked as:
        <cmd> [SPEAR_DJI_SAMPLES2_EXTRA_ARGS...] --samp-rate <rate> <absolute_iq_path>
  SPEAR_DJI_SAMPLES2_EXTRA_ARGS
      Optional extra arguments (shlex-split) inserted before ``--samp-rate``.
  SPEAR_DJI_SAMPLES2_DOCKER_IMAGE
      If set, run via Docker instead of SPEAR_DJI_SAMPLES2_CMD:
        docker run --rm -v <parent>:<parent> <image> <absolute_iq_path>
  SPEAR_DJI_SAMPLES2_TIMEOUT_S
      Subprocess timeout in seconds (default: 180).

Sample rate: decoder supports ~15.36 Msps and ~30.72 Msps. Other rates return
decode_error with an explanatory message (capture settings must match for live tests).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SUPPORTED_SPS = {15_360_000, 30_720_000}


def _parse_json_lines(stdout: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _pick_best_frame(frames: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not frames:
        return None
    # Prefer full telemetry frames (msgtype 16 seen in upstream examples).
    for f in reversed(frames):
        if f.get("msgtype") == 16 and f.get("latitude") is not None:
            return f
    for f in reversed(frames):
        if f.get("latitude") is not None and f.get("longitude") is not None:
            return f
    return frames[-1]


def _map_to_spear_payload(
    frame: Dict[str, Any],
    *,
    iq_path: Path,
    center_freq_hz: int,
    sample_rate_sps: int,
    frame_count: int,
) -> Dict[str, Any]:
    crc = frame.get("crc")
    crc_ok = crc is not None
    lat, lon = frame.get("latitude"), frame.get("longitude")
    verified = bool(
        crc_ok
        and lat is not None
        and lon is not None
        and -90.0 <= float(lat) <= 90.0
        and -180.0 <= float(lon) <= 180.0
    )
    status = "decoded_verified" if verified else "decoded_partial"
    confidence = 0.99 if verified else 0.55

    decoded_fields: Dict[str, Any] = {
        "uas_id": frame.get("serial_no") or frame.get("uuid") or None,
        "model": str(frame.get("product_type", "")) if frame.get("product_type") is not None else None,
        "drone_position": {
            "lat": frame.get("latitude"),
            "lon": frame.get("longitude"),
            "altitude_m": frame.get("altitude") if frame.get("altitude") is not None else frame.get("height"),
        },
        "takeoff_position": {
            "lat": frame.get("home_latitude"),
            "lon": frame.get("home_longitude"),
            "altitude_m": None,
        },
        "operator_position": {
            "lat": frame.get("phone_app_latitude"),
            "lon": frame.get("phone_app_longitude"),
            "altitude_m": None,
        },
        "speed_mps": None,
        "heading_deg": frame.get("yaw"),
    }

    return {
        "protocol": "dji_droneid",
        "status": status,
        "confidence": float(confidence),
        "validation": {"crc_pass": bool(crc_ok), "frame_count": int(frame_count)},
        "decoded_fields": decoded_fields,
        "evidence": {
            "reason": "samples2djidroneid",
            "iq_path": str(iq_path),
            "center_freq_hz": int(center_freq_hz),
            "sample_rate_sps": int(sample_rate_sps),
            "msgtype": frame.get("msgtype"),
            "seqno": frame.get("seqno"),
        },
        "generated_at_ts": time.time(),
    }


def _build_cmd(iq_abs: Path, sample_rate_sps: int) -> Tuple[List[str], str]:
    """Upstream samples2djidroneid.py accepts --samp-rate (15.36e6 or 30.72e6) before the IQ path."""
    sr = str(float(int(sample_rate_sps)))
    image = os.getenv("SPEAR_DJI_SAMPLES2_DOCKER_IMAGE", "").strip()
    if image:
        parent = iq_abs.parent.resolve()
        pstr = str(parent)
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{pstr}:{pstr}",
            image,
            "--samp-rate",
            sr,
            str(iq_abs),
        ]
        return cmd, "docker"
    exe = os.getenv("SPEAR_DJI_SAMPLES2_CMD", "samples2djidroneid").strip()
    parts = shlex.split(exe)
    extra = os.getenv("SPEAR_DJI_SAMPLES2_EXTRA_ARGS", "").strip()
    extra_parts = shlex.split(extra) if extra else []
    cmd = parts + extra_parts + ["--samp-rate", sr, str(iq_abs)]
    return cmd, "native"


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEAR DJI samples2droneid backend")
    parser.add_argument("--iq-path", required=True)
    parser.add_argument("--center-freq-hz", required=True, type=int)
    parser.add_argument("--sample-rate-sps", required=True, type=int)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    iq_path = Path(args.iq_path).resolve()
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sps = int(args.sample_rate_sps)
    if sps not in _SUPPORTED_SPS:
        payload = {
            "protocol": "dji_droneid",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "unsupported_sample_rate_for_samples2djidroneid",
                "sample_rate_sps": sps,
                "supported_sample_rates_sps": sorted(_SUPPORTED_SPS),
                "hint": "Re-run capture at 15.36 Msps or 30.72 Msps for DJI DroneID decode.",
                "iq_path": str(iq_path),
                "center_freq_hz": int(args.center_freq_hz),
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    if not iq_path.is_file():
        payload = {
            "protocol": "dji_droneid",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "iq_file_missing",
                "iq_path": str(iq_path),
                "center_freq_hz": int(args.center_freq_hz),
                "sample_rate_sps": sps,
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    timeout_s = float(os.getenv("SPEAR_DJI_SAMPLES2_TIMEOUT_S", "180"))
    cmd, mode = _build_cmd(iq_path, sps)
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        payload = {
            "protocol": "dji_droneid",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "decoder_executable_not_found",
                "error": str(e),
                "cmd": cmd,
                "mode": mode,
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0
    except subprocess.TimeoutExpired:
        payload = {
            "protocol": "dji_droneid",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "decoder_timeout",
                "timeout_s": timeout_s,
                "cmd": cmd,
                "mode": mode,
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0
    except Exception as e:
        payload = {
            "protocol": "dji_droneid",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "decoder_invocation_failed",
                "error": str(e),
                "cmd": cmd,
                "mode": mode,
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    frames = _parse_json_lines(stdout)
    best = _pick_best_frame(frames)

    if proc.returncode != 0 and not best:
        payload = {
            "protocol": "dji_droneid",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "samples2djidroneid_nonzero_exit",
                "returncode": int(proc.returncode),
                "stderr": stderr[-4000:],
                "stdout": stdout[-2000:],
                "cmd": cmd,
                "mode": mode,
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    if not best:
        payload = {
            "protocol": "dji_droneid",
            "status": "no_decode",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "no_dji_frames_in_decoder_output",
                "returncode": int(proc.returncode),
                "stderr": stderr[-4000:],
                "stdout": stdout[-2000:],
                "cmd": cmd,
                "mode": mode,
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    payload = _map_to_spear_payload(
        best,
        iq_path=iq_path,
        center_freq_hz=int(args.center_freq_hz),
        sample_rate_sps=sps,
        frame_count=len(frames),
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
