#!/usr/bin/env python3
"""
SPEAR Remote ID backend for **offline PCAP sidecars** (Wi‑Fi / NAN / BLE capture),
not for raw bladeRF IQ alone.

ASTM Remote ID is usually carried over Wi‑Fi (beacon / NAN) or Bluetooth LE. A
wideband IQ recording from the SDR does not automatically yield link-layer frames;
you typically need a parallel monitor-mode capture saved next to the IQ file.

Sidecar search (same directory as ``samples.iq``, i.e. the ``iq/`` folder):

  - remote_rid.pcapng
  - remote_rid.pcap
  - rid_sidecar.pcapng
  - rid_sidecar.pcap

Environment:

  SPEAR_RID_PCAP_DECODER_CMD
      Required for decode attempts. After substituting placeholders, the string
      is parsed with ``shlex.split`` and executed with ``subprocess.run``.

      Placeholders (use exactly as shown):

        {PCAP}   absolute path to the sidecar file
        {OUT}    absolute path to ``--output-json`` (your tool should write SPEAR
                 JSON here, or write JSON to stdout — see below)

      Example (hypothetical tool that writes JSON):

        SPEAR_RID_PCAP_DECODER_CMD=/usr/local/bin/my-rid-decoder --pcap {PCAP} --json {OUT}

      If your decoder prints JSON to stdout only, wrap with a shell script that
      redirects into ``{OUT}``.

If ``SPEAR_RID_PCAP_DECODER_CMD`` is unset, this backend writes ``no_decode`` with
reason ``rid_pcap_decoder_cmd_not_configured``.

If no sidecar PCAP is found, writes ``no_decode`` with reason ``rid_pcap_sidecar_not_found``.
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
from typing import Any, Dict, List, Optional


_SIDECAR_NAMES: List[str] = [
    "remote_rid.pcapng",
    "remote_rid.pcap",
    "rid_sidecar.pcapng",
    "rid_sidecar.pcap",
]


def _find_sidecar(iq_path: Path) -> Optional[Path]:
    parent = iq_path.parent
    for name in _SIDECAR_NAMES:
        p = parent / name
        if p.is_file():
            return p.resolve()
    return None


def _load_payload(out_path: Path) -> Optional[Dict[str, Any]]:
    if not out_path.is_file():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _no_decode(reason: str, **extra: Any) -> Dict[str, Any]:
    return {
        "protocol": "remote_id",
        "status": "no_decode",
        "confidence": 0.0,
        "validation": {"crc_pass": False, "frame_count": 0},
        "decoded_fields": {},
        "evidence": {"reason": reason, **extra},
        "generated_at_ts": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEAR RID PCAP sidecar backend")
    parser.add_argument("--iq-path", required=True)
    parser.add_argument("--center-freq-hz", required=True, type=int)
    parser.add_argument("--sample-rate-sps", required=True, type=int)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    iq_path = Path(args.iq_path).resolve()
    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_template = os.getenv("SPEAR_RID_PCAP_DECODER_CMD", "").strip()
    if not cmd_template:
        payload = _no_decode(
            "rid_pcap_decoder_cmd_not_configured",
            iq_path=str(iq_path),
            center_freq_hz=int(args.center_freq_hz),
            sample_rate_sps=int(args.sample_rate_sps),
            sidecar_names=_SIDECAR_NAMES,
            hint="Set SPEAR_RID_PCAP_DECODER_CMD to a decoder that accepts a PCAP and writes JSON to {OUT}.",
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    sidecar = _find_sidecar(iq_path)
    if sidecar is None:
        payload = _no_decode(
            "rid_pcap_sidecar_not_found",
            iq_path=str(iq_path),
            searched_in=str(iq_path.parent),
            sidecar_names=_SIDECAR_NAMES,
            center_freq_hz=int(args.center_freq_hz),
            sample_rate_sps=int(args.sample_rate_sps),
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    if "{PCAP}" not in cmd_template or "{OUT}" not in cmd_template:
        payload = {
            "protocol": "remote_id",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "rid_pcap_decoder_cmd_missing_placeholders",
                "cmd_template": cmd_template,
                "required_placeholders": ["{PCAP}", "{OUT}"],
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    filled = cmd_template.replace("{PCAP}", str(sidecar)).replace("{OUT}", str(out_path))
    try:
        argv = shlex.split(filled)
    except ValueError as e:
        payload = {
            "protocol": "remote_id",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "rid_pcap_decoder_cmd_unparseable",
                "error": str(e),
                "cmd_template": cmd_template,
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    timeout_s = float(os.getenv("SPEAR_RID_PCAP_TIMEOUT_S", "120"))
    try:
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as e:
        payload = {
            "protocol": "remote_id",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "rid_pcap_decoder_invocation_failed",
                "error": str(e),
                "argv": argv,
                "pcap": str(sidecar),
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    if proc.returncode != 0:
        payload = {
            "protocol": "remote_id",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "rid_pcap_decoder_nonzero_exit",
                "returncode": int(proc.returncode),
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-2000:],
                "argv": argv,
                "pcap": str(sidecar),
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    data = _load_payload(out_path)
    if data is None and (proc.stdout or "").strip():
        try:
            data = json.loads((proc.stdout or "").strip())
        except Exception:
            data = None

    if data is None:
        payload = {
            "protocol": "remote_id",
            "status": "decode_error",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": "rid_pcap_decoder_no_json_output",
                "argv": argv,
                "pcap": str(sidecar),
                "stdout": (proc.stdout or "")[-2000:],
            },
            "generated_at_ts": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
