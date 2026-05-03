from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


def _rid_decode_useful(payload: Dict[str, Any]) -> bool:
    st = str(payload.get("status", "")).lower()
    fields = payload.get("decoded_fields") or {}
    return st in ("decoded_verified", "decoded_partial") and bool(fields)


def _default_rid_decoder_cmd() -> str:
    """Resolve ``scripts/rid_decode_wrapper.py`` for SPEAR_RID_DECODER_CMD default."""
    candidates: list[Path] = []
    root = os.environ.get("SPEAR_EDGE_ROOT", "").strip()
    if root:
        candidates.append(Path(root) / "scripts" / "rid_decode_wrapper.py")
    here = Path(__file__).resolve()
    candidates.append(here.parents[3] / "scripts" / "rid_decode_wrapper.py")
    candidates.append(Path.cwd() / "scripts" / "rid_decode_wrapper.py")
    for w in candidates:
        if w.is_file():
            return f"{shlex.quote(sys.executable)} {shlex.quote(str(w))}"
    return ""


class RemoteIdDecoder:
    """
    ASTM Remote ID decode artifact producer for **Wi‑Fi / PCAP workflows**, not the
    bladeRF IQ capture path. SPEAR does not run this from ``CaptureManager``; live RID
    comes from the Wi‑Fi monitor stack (e.g. Kismet → ``rid_update`` on ``/wifi``).

    ``produce_artifact`` is for tooling, tests, or manual sidecars next to an IQ file
    if you choose that workflow offline.
    """

    protocol_name = "remote_id"
    artifact_relpath = Path("decode/remote_id.json")

    def decode(
        self,
        *,
        iq_path: Path,
        center_freq_hz: int,
        sample_rate_sps: int,
    ) -> Dict[str, Any]:
        artifact = self._try_load_artifact(iq_path)
        if artifact is not None:
            return artifact

        in_rid_band = (
            2_300_000_000 <= int(center_freq_hz) <= 2_500_000_000
            or 5_700_000_000 <= int(center_freq_hz) <= 5_950_000_000
        )
        return {
            "protocol": self.protocol_name,
            "status": "no_decode",
            "confidence": 0.0,
            "validation": {
                "crc_pass": False,
                "frame_count": 0,
            },
            "decoded_fields": {},
            "evidence": {
                "reason": "no_remote_id_decode_artifact_found",
                "rid_band_candidate": bool(in_rid_band),
                "iq_path": str(iq_path),
                "center_freq_hz": int(center_freq_hz),
                "sample_rate_sps": int(sample_rate_sps),
            },
        }

    def produce_artifact(
        self,
        *,
        iq_path: Path,
        center_freq_hz: int,
        sample_rate_sps: int,
    ) -> Dict[str, Any]:
        """
        Produce a standardized RID decode artifact.

        1. Optional built-in ASTM Wi‑Fi beacon decode from a PCAP sidecar beside the IQ.
        2. ``SPEAR_RID_DECODER_CMD``, or auto-resolved ``scripts/rid_decode_wrapper.py``.
        """
        in_rid_band = (
            2_300_000_000 <= int(center_freq_hz) <= 2_500_000_000
            or 5_700_000_000 <= int(center_freq_hz) <= 5_950_000_000
        )
        artifact = self._artifact_path(iq_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)

        if not in_rid_band:
            payload = self._no_decode_payload(
                reason="center_freq_not_in_rid_band",
                iq_path=iq_path,
                center_freq_hz=center_freq_hz,
                sample_rate_sps=sample_rate_sps,
                rid_band_candidate=False,
            )
            self._write_artifact(artifact, payload)
            return payload

        from spear_edge.core.decode.rid_wifi_pcap_builtin import (
            decode_astm_from_pcap,
            find_rid_sidecar,
        )

        sidecar = find_rid_sidecar(iq_path)
        if sidecar is not None:
            try:
                pcap_payload = decode_astm_from_pcap(sidecar)
                if _rid_decode_useful(pcap_payload):
                    self._write_artifact(artifact, pcap_payload)
                    return pcap_payload
            except Exception as e:
                print(f"[RID] built-in PCAP decode failed: {e}")

        cmd = os.getenv("SPEAR_RID_DECODER_CMD", "").strip()
        if not cmd:
            cmd = _default_rid_decoder_cmd()
        if not cmd:
            payload = self._no_decode_payload(
                reason="rid_decoder_command_not_configured",
                iq_path=iq_path,
                center_freq_hz=center_freq_hz,
                sample_rate_sps=sample_rate_sps,
                rid_band_candidate=True,
            )
            self._write_artifact(artifact, payload)
            return payload

        payload = self._run_external_decoder(
            cmd=cmd,
            iq_path=iq_path,
            center_freq_hz=center_freq_hz,
            sample_rate_sps=sample_rate_sps,
            artifact_path=artifact,
        )
        self._write_artifact(artifact, payload)
        return payload

    def _try_load_artifact(self, iq_path: Path) -> Dict[str, Any] | None:
        capture_dir = iq_path.parent.parent
        candidates = (
            capture_dir / "decode" / "remote_id.json",
            capture_dir / "features" / "remote_id.json",
            capture_dir / "remote_id.decode.json",
        )
        for artifact_path in candidates:
            if not artifact_path.exists():
                continue
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception as e:
                return {
                    "protocol": self.protocol_name,
                    "status": "decode_error",
                    "confidence": 0.0,
                    "validation": {"crc_pass": False, "frame_count": 0},
                    "decoded_fields": {},
                    "evidence": {
                        "reason": "artifact_parse_error",
                        "artifact_path": str(artifact_path),
                        "error": str(e),
                    },
                }

            validation = payload.get("validation", {})
            crc_pass = bool(validation.get("crc_pass", False))
            frame_count = int(validation.get("frame_count", 0) or 0)
            decoded_fields = payload.get("decoded_fields", {}) or {}
            status = str(payload.get("status", "")).lower()
            if status not in {"decoded_verified", "decoded_partial", "no_decode"}:
                if crc_pass and frame_count > 0:
                    status = "decoded_verified"
                elif decoded_fields:
                    status = "decoded_partial"
                else:
                    status = "no_decode"

            confidence = float(payload.get("confidence", 0.0) or 0.0)
            if status == "decoded_verified" and confidence <= 0.0:
                confidence = 0.99
            if status == "decoded_partial" and confidence <= 0.0:
                confidence = 0.5

            return {
                "protocol": self.protocol_name,
                "status": status,
                "confidence": confidence,
                "validation": {
                    "crc_pass": crc_pass,
                    "frame_count": frame_count,
                },
                "decoded_fields": decoded_fields,
                "evidence": {
                    "reason": "artifact_decode",
                    "artifact_path": str(artifact_path),
                },
            }

        return None

    def _artifact_path(self, iq_path: Path) -> Path:
        capture_dir = iq_path.parent.parent
        return capture_dir / self.artifact_relpath

    def _write_artifact(self, artifact_path: Path, payload: Dict[str, Any]) -> None:
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _no_decode_payload(
        self,
        *,
        reason: str,
        iq_path: Path,
        center_freq_hz: int,
        sample_rate_sps: int,
        rid_band_candidate: bool,
    ) -> Dict[str, Any]:
        return {
            "protocol": self.protocol_name,
            "status": "no_decode",
            "confidence": 0.0,
            "validation": {"crc_pass": False, "frame_count": 0},
            "decoded_fields": {},
            "evidence": {
                "reason": reason,
                "rid_band_candidate": rid_band_candidate,
                "iq_path": str(iq_path),
                "center_freq_hz": int(center_freq_hz),
                "sample_rate_sps": int(sample_rate_sps),
            },
            "generated_at_ts": time.time(),
        }

    def _run_external_decoder(
        self,
        *,
        cmd: str,
        iq_path: Path,
        center_freq_hz: int,
        sample_rate_sps: int,
        artifact_path: Path,
    ) -> Dict[str, Any]:
        base = shlex.split(cmd)
        full_cmd = base + [
            "--iq-path",
            str(iq_path),
            "--center-freq-hz",
            str(int(center_freq_hz)),
            "--sample-rate-sps",
            str(int(sample_rate_sps)),
            "--output-json",
            str(artifact_path),
        ]
        try:
            proc = subprocess.run(
                full_cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as e:
            return {
                "protocol": self.protocol_name,
                "status": "decode_error",
                "confidence": 0.0,
                "validation": {"crc_pass": False, "frame_count": 0},
                "decoded_fields": {},
                "evidence": {
                    "reason": "external_decoder_invocation_failed",
                    "error": str(e),
                    "command": full_cmd,
                },
                "generated_at_ts": time.time(),
            }

        if proc.returncode != 0:
            return {
                "protocol": self.protocol_name,
                "status": "decode_error",
                "confidence": 0.0,
                "validation": {"crc_pass": False, "frame_count": 0},
                "decoded_fields": {},
                "evidence": {
                    "reason": "external_decoder_nonzero_exit",
                    "returncode": int(proc.returncode),
                    "stderr": proc.stderr[-4000:],
                    "stdout": proc.stdout[-1000:],
                },
                "generated_at_ts": time.time(),
            }

        # Decoder may print JSON to stdout or write file directly.
        stdout_txt = (proc.stdout or "").strip()
        if stdout_txt:
            try:
                payload = json.loads(stdout_txt)
                payload.setdefault("protocol", self.protocol_name)
                payload.setdefault("generated_at_ts", time.time())
                return payload
            except Exception:
                pass

        if artifact_path.exists():
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                payload.setdefault("protocol", self.protocol_name)
                payload.setdefault("generated_at_ts", time.time())
                return payload
            except Exception as e:
                return {
                    "protocol": self.protocol_name,
                    "status": "decode_error",
                    "confidence": 0.0,
                    "validation": {"crc_pass": False, "frame_count": 0},
                    "decoded_fields": {},
                    "evidence": {
                        "reason": "external_decoder_output_unreadable",
                        "error": str(e),
                    },
                    "generated_at_ts": time.time(),
                }

        return self._no_decode_payload(
            reason="external_decoder_no_output",
            iq_path=iq_path,
            center_freq_hz=center_freq_hz,
            sample_rate_sps=sample_rate_sps,
            rid_band_candidate=True,
        )
