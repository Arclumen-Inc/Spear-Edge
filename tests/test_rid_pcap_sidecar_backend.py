import json
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_rid_pcap_sidecar_no_cmd(tmp_path: Path, monkeypatch):
    iq = tmp_path / "iq" / "samples.iq"
    iq.parent.mkdir(parents=True, exist_ok=True)
    iq.write_bytes(b"\x00" * 8)
    out = tmp_path / "decode" / "remote_id.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    script = str(_repo_root() / "scripts/rid_pcap_sidecar_backend.py")
    monkeypatch.delenv("SPEAR_RID_PCAP_DECODER_CMD", raising=False)
    old_argv = sys.argv
    try:
        sys.argv = [
            script,
            "--iq-path",
            str(iq),
            "--center-freq-hz",
            "2437000000",
            "--sample-rate-sps",
            "20000000",
            "--output-json",
            str(out),
        ]
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(script, run_name="__main__")
        assert exc.value.code in (0, None)
    finally:
        sys.argv = old_argv

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "no_decode"
    assert data["evidence"]["reason"] == "rid_pcap_decoder_cmd_not_configured"


def test_rid_pcap_sidecar_missing_sidecar(tmp_path: Path):
    iq = tmp_path / "iq" / "samples.iq"
    iq.parent.mkdir(parents=True, exist_ok=True)
    iq.write_bytes(b"\x00" * 8)
    out = tmp_path / "decode" / "remote_id.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    script = str(_repo_root() / "scripts/rid_pcap_sidecar_backend.py")
    old_argv = sys.argv
    try:
        sys.argv = [
            script,
            "--iq-path",
            str(iq),
            "--center-freq-hz",
            "2437000000",
            "--sample-rate-sps",
            "20000000",
            "--output-json",
            str(out),
        ]
        with patch.dict(
            os.environ,
            {"SPEAR_RID_PCAP_DECODER_CMD": "true {PCAP} {OUT}"},
            clear=False,
        ):
            with pytest.raises(SystemExit) as exc:
                runpy.run_path(script, run_name="__main__")
            assert exc.value.code in (0, None)
    finally:
        sys.argv = old_argv

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "no_decode"
    assert data["evidence"]["reason"] == "rid_pcap_sidecar_not_found"


def test_rid_pcap_sidecar_invokes_decoder(tmp_path: Path):
    iq_dir = tmp_path / "iq"
    iq_dir.mkdir(parents=True, exist_ok=True)
    iq = iq_dir / "samples.iq"
    iq.write_bytes(b"\x00" * 8)
    pcap = iq_dir / "remote_rid.pcap"
    pcap.write_bytes(b"\x0a\x0b\x0c\x0d")

    helper = tmp_path / "rid_writer.py"
    helper.write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "pcap = Path(sys.argv[1])",
                "out = Path(sys.argv[2])",
                "assert pcap.is_file()",
                "payload = {",
                '  "protocol": "remote_id",',
                '  "status": "decoded_verified",',
                '  "confidence": 0.9,',
                '  "validation": {"crc_pass": True, "frame_count": 2},',
                '  "decoded_fields": {"uas_id": "RID-PCAP-1", "latitude": 40.0, "longitude": -75.0},',
                '  "evidence": {"reason": "test_helper"},',
                '  "generated_at_ts": 1.0,',
                "}",
                "out.write_text(json.dumps(payload), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "decode" / "remote_id.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    dec_cmd = f"python3 {helper} {{PCAP}} {{OUT}}"
    script = str(_repo_root() / "scripts/rid_pcap_sidecar_backend.py")
    old_argv = sys.argv
    try:
        sys.argv = [
            script,
            "--iq-path",
            str(iq),
            "--center-freq-hz",
            "2437000000",
            "--sample-rate-sps",
            "20000000",
            "--output-json",
            str(out),
        ]
        with patch.dict(
            os.environ,
            {"SPEAR_RID_PCAP_DECODER_CMD": dec_cmd},
            clear=False,
        ):
            with pytest.raises(SystemExit) as exc:
                runpy.run_path(script, run_name="__main__")
            assert exc.value.code in (0, None)
    finally:
        sys.argv = old_argv

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "decoded_verified"
    assert data["decoded_fields"]["uas_id"] == "RID-PCAP-1"
