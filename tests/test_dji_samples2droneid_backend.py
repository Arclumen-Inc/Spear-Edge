import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_dji_samples2_backend_rejects_unsupported_rate(tmp_path: Path):
    iq = tmp_path / "samples.iq"
    iq.write_bytes(b"\x00" * 64)
    out = tmp_path / "out.json"
    cmd = [
        "python",
        str(_repo_root() / "scripts/dji_samples2droneid_backend.py"),
        "--iq-path",
        str(iq),
        "--center-freq-hz",
        "2429500000",
        "--sample-rate-sps",
        "10000000",
        "--output-json",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(_repo_root()))
    assert proc.returncode == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "decode_error"
    assert "unsupported_sample_rate" in data["evidence"]["reason"]


def test_dji_samples2_backend_maps_stdout_json(tmp_path: Path):
    iq = tmp_path / "samples.iq"
    iq.write_bytes(b"\x00" * 64)
    out = tmp_path / "out.json"

    fake_stdout = json.dumps(
        {
            "msgtype": 16,
            "serial_no": "SN-XYZ",
            "latitude": 32.1,
            "longitude": -96.2,
            "altitude": 50.0,
            "home_latitude": 32.09,
            "home_longitude": -96.21,
            "phone_app_latitude": 32.08,
            "phone_app_longitude": -96.22,
            "crc": 1234,
            "yaw": 90.0,
            "product_type": 9,
        }
    )

    def fake_run(*_a, **_k):
        return subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=fake_stdout + "\n",
            stderr="",
        )

    script = str(_repo_root() / "scripts/dji_samples2droneid_backend.py")
    old_argv = sys.argv
    try:
        sys.argv = [
            script,
            "--iq-path",
            str(iq),
            "--center-freq-hz",
            "2429500000",
            "--sample-rate-sps",
            "15360000",
            "--output-json",
            str(out),
        ]
        with patch.dict(os.environ, {"SPEAR_DJI_SAMPLES2_CMD": "samples2djidroneid"}, clear=False):
            with patch("subprocess.run", fake_run):
                with pytest.raises(SystemExit) as exc:
                    runpy.run_path(script, run_name="__main__")
                assert exc.value.code in (0, None)
    finally:
        sys.argv = old_argv

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "decoded_verified"
    assert data["decoded_fields"]["uas_id"] == "SN-XYZ"
    assert data["decoded_fields"]["drone_position"]["lat"] == 32.1
