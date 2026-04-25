import json
import subprocess
from pathlib import Path


def test_rid_backend_stub_emits_verified_decode(tmp_path: Path):
    iq_path = tmp_path / "samples.iq"
    iq_path.write_bytes(b"\x00" * 8)
    out_path = tmp_path / "decode" / "remote_id.json"

    cmd = [
        "python",
        "scripts/rid_backend_stub.py",
        "--iq-path",
        str(iq_path),
        "--center-freq-hz",
        "2437000000",
        "--sample-rate-sps",
        "10000000",
        "--output-json",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert out_path.exists()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["protocol"] == "remote_id"
    assert data["status"] == "decoded_verified"
    assert data["validation"]["crc_pass"] is True
    assert data["validation"]["frame_count"] > 0
    assert "uas_id" in data["decoded_fields"]
    assert "drone_position" in data["decoded_fields"]
    assert "takeoff_position" in data["decoded_fields"]
    assert "operator_position" in data["decoded_fields"]
