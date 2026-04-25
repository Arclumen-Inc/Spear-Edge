import json
import os
import subprocess
from pathlib import Path


def test_dji_wrapper_normalizes_minimal_backend_payload(tmp_path: Path):
    iq_path = tmp_path / "samples.iq"
    iq_path.write_bytes(b"\x00" * 8)
    out_path = tmp_path / "decode" / "dji_droneid.json"

    backend_script = tmp_path / "backend_minimal.py"
    backend_script.write_text(
        "\n".join(
            [
                "import argparse, json, pathlib",
                "p=argparse.ArgumentParser()",
                "p.add_argument('--iq-path', required=True)",
                "p.add_argument('--center-freq-hz', required=True)",
                "p.add_argument('--sample-rate-sps', required=True)",
                "p.add_argument('--output-json', required=True)",
                "a=p.parse_args()",
                "payload={",
                "  'fields': {'id':'DJI-MIN-1','lat':32.8,'lon':-96.7},",
                "  'crc_pass': True,",
                "  'frame_count': 2",
                "}",
                "pathlib.Path(a.output_json).parent.mkdir(parents=True, exist_ok=True)",
                "pathlib.Path(a.output_json).write_text(json.dumps(payload), encoding='utf-8')",
                "print(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["SPEAR_DJI_BACKEND_CMD"] = f"python {backend_script}"
    cmd = [
        "python",
        "scripts/dji_decode_wrapper.py",
        "--iq-path",
        str(iq_path),
        "--center-freq-hz",
        "2437000000",
        "--sample-rate-sps",
        "10000000",
        "--output-json",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["status"] == "decoded_verified"
    assert data["decoded_fields"]["uas_id"] == "DJI-MIN-1"
    assert "latitude" in data["decoded_fields"]
    assert "longitude" in data["decoded_fields"]
    assert data["decoded_fields"]["drone_position"]["lat"] == data["decoded_fields"]["latitude"]
    assert data["decoded_fields"]["drone_position"]["lon"] == data["decoded_fields"]["longitude"]
