import json
from pathlib import Path

from spear_edge.core.decode.dji_droneid_decoder import DjiDroneIdDecoder


def test_dji_decoder_reads_artifact_verified(tmp_path: Path):
    cap_dir = tmp_path / "cap"
    iq_dir = cap_dir / "iq"
    decode_dir = cap_dir / "decode"
    iq_dir.mkdir(parents=True)
    decode_dir.mkdir(parents=True)
    iq_path = iq_dir / "samples.iq"
    iq_path.write_bytes(b"\x00" * 8)

    artifact = {
        "status": "decoded_verified",
        "confidence": 0.94,
        "validation": {"crc_pass": True, "frame_count": 3},
        "decoded_fields": {"uas_id": "DJI-STUB-123"},
    }
    (decode_dir / "dji_droneid.json").write_text(json.dumps(artifact), encoding="utf-8")

    out = DjiDroneIdDecoder().decode(
        iq_path=iq_path,
        center_freq_hz=2_437_000_000,
        sample_rate_sps=10_000_000,
    )
    assert out["status"] == "decoded_verified"
    assert out["decoded_fields"]["uas_id"] == "DJI-STUB-123"


def test_dji_produce_artifact_writes_standard_file(tmp_path: Path):
    cap_dir = tmp_path / "cap"
    iq_dir = cap_dir / "iq"
    iq_dir.mkdir(parents=True)
    iq_path = iq_dir / "samples.iq"
    iq_path.write_bytes(b"\x00" * 8)

    decoder = DjiDroneIdDecoder()
    produced = decoder.produce_artifact(
        iq_path=iq_path,
        center_freq_hz=2_437_000_000,
        sample_rate_sps=10_000_000,
    )
    artifact = cap_dir / "decode" / "dji_droneid.json"
    assert artifact.exists()

    on_disk = json.loads(artifact.read_text(encoding="utf-8"))
    assert on_disk["protocol"] == "dji_droneid"
    assert on_disk["status"] == "no_decode"
    assert produced["evidence"]["reason"] == "dji_decoder_command_not_configured"
