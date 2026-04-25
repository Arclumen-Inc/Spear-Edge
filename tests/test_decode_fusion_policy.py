from spear_edge.core.decode.fusion import fuse_protocol_and_ml


def test_fusion_prefers_verified_protocol_over_ml():
    protocol_result = {
        "protocol": "remote_id",
        "status": "decoded_verified",
        "confidence": 0.98,
    }
    ml_result = {"label": "dji_mavic", "confidence": 0.82}

    out = fuse_protocol_and_ml(protocol_result=protocol_result, ml_result=ml_result)
    assert out["source"] == "protocol"
    assert out["status"] == "protocol_primary"
    assert out["label"] == "remote_id"


def test_fusion_marks_partial_protocol_match():
    protocol_result = {
        "protocol": "dji_droneid",
        "status": "decoded_partial",
        "confidence": 0.51,
    }
    ml_result = {"label": "unknown", "confidence": 0.44}

    out = fuse_protocol_and_ml(protocol_result=protocol_result, ml_result=ml_result)
    assert out["source"] == "fusion"
    assert out["status"] == "partial_protocol_match"


def test_fusion_falls_back_to_ml_without_decode():
    protocol_result = {
        "protocol": "remote_id",
        "status": "no_decode",
        "confidence": 0.0,
    }
    ml_result = {"label": "elrs", "confidence": 0.72}

    out = fuse_protocol_and_ml(protocol_result=protocol_result, ml_result=ml_result)
    assert out["source"] == "ml"
    assert out["status"] == "ml_primary"
    assert out["label"] == "elrs"
