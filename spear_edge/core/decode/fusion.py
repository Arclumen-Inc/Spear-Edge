from __future__ import annotations

from typing import Any, Dict, Optional


def _is_verified(protocol_result: Optional[Dict[str, Any]]) -> bool:
    if not protocol_result:
        return False
    return str(protocol_result.get("status", "")).lower() == "decoded_verified"


def _is_partial(protocol_result: Optional[Dict[str, Any]]) -> bool:
    if not protocol_result:
        return False
    return str(protocol_result.get("status", "")).lower() == "decoded_partial"


def fuse_protocol_and_ml(
    *,
    protocol_result: Optional[Dict[str, Any]],
    ml_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Phase 1 policy:
    - decoded_verified => protocol authoritative, ML shadow
    - decoded_partial  => partial_protocol_match
    - no decode        => ML primary (when available)
    """
    if _is_verified(protocol_result):
        return {
            "source": "protocol",
            "status": "protocol_primary",
            "label": protocol_result.get("protocol", "protocol_decode"),
            "confidence": float(protocol_result.get("confidence", 0.0)),
            "evidence": {
                "policy": "decoded_verified_protocol_primary",
                "protocol_status": protocol_result.get("status"),
                "ml_shadow_present": ml_result is not None,
            },
        }

    if _is_partial(protocol_result):
        return {
            "source": "fusion",
            "status": "partial_protocol_match",
            "label": protocol_result.get("protocol", "partial_protocol_match"),
            "confidence": float(protocol_result.get("confidence", 0.0)),
            "evidence": {
                "policy": "decoded_partial_with_ml_shadow",
                "protocol_status": protocol_result.get("status"),
                "ml_shadow_present": ml_result is not None,
            },
        }

    if ml_result is not None:
        return {
            "source": "ml",
            "status": "ml_primary",
            "label": ml_result.get("label", "unknown"),
            "confidence": float(ml_result.get("confidence", 0.0)),
            "evidence": {
                "policy": "no_protocol_decode_ml_primary",
                "ml_shadow_present": True,
            },
        }

    return {
        "source": "none",
        "status": "no_decision",
        "label": "unknown",
        "confidence": 0.0,
        "evidence": {
            "policy": "no_protocol_decode_no_ml_result",
            "ml_shadow_present": False,
        },
    }
