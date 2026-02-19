from fastapi import APIRouter
from spear_edge.core import state

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
def health():
    return {"ok": True}

@router.get("/status")
def status():
    return {
        "ok": True,
        "mode": getattr(state, "edge_mode", "manual"),
        "sdr_open": state.sdr is not None,
        "gps": state.gps.get_status() if state.gps else None,
    }

@router.get("/sdr")
def sdr_health():
    """
    Get SDR health metrics for monitoring dashboard.
    """
    if state.engine:
        return state.engine.get_sdr_health()
    return {
        "status": "unknown",
        "error": "Orchestrator not available",
    }