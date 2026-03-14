from fastapi import APIRouter
from spear_edge.core import state

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
def health():
    return {"ok": True}

@router.get("/status")
def status():
    scan_running = False
    mode = "manual"
    try:
        if getattr(state, "engine", None) is not None:
            scan_running = state.engine.status().get("scan_running", False)
            mode = getattr(state.engine, "mode", "manual")
    except Exception:
        pass
    return {
        "ok": True,
        "mode": mode,
        "sdr_open": state.sdr is not None,
        "scan_running": scan_running,
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