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
    sdr_open = False
    try:
        if getattr(state, "engine", None) is not None:
            scan_running = state.engine.status().get("scan_running", False)
            mode = getattr(state.engine, "mode", "manual")
        if getattr(state, "sdr", None) is not None:
            sdr_open = bool(getattr(state.sdr, "connected", True))
    except Exception:
        pass
    wifi_monitor = None
    try:
        if getattr(state, "wifi_monitor", None) is not None:
            wifi_monitor = state.wifi_monitor.get_status()
    except Exception:
        wifi_monitor = None
    return {
        "ok": True,
        "mode": mode,
        "sdr_open": sdr_open,
        "scan_running": scan_running,
        "gps": state.gps.get_status() if state.gps else None,
        "wifi_monitor": wifi_monitor,
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