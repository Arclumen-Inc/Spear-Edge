from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import time

from spear_edge.core.bus.models import CaptureRequest

class ManualCaptureRequest(BaseModel):
    reason: str = "manual"
    freq_hz: float
    sample_rate_sps: Optional[int] = None
    bandwidth_hz: Optional[int] = None
    gain_mode: Optional[str] = None
    gain_db: Optional[float] = None
    rx_channel: int = 0
    duration_s: float = 5.0
    source_node: Optional[str] = None
    scan_plan: Optional[str] = None
    classification: Optional[str] = None

def bind(orchestrator):
    router = APIRouter(prefix="/api/capture", tags=["capture"])

    @router.post("/start")
    async def manual_capture(req: ManualCaptureRequest, request: Request):
        orch = request.app.state.orchestrator

        print("[CAPTURE ROUTE] Manual capture requested:", req.dict())

        try:
            # Use provided sample_rate_sps or fallback to current SDR scan sample rate if available
            sdr_cfg = getattr(orch, "sdr_config", None)
            default_sr = int(sdr_cfg.sample_rate_sps) if (sdr_cfg and sdr_cfg.sample_rate_sps) else 10_000_000
            sample_rate = int(req.sample_rate_sps) if req.sample_rate_sps else default_sr
            
            # Create CaptureRequest with all required fields
            # Include ALL SDR settings from manual capture in meta field
            # Always build meta unconditionally so downstream logic is stable
            meta = {
                "bandwidth_hz": req.bandwidth_hz,
                "gain_mode": req.gain_mode,
                "gain_db": req.gain_db,
                "classification": req.classification,
            }
            
            capture_req = CaptureRequest(
                ts=time.time(),
                reason=req.reason,
                freq_hz=req.freq_hz,
                sample_rate_sps=sample_rate,
                duration_s=req.duration_s,
                rx_channel=req.rx_channel,
                scan_plan=req.scan_plan,
                priority=50,
                source_node=req.source_node,
                meta=meta,
            )

            ok = orch.capture_mgr.submit_nowait(capture_req)

            print(f"[CAPTURE ROUTE] Queued successfully: {ok}")

            return {
                "accepted": ok,
                "action": "capture_started" if ok else "queue_full"
            }

        except Exception as e:
            print("[CAPTURE ROUTE] ERROR creating or queuing capture:", e)
            import traceback
            traceback.print_exc()
            raise  # re-raise to return 500 with details in logs

    return router