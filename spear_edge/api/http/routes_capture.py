from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import time
import json
from pathlib import Path

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

    @router.post("/label")
    async def label_capture(request: Request):
        """
        Update classification label for a capture.
        Request: {capture_dir: "20260307_043820_915000000Hz_...", label: "elrs"}
        """
        try:
            body = await request.json()
            capture_dir_name = body.get("capture_dir")
            label = body.get("label")
            
            print(f"[CAPTURE ROUTE] Label request: capture_dir={capture_dir_name}, label={label}")
            
            if not capture_dir_name or not label:
                error_msg = "Missing capture_dir or label"
                print(f"[CAPTURE ROUTE] {error_msg}")
                return {"ok": False, "error": error_msg}
            
            # Find capture directory (use absolute path for reliability)
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            captures_dir = project_root / "data" / "artifacts" / "captures"
            capture_dir = captures_dir / capture_dir_name
            
            print(f"[CAPTURE ROUTE] Looking for capture in: {capture_dir}")
            
            if not capture_dir.exists():
                # Try dataset_raw
                dataset_dir = project_root / "data" / "dataset_raw"
                capture_dir = dataset_dir / capture_dir_name
                print(f"[CAPTURE ROUTE] Trying dataset_raw: {capture_dir}")
                if not capture_dir.exists():
                    error_msg = f"Capture directory not found: {capture_dir_name}"
                    print(f"[CAPTURE ROUTE] {error_msg}")
                    return {"ok": False, "error": error_msg}
            
            # Update capture.json
            json_path = capture_dir / "capture.json"
            if not json_path.exists():
                error_msg = f"capture.json not found in {capture_dir}"
                print(f"[CAPTURE ROUTE] {error_msg}")
                return {"ok": False, "error": error_msg}
            
            print(f"[CAPTURE ROUTE] Reading {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Update or create classification
            if "classification" not in data:
                data["classification"] = {}
            
            old_label = data["classification"].get("label", "none")
            data["classification"]["label"] = label
            data["classification"]["confidence"] = 1.0  # Manual label = 100% confidence
            data["classification"]["model"] = "manual_label"
            data["classification"]["labeled_at"] = time.time()
            
            print(f"[CAPTURE ROUTE] Updating label from '{old_label}' to '{label}'")
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[CAPTURE ROUTE] Updated {json_path}")
            
            # Also update dataset_raw copy if it exists
            dataset_raw_dir = project_root / "data" / "dataset_raw" / capture_dir_name
            if dataset_raw_dir.exists():
                dataset_json = dataset_raw_dir / "capture.json"
                if dataset_json.exists():
                    print(f"[CAPTURE ROUTE] Updating dataset_raw copy: {dataset_json}")
                    with open(dataset_json, 'r') as f:
                        dataset_data = json.load(f)
                    dataset_data["classification"] = data["classification"]
                    with open(dataset_json, 'w') as f:
                        json.dump(dataset_data, f, indent=2)
                    print(f"[CAPTURE ROUTE] Updated dataset_raw copy")
            
            print(f"[CAPTURE ROUTE] Successfully labeled {capture_dir_name} as {label}")
            return {"ok": True, "label": label, "capture_dir": capture_dir_name}
        
        except Exception as e:
            error_msg = f"Error labeling capture: {e}"
            print(f"[CAPTURE ROUTE] {error_msg}")
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": str(e)}

    return router