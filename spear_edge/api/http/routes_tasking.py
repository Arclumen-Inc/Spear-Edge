from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from spear_edge.core.sdr.base import GainMode, SdrConfig

print("[ROUTES_TASKING] loaded")
# -------------------------------------------------
# Request models
# -------------------------------------------------

class StartScanRequest(BaseModel):
    center_freq_hz: int
    sample_rate_sps: int
    fft_size: int = 2048
    fps: float = 15.0


class SetModeRequest(BaseModel):
    mode: str  # "manual" | "armed"


class AutoPolicyRequest(BaseModel):
    enabled: bool | None = None
    min_confidence: float | None = None
    global_cooldown_s: float | None = None
    per_node_cooldown_s: float | None = None
    per_freq_cooldown_s: float | None = None
    freq_bin_hz: int | None = None
    max_captures_per_min: int | None = None


class SdrConfigRequest(BaseModel):
    center_freq_hz: int
    sample_rate_sps: int
    gain_mode: GainMode = GainMode.MANUAL
    gain_db: float = 0.0  # Default 0 dB - user can adjust via UI slider
    rx_channel: int = 0
    bandwidth_hz: int | None = None
    # LNA gain is now automatically optimized by bladerf_set_gain() - no manual control
    bt200_enabled: bool | None = None  # BT200 external LNA enabled
    dual_channel: bool = False  # Dual RX mode


# -------------------------------------------------
# Router binding
# -------------------------------------------------

def bind(orchestrator) -> APIRouter:
    router = APIRouter(prefix="/live", tags=["live"])
    
    @router.post("/start")
    async def start(req: StartScanRequest):
        print(f"[LIVE] /start ENTERED with req: {req}")
    
        # Set SDR config first (so start_scan can use it)
        # This avoids redundant configuration in start_scan
        cfg = SdrConfig(
            center_freq_hz=req.center_freq_hz,
            sample_rate_sps=req.sample_rate_sps,
            gain_mode=GainMode.MANUAL,
                gain_db=0.0,  # Default 0 dB - user can adjust via UI slider
            rx_channel=0,
            bandwidth_hz=None,
        )
        orchestrator.sdr_config = cfg
    
        # Let start_scan handle everything (it will stop existing scan, configure SDR, start tasks)
        # This is more efficient than doing it twice
        print(f"[LIVE] Starting scan: freq={req.center_freq_hz/1e6:.3f} MHz, rate={req.sample_rate_sps/1e6:.2f} MS/s, fft={req.fft_size}, fps={req.fps}")
        await orchestrator.start_scan(
            center_freq_hz=req.center_freq_hz,
            sample_rate_sps=req.sample_rate_sps,
            fft_size=req.fft_size,
            fps=req.fps,
        )
        print("[LIVE] Scan started successfully")
    
        return {"ok": True}

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _get_scan_running() -> bool:
        """
        Best-effort scan running flag.
        Your orchestrator.status() already exists; we'll rely on it.
        """
        try:
            st = orchestrator.status()
            return bool(st.get("scan_running"))
        except Exception:
            return False

    def _safe_apply_gain_only(cfg: SdrConfig) -> Optional[str]:
        """
        Apply ONLY gain settings live. No tune, no sample rate changes.
        Returns error string if failed, else None.
        """
        try:
            sdr = orchestrator.sdr
            if hasattr(sdr, "set_rx_channel"):
                sdr.set_rx_channel(cfg.rx_channel)

            if hasattr(sdr, "set_gain_mode"):
                sdr.set_gain_mode(cfg.gain_mode)

            if cfg.gain_mode == GainMode.MANUAL and hasattr(sdr, "set_gain"):
                sdr.set_gain(cfg.gain_db)

            return None
        except Exception as e:
            return str(e)

    def _build_cfg(req: SdrConfigRequest) -> SdrConfig:
        return SdrConfig(
            center_freq_hz=req.center_freq_hz,
            sample_rate_sps=req.sample_rate_sps,
            gain_mode=req.gain_mode,
            gain_db=req.gain_db,
            rx_channel=req.rx_channel,
            bandwidth_hz=req.bandwidth_hz,
            # LNA gain is now automatically optimized by bladerf_set_gain() - no manual control needed
            bt200_enabled=getattr(req, 'bt200_enabled', None),
            dual_channel=getattr(req, 'dual_channel', False),
        )

    # -------------------------------------------------
    # Live scan control
    # -------------------------------------------------

    @router.post("/live/start")
    async def live_start(req: StartScanRequest):
        print("[ROUTES_TASKING] live_start ENTERED")    
        """
        Start (or restart) live FFT scanning.

        IMPORTANT:
        - Do NOT manually tune/configure the SDR here.
        - Orchestrator is the single owner of SDR + RX + scan lifecycle.
        """
        print("[API] live_start called", req)

        # Orchestrator is responsible for doing the right thing:
        # - stop existing scan if needed
        # - reconfigure SDR once
        # - start RX + FFT pipeline once
        await orchestrator.start_scan(
            center_freq_hz=req.center_freq_hz,
            sample_rate_sps=req.sample_rate_sps,
            fft_size=req.fft_size,
            fps=req.fps,
        )

        return {"ok": True, "status": orchestrator.status()}

    @router.post("/stop")
    async def live_stop():
        """
        Stop live FFT scanning cleanly (prevents bladeRF buffer timeouts
        and keeps page refresh safe).
        """
        print("[API] /live/stop called")
        await orchestrator.stop_scan()
        return {"ok": True, "status": orchestrator.status()}

    # -------------------------------------------------
    # Mode control
    # -------------------------------------------------

    @router.post("/mode/set")
    def set_mode(req: SetModeRequest):
        if req.mode not in ("manual", "armed"):
            return {
                "ok": False,
                "error": "mode_not_operator_settable",
                "allowed": ["manual", "armed"],
            }

        orchestrator.mode = req.mode
        return {"ok": True, "mode": orchestrator.mode}

    @router.get("/mode")
    def get_mode():
        return {"mode": orchestrator.mode}

    # -------------------------------------------------
    # Auto-capture policy
    # -------------------------------------------------

    @router.get("/auto/policy")
    def get_auto_policy():
        return asdict(orchestrator.auto_policy)

    @router.post("/auto/policy")
    def set_auto_policy(req: AutoPolicyRequest):
        p = orchestrator.auto_policy
        for k, v in req.model_dump(exclude_none=True).items():
            setattr(p, k, v)
        return {"ok": True, "auto_policy": asdict(p)}

    # -------------------------------------------------
    # Captures (READ ONLY)
    # -------------------------------------------------

    @router.get("/api/captures")
    def list_captures(limit: int = 50):
        # Keep your debug line (safe)
        print("[CAPTURE_LOG_READ]", id(orchestrator), 0)
        return {"captures": orchestrator.list_captures(limit=limit)}

    # -------------------------------------------------
    # SDR config (safe: avoids double-tune + avoids mid-stream retune)
    # -------------------------------------------------

    @router.post("/sdr/config")
    async def set_sdr_config(req: SdrConfigRequest):
        """
        Apply SDR config safely.

        Rules:
        - If TASKED: reject (UI should be locked anyway)
        - If scan is running:
            - If ONLY gain changed: apply gain live (no restart)
            - Else: restart scan via orchestrator (single owner)
        - If scan not running: apply config once via sdr.apply_config()
        """
        if getattr(orchestrator, "mode", None) == "tasked":
            return {"ok": False, "error": "SDR locked in TASKED mode"}

        new_cfg = _build_cfg(req)
        prev_cfg = getattr(orchestrator, "sdr_config", None)
        scan_running = _get_scan_running()

        # Determine whether ONLY gain changed
        only_gain_changed = False
        if prev_cfg is not None:
            try:
                only_gain_changed = (
                    new_cfg.center_freq_hz == prev_cfg.center_freq_hz
                    and new_cfg.sample_rate_sps == prev_cfg.sample_rate_sps
                    and new_cfg.rx_channel == prev_cfg.rx_channel
                    and new_cfg.bandwidth_hz == prev_cfg.bandwidth_hz
                    and new_cfg.gain_mode == prev_cfg.gain_mode
                    and float(new_cfg.gain_db) != float(prev_cfg.gain_db)
                )
            except Exception:
                only_gain_changed = False

        # Store config so /sdr/info reflects "what we want"
        orchestrator.sdr_config = new_cfg

        # If scanning and only gain changed -> apply gain live
        if scan_running and only_gain_changed:
            err = _safe_apply_gain_only(new_cfg)
            if err:
                return {"ok": False, "error": "gain_update_failed", "detail": err}
            return {"ok": True, "config": new_cfg.__dict__, "note": "gain updated live"}

        # If scanning and tuning/rate changed -> restart scan cleanly
        if scan_running and not only_gain_changed:
            # Keep current FFT params if possible
            st = orchestrator.status()
            # Use current FFT size/FPS from status, or fallback to defaults
            # Default to 4096 (new default) instead of 1024 (old USB overflow workaround)
            fft_size = int(st.get("fft_size") or 4096)
            fps = float(st.get("fps") or 15.0)

            await orchestrator.start_scan(
                center_freq_hz=new_cfg.center_freq_hz,
                sample_rate_sps=new_cfg.sample_rate_sps,
                fft_size=fft_size,
                fps=fps,
            )
            return {"ok": True, "config": new_cfg.__dict__, "note": "scan restarted with new tuning"}

        # Not scanning -> apply config once, no extra tune calls
        try:
            orchestrator.sdr.apply_config(new_cfg)
        except Exception as e:
            return {"ok": False, "error": "apply_config_failed", "detail": str(e)}

        return {"ok": True, "config": new_cfg.__dict__}

    # -------------------------------------------------
    # SDR info (READ ONLY) — MUST NEVER THROW
    # -------------------------------------------------

    @router.get("/sdr/info")
    def get_sdr_info():
        try:
            sdr = getattr(orchestrator, "sdr", None)
            if not sdr:
                return {"connected": False}

            info: Dict[str, Any] = {
                "connected": True,
                "driver": getattr(sdr, "driver", None),
                "rx_channels": getattr(sdr, "max_rx_channels", 1),
                "supports_agc": getattr(sdr, "supports_agc", False),
                "current_config": getattr(orchestrator, "sdr_config", None).__dict__
                if getattr(orchestrator, "sdr_config", None)
                else None,
            }

            if hasattr(sdr, "get_info"):
                try:
                    info["device"] = sdr.get_info()
                except Exception:
                    info["device"] = None
            else:
                info["device"] = None

            return info

        except Exception as e:
            return {"connected": False, "error": "sdr_info_failed", "detail": str(e)}
    
    @router.get("/captures")
    def list_captures(limit: int = 50):
        """
        Read-only capture log for UI.
        """
        try:
            return {
                "captures": orchestrator.list_captures(limit=limit)
            }
        except Exception as e:
            return {
                "captures": [],
                "error": str(e),
            }
            
    return router
