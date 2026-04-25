import asyncio
import logging
import os
import signal
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from spear_edge.settings import settings

# Configure logging level - suppress INFO logs by default, show WARNING and above
# Can be overridden with SPEAR_LOG_LEVEL env var (DEBUG, INFO, WARNING, ERROR)
log_level = os.getenv("SPEAR_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.WARNING),
    format='%(levelname)s: %(message)s'
)

# Suppress uvicorn access logs (HTTP request logs) - they're too verbose
# Only show WARNING and above for uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.setLevel(logging.WARNING)

# Also suppress uvicorn general INFO logs
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.WARNING)

from spear_edge.core.orchestrator.orchestrator import Orchestrator

from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice

from spear_edge.api.http.routes_health import router as health_router
from spear_edge.api.http.routes_hub import router as hub_router
from spear_edge.api.http.routes_tasking import bind as bind_tasking
from spear_edge.api.http.routes_tripwire import bind as bind_tripwire
from spear_edge.api.http.routes_edge_mode import bind as bind_edge_mode
from spear_edge.api.http.routes_network import router as network_router
from spear_edge.api.http.routes_wifi_monitor import bind as bind_wifi_monitor

from spear_edge.api.ws.live_fft_ws import live_fft_ws
from spear_edge.api.ws.events_ws import events_ws

from spear_edge.core import state
from spear_edge.core.gps.gpsd import GpsdClient
from spear_edge.core.wifi_monitor import WifiMonitorConfig, WifiMonitorManager

from spear_edge.api.ws.tripwire_link_ws import tripwire_link_ws

from spear_edge.api.http.routes_capture import bind as bind_capture
from spear_edge.api.http.routes_ml import router as ml_router

# ------------------------------------------------------------
# SDR factory (native libbladerf only)
# ------------------------------------------------------------
def make_sdr():
    """
    Initialize native libbladerf driver.
    Raises exception if bladeRF hardware is not available.
    """
    return BladeRFNativeDevice()


# ------------------------------------------------------------
# App factory
# ------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    # --------------------------------------------------------
    # Middleware
    # --------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------
    # Core runtime objects (SINGLE SOURCE OF TRUTH)
    # --------------------------------------------------------
    sdr = make_sdr()
    orchestrator = Orchestrator(sdr)
    # Use capture_manager from orchestrator to avoid duplicate model loading
    capture_manager = orchestrator.capture_mgr

    app.state.orchestrator = orchestrator
    app.state.capture_manager = capture_manager

    hop_channels = []
    for token in str(settings.WIFI_MONITOR_HOP_CHANNELS).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            hop_channels.append(int(token))
        except ValueError:
            pass
    if not hop_channels:
        hop_channels = [1, 6, 11, 36, 44, 149]
    wifi_cfg = WifiMonitorConfig(
        enabled=False,
        backend=settings.WIFI_MONITOR_BACKEND,
        iface=settings.WIFI_MONITOR_IFACE,
        channel_mode=settings.WIFI_MONITOR_CHANNEL_MODE,
        hop_channels=hop_channels,
        poll_interval_s=settings.WIFI_MONITOR_POLL_INTERVAL_S,
        kismet_cmd=settings.WIFI_MONITOR_KISMET_CMD,
        kismet_url=settings.WIFI_MONITOR_KISMET_URL,
        kismet_username=settings.WIFI_MONITOR_KISMET_USERNAME,
        kismet_password=settings.WIFI_MONITOR_KISMET_PASSWORD,
        kismet_timeout_s=settings.WIFI_MONITOR_KISMET_TIMEOUT_S,
    )
    app.state.wifi_monitor = WifiMonitorManager(orchestrator.bus, wifi_cfg, cot=orchestrator.cot)

    # --------------------------------------------------------
    # Global shared state (legacy-safe)
    # --------------------------------------------------------
    state.engine = orchestrator
    state.sdr = sdr
    state.wifi_monitor = app.state.wifi_monitor

    state.gps = GpsdClient(
        poll_interval_s=settings.GPS_POLL_INTERVAL_S,
        host=settings.GPSD_HOST,
        port=settings.GPSD_PORT,
        on_fix=lambda fix: orchestrator.cot.update_gps(fix),
    )
    state.gps.start()

    # --------------------------------------------------------
    # HTTP API ROUTES
    # --------------------------------------------------------
    app.include_router(health_router)
    app.include_router(hub_router)
    app.include_router(bind_tasking(orchestrator))
    app.include_router(bind_tripwire())
    app.include_router(bind_edge_mode())
    app.include_router(bind_capture(orchestrator))
    app.include_router(ml_router)
    app.include_router(network_router)
    app.include_router(bind_wifi_monitor())
    # --------------------------------------------------------
    # WEBSOCKET ROUTES (⚠️ MUST COME BEFORE StaticFiles)
    # --------------------------------------------------------
    @app.websocket("/ws/live_fft")
    async def ws_live_fft(websocket: WebSocket):
        await live_fft_ws(websocket, orchestrator)

    @app.websocket("/ws/notify")
    async def ws_notify(websocket: WebSocket):
        await events_ws(websocket, orchestrator)

    @app.websocket("/ws/tripwire_link")
    async def ws_tripwire_link(websocket: WebSocket):
        await tripwire_link_ws(websocket, app.state.orchestrator)
    
    @app.websocket("/ws/tripwire")
    async def ws_tripwire(websocket: WebSocket):
        """Alias for /ws/tripwire_link - matches Tripwire v2.0 integration doc."""
        await tripwire_link_ws(websocket, app.state.orchestrator)
    
    # --------------------------------------------------------
    # FAVICON HANDLER (prevent 404 warnings)
    # --------------------------------------------------------
    @app.get("/favicon.ico")
    async def favicon():
        from fastapi.responses import Response
        # Return empty response to suppress browser requests
        return Response(content=b"", media_type="image/x-icon")
    
    # --------------------------------------------------------
    # ML DASHBOARD ROUTE
    # --------------------------------------------------------
    @app.get("/ml")
    async def ml_dashboard():
        from fastapi.responses import FileResponse
        from pathlib import Path
        
        # Use same path as StaticFiles mount (relative to project root)
        ml_html_path = Path("spear_edge/ui/web/ml.html")
        
        if ml_html_path.exists():
            return FileResponse(str(ml_html_path.resolve()), media_type="text/html")
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"ML dashboard not found at {ml_html_path.resolve()}")

    @app.get("/wifi")
    async def wifi_dashboard():
        from fastapi.responses import FileResponse
        from pathlib import Path

        wifi_html_path = Path("spear_edge/ui/web/wifi.html")
        if wifi_html_path.exists():
            return FileResponse(str(wifi_html_path.resolve()), media_type="text/html")
        else:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Wi-Fi dashboard not found at {wifi_html_path.resolve()}")
        
    # --------------------------------------------------------
    # STATIC UI (⚠️ MUST BE LAST)
    # --------------------------------------------------------
    app.mount(
        "/",
        StaticFiles(directory="spear_edge/ui/web", html=True),
        name="ui",
    )

    # --------------------------------------------------------
    # Lifecycle hooks
    # --------------------------------------------------------
    @app.on_event("startup")
    async def _startup():
        if hasattr(capture_manager, "start"):
            await capture_manager.start()
        # Send online status if already in ARMED mode
        if orchestrator.mode == "armed":
            count = orchestrator._count_connected_tripwires()
            orchestrator._send_atak_status(online=True, tripwire_count=count)
        if settings.WIFI_MONITOR_AUTOSTART:
            try:
                await app.state.wifi_monitor.start()
            except Exception as e:
                print(f"[WIFI MONITOR] Autostart failed: {e}")
        
        # Register signal handler to stop scan immediately on SIGINT/SIGTERM
        # This allows graceful shutdown on first Ctrl+C instead of waiting
        _shutting_down = False
        
        def _signal_handler(signum, frame):
            nonlocal _shutting_down
            print(f"\n[SIGNAL] Received signal {signum}, stopping scan...")
            
            # Stop scan synchronously by setting flags
            if orchestrator._scan:
                orchestrator._scan._running = False
            if orchestrator._rx_task:
                orchestrator._rx_task._running = False
                orchestrator._rx_task._stop_event.set()
            # Deactivate SDR stream to unblock reads
            if orchestrator.sdr and hasattr(orchestrator.sdr, '_deactivate_stream'):
                try:
                    orchestrator.sdr._deactivate_stream()
                except Exception:
                    pass
            
            if _shutting_down:
                # Second signal - force exit
                print("[SIGNAL] Force quit...")
                os._exit(1)
            
            _shutting_down = True
            # Restore default handler and re-raise signal to let uvicorn handle it
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        
        # Install handler
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    @app.on_event("shutdown")
    async def _shutdown():
        print("[SHUTDOWN] Starting graceful shutdown...")
        shutdown_timeout_s = 10.0
        
        # Send offline status to ATAK
        try:
            orchestrator._send_atak_status(online=False)
        except Exception:
            pass
        
        # Stop capture manager first (it might be holding SDR resources)
        try:
            print("[SHUTDOWN] Stopping Wi-Fi monitor...")
            await app.state.wifi_monitor.stop()
        except Exception as e:
            print(f"[SHUTDOWN] WARNING: wifi monitor stop error: {e}")

        # Stop capture manager first (it might be holding SDR resources)
        try:
            if hasattr(capture_manager, "stop"):
                print("[SHUTDOWN] Stopping capture manager...")
                await asyncio.wait_for(capture_manager.stop(), timeout=shutdown_timeout_s)
                print("[SHUTDOWN] Capture manager stopped")
        except asyncio.TimeoutError:
            print("[SHUTDOWN] WARNING: capture_manager.stop() timed out")
        except Exception as e:
            print(f"[SHUTDOWN] WARNING: capture_manager error: {e}")
        
        # Stop GPS client
        try:
            if state.gps:
                print("[SHUTDOWN] Stopping GPS client...")
                state.gps.stop()
                print("[SHUTDOWN] GPS client stopped")
        except Exception as e:
            print(f"[SHUTDOWN] WARNING: GPS stop error: {e}")
        
        # Close orchestrator (SDR, scan tasks, rx task)
        try:
            print("[SHUTDOWN] Closing orchestrator...")
            await asyncio.wait_for(orchestrator.close(), timeout=shutdown_timeout_s)
            print("[SHUTDOWN] Orchestrator closed")
        except asyncio.TimeoutError:
            print("[SHUTDOWN] WARNING: orchestrator.close() timed out - forcing SDR close")
            try:
                if state.sdr:
                    state.sdr.close()
            except Exception:
                pass
        except Exception as e:
            print(f"[SHUTDOWN] WARNING: orchestrator error: {e}")
        
        print("[SHUTDOWN] Graceful shutdown complete")

    return app


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
app = create_app()
