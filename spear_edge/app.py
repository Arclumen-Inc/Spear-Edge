import asyncio
import logging
import os
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

from spear_edge.api.ws.live_fft_ws import live_fft_ws
from spear_edge.api.ws.events_ws import events_ws

from spear_edge.core import state
from spear_edge.core.gps.gpsd import GpsdClient

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

    # --------------------------------------------------------
    # Global shared state (legacy-safe)
    # --------------------------------------------------------
    state.engine = orchestrator
    state.sdr = sdr

    state.gps = GpsdClient()
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

    @app.on_event("shutdown")
    async def _shutdown():
        shutdown_timeout_s = 15.0
        try:
            orchestrator._send_atak_status(online=False)
        except Exception:
            pass
        try:
            if hasattr(capture_manager, "stop"):
                await asyncio.wait_for(capture_manager.stop(), timeout=shutdown_timeout_s)
        except asyncio.TimeoutError:
            logging.warning("[SHUTDOWN] capture_manager.stop() timed out")
        except Exception:
            pass
        try:
            await asyncio.wait_for(orchestrator.close(), timeout=shutdown_timeout_s)
        except asyncio.TimeoutError:
            logging.warning("[SHUTDOWN] orchestrator.close() timed out")
        except Exception:
            pass

    return app


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
app = create_app()
