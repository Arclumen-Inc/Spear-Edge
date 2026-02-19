from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from spear_edge.settings import settings

from spear_edge.core.orchestrator.orchestrator import Orchestrator
from spear_edge.core.capture.capture_manager import CaptureManager

from spear_edge.core.sdr.soapy import SoapySDRDevice
from spear_edge.core.sdr.mock import MockSDR

from spear_edge.api.http.routes_health import router as health_router
from spear_edge.api.http.routes_hub import router as hub_router
from spear_edge.api.http.routes_tasking import bind as bind_tasking
from spear_edge.api.http.routes_tripwire import bind as bind_tripwire
from spear_edge.api.http.routes_edge_mode import bind as bind_edge_mode

from spear_edge.api.ws.live_fft_ws import live_fft_ws
from spear_edge.api.ws.events_ws import events_ws

from spear_edge.core import state
from spear_edge.core.gps.gpsd import GpsdClient

from spear_edge.api.ws.tripwire_link_ws import tripwire_link_ws

from spear_edge.api.http.routes_capture import bind as bind_capture

# ------------------------------------------------------------
# SDR factory (auto-detect, safe fallback)
# ------------------------------------------------------------
def make_sdr():
    try:
        return SoapySDRDevice()
    except Exception as e:
        print(f"[SDR] SoapySDR init failed, falling back to MockSDR: {e}")
        return MockSDR()


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
    capture_manager = CaptureManager(orchestrator)

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
    # --------------------------------------------------------
    # WEBSOCKET ROUTES (⚠️ MUST COME BEFORE StaticFiles)
    # --------------------------------------------------------
    @app.websocket("/ws/live_fft")
    async def ws_live_fft(websocket: WebSocket):
        await live_fft_ws(websocket, orchestrator)

    @app.websocket("/ws/notify")
    async def ws_notify(websocket: WebSocket):
        await events_ws(websocket, orchestrator)

    @app.websocket("/ws/tripwire")
    async def ws_tripwire(websocket: WebSocket):
        await tripwire_link_ws(websocket, app.state.orchestrator)
    
    @app.websocket("/ws/tripwire_link")
    async def ws_tripwire_link(websocket: WebSocket):
        await tripwire_link_ws(websocket, app.state.orchestrator)
        
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
        orchestrator._send_atak_status(online=False)
        if hasattr(capture_manager, "stop"):
            await capture_manager.stop()
        await orchestrator.close()

    return app


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
app = create_app()
