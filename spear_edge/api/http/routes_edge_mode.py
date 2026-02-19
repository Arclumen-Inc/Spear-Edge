from fastapi import APIRouter, Request

def bind():
    router = APIRouter(prefix="/api/edge", tags=["edge"])

    @router.get("/mode")
    async def get_mode(request: Request):
        orch = request.app.state.orchestrator
        return {"mode": orch.mode}

    @router.post("/mode/{mode}")
    async def set_mode(mode: str, request: Request):
        orch = request.app.state.orchestrator

        mode = (mode or "").strip().lower()
        if mode not in ("manual", "armed"):
            return {"ok": False, "error": "mode must be manual|armed"}

        orch.mode = mode
        orch.bus.publish_nowait("edge_mode", {"mode": mode})
        return {"ok": True, "mode": mode}

    return router
