from fastapi import APIRouter, Request

def bind():
    router = APIRouter(prefix="/api/edge", tags=["edge"])

    @router.get("/mode")
    async def get_mode(request: Request):
        orch = request.app.state.orchestrator
        return {"mode": orch.mode}

    @router.post("/mode/{mode}")
    async def set_edge_mode(mode: str, request: Request):
        """
        Set Edge mode (manual or armed).
        Uses orchestrator.set_mode() to ensure:
        - ATAK status notifications
        - Tripwire WebSocket broadcasts
        - Event bus updates
        """
        orch = request.app.state.orchestrator

        mode = (mode or "").strip().lower()
        if mode not in ("manual", "armed"):
            return {"ok": False, "error": "mode must be manual|armed"}

        try:
            orch.set_mode(mode)
            return {"ok": True, "mode": orch.mode}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    return router
