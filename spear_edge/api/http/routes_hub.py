from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
import json
from pathlib import Path

# This file stores last-seen tripwire nodes
NODES_FILE = Path("data/artifacts/tripwire_nodes.json")

router = APIRouter(prefix="/api/hub", tags=["hub"])


class CotIdentityBody(BaseModel):
    edge_id: str = Field(..., min_length=1, max_length=64, description="CoT uid + callsign (sanitized server-side)")


@router.get("/nodes")
def get_nodes():
    """
    Read-only view of known tripwire nodes.
    """
    if not NODES_FILE.exists():
        return {"nodes": {}}

    try:
        with NODES_FILE.open("r") as f:
            data = json.load(f)
    except Exception:
        data = {}

    return {"nodes": data}


@router.get("/cot-identity")
def get_cot_identity(request: Request):
    """Current Cursor on Target (CoT) uid and callsign used for multicast."""
    cot = request.app.state.orchestrator.cot
    return {"edge_id": cot.callsign, "uid": cot.uid, "callsign": cot.callsign}


@router.put("/cot-identity")
def put_cot_identity(request: Request, body: CotIdentityBody):
    """Set CoT uid + callsign (same value) and persist to data/artifacts/cot_identity.json."""
    cot = request.app.state.orchestrator.cot
    out = cot.apply_edge_identity(body.edge_id)
    cot.persist_identity()
    return {"ok": True, **out}
