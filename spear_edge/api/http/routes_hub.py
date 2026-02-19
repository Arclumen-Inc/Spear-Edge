from fastapi import APIRouter
import time
import json
from pathlib import Path

# This file stores last-seen tripwire nodes
NODES_FILE = Path("data/artifacts/tripwire_nodes.json")

router = APIRouter(prefix="/api/hub", tags=["hub"])


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
