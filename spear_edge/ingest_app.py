from __future__ import annotations

import httpx
import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# Storage
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "artifacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NODES_FILE = DATA_DIR / "tripwire_nodes.json"


def load_nodes() -> Dict[str, Any]:
    if NODES_FILE.exists():
        try:
            return json.loads(NODES_FILE.read_text())
        except Exception:
            pass
    return {"nodes": {}}


def save_nodes(data: Dict[str, Any]) -> None:
    NODES_FILE.write_text(json.dumps(data, indent=2))


# -------------------------------------------------
# FastAPI app (INGEST)
# -------------------------------------------------

app = FastAPI(title="SPEAR-Edge Ingest")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# Tripwire ingest endpoint (AUTHORITATIVE)
# -------------------------------------------------

@app.post("/api/tripwire/event")
async def ingest_tripwire_event(req: Request):
    payload = await req.json()
    now = time.time()

    node_id = payload.get("node_id", "tripwire-unknown")

    data = load_nodes()
    nodes = data.setdefault("nodes", {})

    node = nodes.setdefault(node_id, {})
    node["node_id"] = node_id
    node["connected"] = True
    node["last_seen"] = now

    # Derive REAL remote IP (handles local forward from Tripwire → ingest → Edge)
    client_host = "unknown"
    if req.client:
        client_host = req.client.host
    
    # If request comes from localhost (forward from actual Tripwire), use X-Forwarded-For if present
    if client_host in ("127.0.0.1", "::1", "localhost"):
        forwarded = req.headers.get("x-forwarded-for")
        if forwarded:
            # Take the first (original client) IP
            client_host = forwarded.split(",")[0].strip()
    
    node["remote_ip"] = client_host

    # Store last event (authoritative payload slice)
    node["last_event"] = {
        "classification": payload.get("classification"),
        "freq_hz": payload.get("freq_hz"),
        "delta_db": payload.get("delta_db"),
        "remarks": payload.get("remarks"),
        "scan_plan": payload.get("scan_plan"),
        "timestamp": payload.get("timestamp", now),
    }

    save_nodes(data)

    # -------------------------------------------------
    # Forward cue to Edge UI app (best effort)
    # -------------------------------------------------
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            await client.post(
                "http://127.0.0.1:8080/api/tripwire/cue",
                json=payload,
                headers={"X-Forwarded-For": node["remote_ip"]},  # ← ADD THIS
            )
        except Exception:
            pass

    return {"status": "ok"}
