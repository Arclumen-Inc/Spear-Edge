from __future__ import annotations
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional


@dataclass
class TripwireNodeState:
    node_id: str
    callsign: str
    ip: str
    sdr: str | None = None
    gps: dict | None = None
    last_seen: float = 0.0
    system_time: float | None = None
    last_events: List[dict] = field(default_factory=list)  # newest first
    connected: bool = False
    transport: str | None = None  # "ws" or "http"
    ws_connected_at: float | None = None
    ws_last_seen: float | None = None

class TripwireRegistry:
    def __init__(self, max_nodes: int = 3):
        self.max_nodes = max_nodes
        self._nodes: Dict[str, TripwireNodeState] = {}

    def update_from_event(self, ev: dict, client_ip: str) -> None:
        now = time()
        node_id = str(ev.get("node_id") or "").strip()
        if not node_id:
            return
    
        # Get or create node
        n = self._nodes.setdefault(node_id, TripwireNodeState(
            node_id=node_id,
            callsign=str(ev.get("callsign") or node_id).strip(),
            ip=client_ip,
            connected=False,
            last_seen=now
        ))
    
        # Always update from event (even if WS connected)
        n.last_seen = now
        n.ip = client_ip  # update IP if changed
        n.last_events.insert(0, ev)  # newest first
        if len(n.last_events) > 10:
            n.last_events.pop()
    
        # Do NOT set connected=True here — only WS does that
        # This prevents HTTP from marking as "connected"
            
    def mark_ws_connected(
        self,
        node_id: str,
        callsign: str,
        ip: str,
        meta: dict | None,
        gps: dict | None,
        system_time: float | None,
    ) -> None:
        now = time()
    
        # Enforce max node limit (do not evict existing nodes)
        if node_id not in self._nodes and len(self._nodes) >= self.max_nodes:
            return
    
        # Get or create node
        node = self._nodes.get(node_id)
        if node is None:
            node = TripwireNodeState(
                node_id=node_id,
                callsign=callsign or node_id,
                ip=ip,
            )
            self._nodes[node_id] = node
    
        # Update identity & timing
        node.callsign = callsign or node.node_id
        node.ip = ip
        node.gps = gps
        node.system_time = system_time
        node.last_seen = now
    
        # Extract SDR info from meta (best effort)
        sdr = None
        if isinstance(meta, dict):
            sdr = meta.get("sdr_driver") or meta.get("sdr")
        node.sdr = sdr
    
        # Mark WS connectivity
        node.connected = True
        node.transport = "ws"
        node.ws_connected_at = node.ws_connected_at or now
        node.ws_last_seen = now
    
    def mark_ws_heartbeat(self, node_id: str):
        n = self._nodes.get(node_id)
        if not n:
            return
        now = time()
        n.connected = True
        n.transport = "ws"
        n.ws_last_seen = now
        n.last_seen = now
    
    
    def mark_ws_disconnected(self, node_id: str):
        n = self._nodes.get(node_id)
        if not n:
            return
        n.connected = False
    
    
    def update_ws_status(self, node_id: str, status: dict):
        # optional extension point
        self.mark_ws_heartbeat(node_id)
        
    def snapshot(self) -> List[dict]:
        # stable order (sorted by node_id)
        out = []
        for node_id in sorted(self._nodes.keys()):
            n = self._nodes[node_id]
            out.append({
                "node_id": n.node_id,
                "callsign": n.callsign,
                "ip": n.ip,
                "sdr": n.sdr,
                "gps": n.gps,
                "system_time": n.system_time,
                "last_seen": n.last_seen,
                "last_events": n.last_events,
                "connected": n.connected,
                "transport": n.transport,
                "ws_connected_at": n.ws_connected_at,
                "ws_last_seen": n.ws_last_seen,
            })
        return out
