from __future__ import annotations

import csv
import json
import io
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from spear_edge.settings import settings


class WifiMonitorConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    backend: Optional[str] = None
    iface: Optional[str] = None
    channel_mode: Optional[str] = None
    hop_channels: Optional[List[int]] = None
    poll_interval_s: Optional[float] = None
    kismet_cmd: Optional[str] = None
    kismet_url: Optional[str] = None
    kismet_username: Optional[str] = None
    kismet_password: Optional[str] = None
    kismet_timeout_s: Optional[float] = None


class AddDatasourceRequest(BaseModel):
    source: str


class SetChannelRequest(BaseModel):
    uuid: str
    channel: str


class SetHopRequest(BaseModel):
    uuid: str
    channels: List[str]
    rate: float = 5.0


class SourceActionRequest(BaseModel):
    uuid: str


class PresenceAlertRequest(BaseModel):
    alert_type: str  # devicefound|devicelost
    action: str      # add|remove
    mac: str


def bind():
    router = APIRouter(prefix="/api/wifi-monitor", tags=["wifi-monitor"])

    async def _manager_call(method: str, path: str) -> Dict[str, Any]:
        base = (settings.WIFI_MANAGER_URL or "").rstrip("/")
        if not base:
            return {"ok": False, "error": "manager_url_not_configured"}
        url = f"{base}{path}"
        headers = {"Accept": "application/json"}
        if settings.WIFI_MANAGER_TOKEN:
            headers["Authorization"] = f"Bearer {settings.WIFI_MANAGER_TOKEN}"
        req = urllib.request.Request(url=url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                txt = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(txt) if txt else {}
                return {"ok": True, "status_code": resp.status, "data": data}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @router.get("/status")
    async def wifi_monitor_status(request: Request):
        mgr = request.app.state.wifi_monitor
        return {"ok": True, "status": mgr.get_status()}

    @router.get("/live")
    async def wifi_monitor_live(request: Request, limit: int = 100):
        mgr = request.app.state.wifi_monitor
        return {"ok": True, **mgr.get_recent_events(limit=limit)}

    @router.get("/export")
    async def wifi_monitor_export(request: Request, kind: str = "rid", format: str = "jsonl", limit: int = 1000):
        mgr = request.app.state.wifi_monitor
        events = mgr.get_recent_events(limit=limit)
        key = "rid_updates" if str(kind).lower() != "wifi" else "wifi_intel_updates"
        rows = list(events.get(key, []))

        fmt = str(format).lower()
        if fmt == "csv":
            buff = io.StringIO()
            writer = csv.writer(buff)
            writer.writerow(["ts", "source", "lat", "lon", "type", "summary"])
            for row in rows:
                ts = row.get("ts")
                src = row.get("source")
                obs = dict(row.get("observer_position", {}) or {})
                lat = obs.get("lat")
                lon = obs.get("lon")
                item_type = "rid_update" if key == "rid_updates" else "wifi_intel_update"
                summary = ""
                if item_type == "rid_update":
                    p = dict(row.get("protocol_result", {}) or {})
                    f = dict(p.get("decoded_fields", {}) or {})
                    summary = str(f.get("uas_id") or p.get("protocol") or "rid")
                else:
                    summary = str(row.get("status") or "wifi")
                writer.writerow([ts, src, lat, lon, item_type, summary])
            return PlainTextResponse(content=buff.getvalue(), media_type="text/csv")

        body = "\n".join(json.dumps(r, separators=(",", ":"), ensure_ascii=True) for r in rows)
        if body:
            body += "\n"
        return PlainTextResponse(content=body, media_type="application/x-ndjson")

    @router.post("/start")
    async def wifi_monitor_start(request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.start()

    @router.post("/stop")
    async def wifi_monitor_stop(request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.stop()

    @router.post("/config")
    async def wifi_monitor_config(update: WifiMonitorConfigUpdate, request: Request):
        mgr = request.app.state.wifi_monitor
        data: Dict[str, Any] = update.model_dump(exclude_none=True)
        status = mgr.update_config(data)
        return {"ok": True, "status": status}

    @router.post("/test-kismet")
    async def wifi_monitor_test_kismet(request: Request):
        mgr = request.app.state.wifi_monitor
        result = await mgr.test_connection()
        return {"ok": True, "result": result}

    @router.get("/datasources")
    async def wifi_monitor_datasources(request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.list_datasources()

    @router.get("/interfaces")
    async def wifi_monitor_interfaces(request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.list_interfaces()

    @router.post("/datasource/add")
    async def wifi_monitor_add_datasource(payload: AddDatasourceRequest, request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.add_datasource(payload.source)

    @router.post("/datasource/set-channel")
    async def wifi_monitor_set_channel(payload: SetChannelRequest, request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.set_channel(payload.uuid, payload.channel)

    @router.post("/datasource/set-hop")
    async def wifi_monitor_set_hop(payload: SetHopRequest, request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.set_hop(payload.uuid, payload.channels, payload.rate)

    @router.post("/datasource/open")
    async def wifi_monitor_open_source(payload: SourceActionRequest, request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.open_source(payload.uuid)

    @router.post("/datasource/close")
    async def wifi_monitor_close_source(payload: SourceActionRequest, request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.close_source(payload.uuid)

    @router.get("/alerts")
    async def wifi_monitor_alerts(request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.list_alerts()

    @router.post("/alerts/presence")
    async def wifi_monitor_presence_alert(payload: PresenceAlertRequest, request: Request):
        mgr = request.app.state.wifi_monitor
        return await mgr.set_presence_alert(payload.alert_type, payload.action, payload.mac)

    @router.get("/manager/kismet/status")
    async def wifi_manager_kismet_status():
        return await _manager_call("GET", "/kismet/status")

    @router.post("/manager/kismet/start")
    async def wifi_manager_kismet_start():
        return await _manager_call("POST", "/kismet/start")

    @router.post("/manager/kismet/stop")
    async def wifi_manager_kismet_stop():
        return await _manager_call("POST", "/kismet/stop")

    return router
