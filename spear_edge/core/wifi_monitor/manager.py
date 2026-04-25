from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from spear_edge.core.bus.event_bus import EventBus

from .models import WifiMonitorConfig, WifiMonitorStatus
from .provider_base import WifiProviderBase
from .provider_generic import GenericProvider
from .provider_kismet import KismetProvider


class WifiMonitorManager:
    def __init__(self, bus: EventBus, config: WifiMonitorConfig, cot: Any | None = None):
        self.bus = bus
        self.config = config
        self.cot = cot
        self.status = WifiMonitorStatus(
            enabled=config.enabled,
            running=False,
            backend=config.backend,
            iface=config.iface,
            channel_mode=config.channel_mode,
        )
        self._task: Optional[asyncio.Task] = None
        self._stop_evt = asyncio.Event()
        self._provider = self._make_provider(config.backend)
        self._cot_last_sent: Dict[str, float] = {}
        self._recent_rid = deque(maxlen=500)
        self._recent_wifi = deque(maxlen=500)
        self._store_dir = Path("/home/spear/spear-edgev1_0/data/artifacts/wifi")
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._rid_log_path = self._store_dir / "rid_updates.jsonl"
        self._wifi_log_path = self._store_dir / "wifi_intel_updates.jsonl"

    def _make_provider(self, backend: str) -> WifiProviderBase:
        b = (backend or "").strip().lower()
        if b == "generic":
            return GenericProvider()
        return KismetProvider()

    async def start(self) -> Dict[str, Any]:
        if self._task and not self._task.done():
            return {"ok": True, "status": self.status.as_dict(), "message": "already_running"}
        self._stop_evt.clear()
        self.status.enabled = True
        self.status.running = True
        self.status.backend = self.config.backend
        self.status.iface = self.config.iface
        self.status.channel_mode = self.config.channel_mode
        self._provider = self._make_provider(self.config.backend)
        await self._provider.start(self.config)
        self._task = asyncio.create_task(self._run_loop(), name="wifi-monitor-loop")
        self.bus.publish_nowait("wifi_intel_update", {"event": "wifi_monitor_started", "ts": time.time()})
        return {"ok": True, "status": self.status.as_dict()}

    async def stop(self) -> Dict[str, Any]:
        self.status.enabled = False
        self.status.running = False
        self._stop_evt.set()
        await self._provider.stop()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self.bus.publish_nowait("wifi_intel_update", {"event": "wifi_monitor_stopped", "ts": time.time()})
        return {"ok": True, "status": self.status.as_dict()}

    async def test_connection(self) -> Dict[str, Any]:
        provider = self._make_provider(self.config.backend)
        try:
            return await provider.test_connection(self.config)
        except Exception as e:
            return {"ok": False, "backend": self.config.backend, "error": str(e)}

    async def _provider_call(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        provider = self._make_provider(self.config.backend)
        fn = getattr(provider, method)
        try:
            return await fn(self.config, *args, **kwargs)
        except Exception as e:
            return {"ok": False, "backend": self.config.backend, "error": str(e)}

    async def list_datasources(self) -> Dict[str, Any]:
        return await self._provider_call("list_datasources")

    async def list_interfaces(self) -> Dict[str, Any]:
        return await self._provider_call("list_interfaces")

    async def add_datasource(self, source: str) -> Dict[str, Any]:
        return await self._provider_call("add_datasource", source)

    async def set_channel(self, uuid: str, channel: str) -> Dict[str, Any]:
        return await self._provider_call("set_channel", uuid, channel)

    async def set_hop(self, uuid: str, channels: list[str], rate: float) -> Dict[str, Any]:
        return await self._provider_call("set_hop", uuid, channels, rate)

    async def open_source(self, uuid: str) -> Dict[str, Any]:
        return await self._provider_call("open_source", uuid)

    async def close_source(self, uuid: str) -> Dict[str, Any]:
        return await self._provider_call("close_source", uuid)

    async def list_alerts(self) -> Dict[str, Any]:
        return await self._provider_call("list_alerts")

    async def set_presence_alert(self, alert_type: str, action: str, mac: str) -> Dict[str, Any]:
        return await self._provider_call("set_presence_alert", alert_type, action, mac)

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        if "enabled" in updates:
            self.config.enabled = bool(updates["enabled"])
        if "backend" in updates and updates["backend"]:
            self.config.backend = str(updates["backend"]).strip().lower()
        if "iface" in updates and updates["iface"]:
            self.config.iface = str(updates["iface"]).strip()
        if "channel_mode" in updates and updates["channel_mode"]:
            self.config.channel_mode = str(updates["channel_mode"]).strip().lower()
        if "hop_channels" in updates and isinstance(updates["hop_channels"], list):
            cleaned = [int(x) for x in updates["hop_channels"] if str(x).strip()]
            if cleaned:
                self.config.hop_channels = cleaned
        if "poll_interval_s" in updates:
            try:
                self.config.poll_interval_s = max(0.5, float(updates["poll_interval_s"]))
            except Exception:
                pass
        if "kismet_cmd" in updates:
            self.config.kismet_cmd = str(updates["kismet_cmd"] or "").strip()
        if "kismet_url" in updates:
            self.config.kismet_url = str(updates["kismet_url"] or "").strip()
        if "kismet_username" in updates:
            self.config.kismet_username = str(updates["kismet_username"] or "").strip()
        if "kismet_password" in updates:
            self.config.kismet_password = str(updates["kismet_password"] or "").strip()
        if "kismet_timeout_s" in updates:
            try:
                self.config.kismet_timeout_s = max(1.0, float(updates["kismet_timeout_s"]))
            except Exception:
                pass
        self.status.backend = self.config.backend
        self.status.iface = self.config.iface
        self.status.channel_mode = self.config.channel_mode
        return self.status.as_dict()

    def get_status(self) -> Dict[str, Any]:
        out = self.status.as_dict()
        out["config"] = {
            "enabled": self.config.enabled,
            "backend": self.config.backend,
            "iface": self.config.iface,
            "channel_mode": self.config.channel_mode,
            "hop_channels": self.config.hop_channels,
            "poll_interval_s": self.config.poll_interval_s,
            "kismet_cmd_configured": bool(self.config.kismet_cmd.strip()),
            "kismet_url": self.config.kismet_url,
            "kismet_username": self.config.kismet_username,
            "kismet_password_configured": bool(self.config.kismet_password),
            "kismet_timeout_s": self.config.kismet_timeout_s,
        }
        out["storage"] = {
            "dir": str(self._store_dir),
            "rid_log": str(self._rid_log_path),
            "wifi_log": str(self._wifi_log_path),
            "recent_rid_count": len(self._recent_rid),
            "recent_wifi_count": len(self._recent_wifi),
        }
        return out

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
                f.write("\n")
        except Exception:
            # Non-fatal: telemetry persistence should not break monitor loop.
            pass

    def _observer_position(self) -> Dict[str, Any]:
        gps = {}
        try:
            gps = dict(getattr(self.cot, "_gps_cache", {}) or {})
        except Exception:
            gps = {}
        lat = gps.get("gps_lat")
        lon = gps.get("gps_lon")
        if lat is None or lon is None:
            return {}
        out: Dict[str, Any] = {"lat": lat, "lon": lon}
        if gps.get("gps_alt_ft") is not None:
            out["alt_ft"] = gps.get("gps_alt_ft")
        if gps.get("gps_fix") is not None:
            out["fix"] = gps.get("gps_fix")
        return out

    def get_recent_events(self, limit: int = 100) -> Dict[str, Any]:
        n = max(1, min(int(limit), 1000))
        return {
            "rid_updates": list(self._recent_rid)[-n:],
            "wifi_intel_updates": list(self._recent_wifi)[-n:],
        }

    def _coerce_lat_lon(self, lat: Any, lon: Any) -> Tuple[float, float] | None:
        try:
            latf = float(lat)
            lonf = float(lon)
        except Exception:
            return None
        if not (-90.0 <= latf <= 90.0 and -180.0 <= lonf <= 180.0):
            return None
        return latf, lonf

    def _emit_cot_if_geolocated(self, payload: Dict[str, Any]) -> None:
        if self.cot is None:
            return
        protocol_result = dict(payload.get("protocol_result", {}) or {})
        fields = dict(protocol_result.get("decoded_fields", {}) or {})
        status = str(protocol_result.get("status", "decoded_partial"))
        confidence = float(protocol_result.get("confidence", 0.0) or 0.0)

        uas_id = str(fields.get("uas_id") or "RID")
        operator_id = str(fields.get("operator_id") or "")
        now = time.time()

        markers = []
        drone_pos = fields.get("drone_position", {}) or {}
        operator_pos = fields.get("operator_position", {}) or {}
        takeoff_pos = fields.get("takeoff_position", {}) or {}

        d = self._coerce_lat_lon(drone_pos.get("lat", fields.get("latitude")), drone_pos.get("lon", fields.get("longitude")))
        if d:
            markers.append(("drone", d[0], d[1], f"RID Drone {uas_id}"))
        o = self._coerce_lat_lon(operator_pos.get("lat", fields.get("operator_lat")), operator_pos.get("lon", fields.get("operator_lon")))
        if o:
            markers.append(("operator", o[0], o[1], f"RID Operator {operator_id or uas_id}"))
        t = self._coerce_lat_lon(takeoff_pos.get("lat", fields.get("takeoff_lat")), takeoff_pos.get("lon", fields.get("takeoff_lon")))
        if t:
            markers.append(("takeoff", t[0], t[1], f"RID Takeoff {uas_id}"))

        if not markers:
            obs = dict(payload.get("observer_position", {}) or {})
            o = self._coerce_lat_lon(obs.get("lat"), obs.get("lon"))
            if o:
                markers.append(("observer", o[0], o[1], f"RID Detection {uas_id}"))
            else:
                return

        for role, lat, lon, callsign in markers:
            uid = f"{self.cot.uid}-rid-{role}-{uas_id}"
            last = self._cot_last_sent.get(uid, 0.0)
            # Throttle repeated updates for same marker.
            if (now - last) < 10.0:
                continue
            remarks = f"RID {role} · status={status} · confidence={confidence:.2f}"
            xml = self.cot.build_alert_cot(
                uid=uid,
                lat=lat,
                lon=lon,
                alert_type="a-u-E-U",
                callsign=callsign,
                remarks=remarks,
                stale_s=120,
            )
            self.cot.send_event(xml)
            self._cot_last_sent[uid] = now

    async def _run_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                result = await self._provider.poll(self.config)
                now = time.time()
                self.status.last_seen_ts = now

                rid_updates = list(result.get("rid_updates", []))
                for upd in rid_updates:
                    self.status.last_decode_ts = now
                    self.status.rid_detections += 1
                    payload = {"source": "RID_WIFI", "ts": now, **upd}
                    obs = self._observer_position()
                    if obs:
                        payload["observer_position"] = obs
                    self.bus.publish_nowait("rid_update", payload)
                    self._emit_cot_if_geolocated(payload)
                    self._recent_rid.append(payload)
                    self._append_jsonl(self._rid_log_path, payload)

                wifi_intel = dict(result.get("wifi_intel", {}))
                if wifi_intel:
                    self.status.wifi_updates += 1
                    wifi_intel.setdefault("ts", now)
                    wifi_intel.setdefault("source", self.config.backend.upper())
                    obs = self._observer_position()
                    if obs:
                        wifi_intel["observer_position"] = obs
                    self.bus.publish_nowait("wifi_intel_update", wifi_intel)
                    self._recent_wifi.append(wifi_intel)
                    self._append_jsonl(self._wifi_log_path, wifi_intel)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.status.error_count += 1
                self.status.last_error = str(e)
                self.bus.publish_nowait(
                    "wifi_intel_update",
                    {
                        "source": self.config.backend.upper(),
                        "ts": time.time(),
                        "anomalies": [{"type": "monitor_loop_error", "detail": str(e)}],
                    },
                )
            await asyncio.sleep(max(0.5, float(self.config.poll_interval_s)))
