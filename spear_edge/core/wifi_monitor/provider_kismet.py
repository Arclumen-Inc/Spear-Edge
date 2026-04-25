from __future__ import annotations

import asyncio
import base64
import json
import shlex
import subprocess
import time
import urllib.request
from typing import Any, Dict, List, Tuple

from .models import WifiMonitorConfig
from .provider_base import WifiProviderBase


class KismetProvider(WifiProviderBase):
    """
    Kismet-backed provider adapter.

    This adapter intentionally stays lightweight: it can execute an optional
    external command that emits one JSON object per invocation and normalize
    the result into SPEAR's rid_update / wifi_intel_update contract.
    """

    name = "kismet"

    def _auth_header(self, config: WifiMonitorConfig) -> Dict[str, str]:
        if not config.kismet_username:
            return {}
        token = f"{config.kismet_username}:{config.kismet_password or ''}".encode("utf-8")
        b64 = base64.b64encode(token).decode("ascii")
        return {"Authorization": f"Basic {b64}"}

    def _join_url(self, base: str, path: str) -> str:
        b = (base or "").rstrip("/")
        p = path if path.startswith("/") else f"/{path}"
        return f"{b}{p}"

    async def _http_json(self, url: str, headers: Dict[str, str], timeout_s: float) -> Tuple[bool, Any, str]:
        req = urllib.request.Request(url=url, method="GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                return True, json.loads(body), ""
        except Exception as e:
            return False, None, str(e)

    async def _http_json_post(
        self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: float
    ) -> Tuple[bool, Any, str]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            method="POST",
            headers={**headers, "Content-Type": "application/json"},
            data=body,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
                return True, json.loads(raw) if raw else {}, ""
        except Exception as e:
            return False, None, str(e)

    def _parse_frame_mix(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Best-effort aggregation from heterogeneous Kismet fields.
        out: Dict[str, int] = {"beacon": 0, "probe": 0, "data": 0, "mgmt": 0}
        for d in devices:
            dot11 = d.get("kismet.device.base.packets.llc", {}) or d.get("dot11", {}) or {}
            if isinstance(dot11, dict):
                for key in list(out.keys()):
                    val = dot11.get(key) or dot11.get(f"dot11.{key}")
                    if isinstance(val, (int, float)):
                        out[key] += int(val)
        return out

    def _parse_emitters(self, devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        emitters: List[Dict[str, Any]] = []
        for d in devices:
            mac = (
                d.get("kismet.device.base.macaddr")
                or d.get("kismet.device.base.key")
                or d.get("mac")
                or d.get("bssid")
            )
            if not mac:
                continue
            vendor = d.get("kismet.device.base.manuf") or d.get("vendor") or "unknown"
            packets = (
                d.get("kismet.device.base.packets.total")
                or d.get("packets")
                or d.get("kismet.device.base.datasize")
                or 0
            )
            devtype = d.get("kismet.device.base.type") or d.get("type") or "unknown"
            channel = d.get("kismet.device.base.channel") or d.get("channel")
            signal = d.get("kismet.device.base.signal") or d.get("signal")
            emitters.append(
                {
                    "mac": str(mac),
                    "vendor": str(vendor),
                    "packets": int(packets or 0),
                    "type": str(devtype),
                    "channel": channel,
                    "signal": signal,
                }
            )
        emitters.sort(key=lambda x: x.get("packets", 0), reverse=True)
        return emitters[:20]

    def _parse_channels(self, channels_raw: Any) -> List[Dict[str, Any]]:
        channels: List[Dict[str, Any]] = []
        if isinstance(channels_raw, list):
            for item in channels_raw:
                if not isinstance(item, dict):
                    continue
                ch = item.get("kismet.channelrec.channel") or item.get("channel")
                pk = (
                    item.get("kismet.channelrec.packets")
                    or item.get("packets")
                    or item.get("kismet.channelrec.signal_rrd")
                    or 0
                )
                if ch is None:
                    continue
                channels.append({"channel": ch, "packets": int(pk or 0)})
        channels.sort(key=lambda x: x.get("packets", 0), reverse=True)
        return channels[:30]

    def _extract_rid_updates(self, devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        for d in devices:
            rid = d.get("rid") or d.get("remote_id") or d.get("droneid")
            if not isinstance(rid, dict):
                continue
            status = str(rid.get("status", "decoded_partial"))
            fields = dict(rid.get("decoded_fields", {}) or {})
            # Normalize common aliases for location/operator.
            if "uas_id" not in fields:
                for k in ("serial", "serial_no", "id", "uasId"):
                    if rid.get(k):
                        fields["uas_id"] = rid.get(k)
                        break
            if "operator_id" not in fields and rid.get("operator_id"):
                fields["operator_id"] = rid.get("operator_id")
            for pos_key, lk, ok in (
                ("drone_position", "latitude", "longitude"),
                ("operator_position", "operator_lat", "operator_lon"),
                ("takeoff_position", "home_latitude", "home_longitude"),
            ):
                if pos_key not in fields and rid.get(lk) is not None and rid.get(ok) is not None:
                    fields[pos_key] = {"lat": rid.get(lk), "lon": rid.get(ok), "altitude_m": rid.get("altitude_m")}
            updates.append(
                {
                    "protocol_result": {
                        "protocol": "remote_id",
                        "status": status,
                        "confidence": float(rid.get("confidence", 0.6) or 0.6),
                        "decoded_fields": fields,
                        "validation": dict(rid.get("validation", {}) or {}),
                        "evidence": {"reason": "kismet_rid_extract"},
                    },
                    "source": "RID_WIFI",
                    "ts": time.time(),
                }
            )
        return updates

    async def _poll_kismet_http(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        base = (config.kismet_url or "").strip()
        if not base:
            return {
                "rid_updates": [],
                "wifi_intel": {
                    "source": "kismet",
                    "ts": time.time(),
                    "channels": [],
                    "top_emitters": [],
                    "frame_mix": {},
                    "anomalies": [],
                    "status": "idle_no_kismet_url",
                },
            }

        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        status_url = self._join_url(base, "/system/status.json")
        devices_url = self._join_url(base, "/devices/views/phy/IEEE802.11/devices.json")
        devices_fallback_url = self._join_url(base, "/devices/views/phy-IEEE802.11/devices.json")
        channels_url = self._join_url(base, "/channels/channels.json")
        datasources_url = self._join_url(base, "/datasource/all_sources.json")

        ok_status, status_data, err_status = await self._http_json(status_url, headers, timeout_s)
        ok_devices, devices_data, err_devices = await self._http_json(devices_url, headers, timeout_s)
        if not ok_devices:
            ok_devices, devices_data, err_devices = await self._http_json(devices_fallback_url, headers, timeout_s)
        ok_channels, channels_data, err_channels = await self._http_json(channels_url, headers, timeout_s)
        ok_sources, sources_data, err_sources = await self._http_json(datasources_url, headers, timeout_s)

        anomalies = []
        if not ok_status:
            anomalies.append({"type": "kismet_status_error", "detail": err_status})
        if not ok_devices:
            anomalies.append({"type": "kismet_devices_error", "detail": err_devices})
        if not ok_channels:
            anomalies.append({"type": "kismet_channels_error", "detail": err_channels})
        if not ok_sources:
            anomalies.append({"type": "kismet_sources_error", "detail": err_sources})

        devices = devices_data if isinstance(devices_data, list) else []
        rid_updates = self._extract_rid_updates(devices) if ok_devices else []

        frame_mix = self._parse_frame_mix(devices) if ok_devices else {}
        emitters = self._parse_emitters(devices) if ok_devices else []
        channels = self._parse_channels(channels_data) if ok_channels else []
        datasources = sources_data if isinstance(sources_data, list) else []
        datasources_summary = []
        for src in datasources:
            name = src.get("kismet.datasource.name") or src.get("name") or src.get("kismet.datasource.uuid") or "source"
            running = bool(src.get("kismet.datasource.running", src.get("running", False)))
            packets = src.get("kismet.datasource.num_packets") or src.get("num_packets") or 0
            chan = src.get("kismet.datasource.channel") or src.get("channel")
            hop = src.get("kismet.datasource.hopping") or src.get("hopping")
            datasources_summary.append(
                {
                    "name": str(name),
                    "running": running,
                    "packets": int(packets or 0),
                    "channel": chan,
                    "hopping": bool(hop) if hop is not None else False,
                }
            )

        status_text = "ok"
        if anomalies:
            status_text = "partial_error"
        if not ok_status and not ok_devices and not ok_channels:
            status_text = "unreachable"

        return {
            "rid_updates": rid_updates,
            "wifi_intel": {
                "source": "kismet",
                "ts": time.time(),
                "channels": channels,
                "top_emitters": emitters,
                "frame_mix": frame_mix,
                "anomalies": anomalies,
                "mesh_hints": [],
                "status": status_text,
                "kismet": status_data if isinstance(status_data, dict) else {},
                "datasources": datasources_summary,
                "devices": emitters,
            },
        }

    async def test_connection(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        out = await self._poll_kismet_http(config)
        intel = dict(out.get("wifi_intel", {}))
        anomalies = list(intel.get("anomalies", []))
        status = str(intel.get("status", "unknown"))
        ok = status not in {"unreachable"} and not any(
            str(a.get("type", "")).endswith("_error") for a in anomalies
        )
        return {
            "ok": ok,
            "backend": "kismet",
            "status": status,
            "anomalies": anomalies,
            "rid_candidates": len(out.get("rid_updates", [])),
            "channels_seen": len(intel.get("channels", [])),
            "emitters_seen": len(intel.get("top_emitters", [])),
        }

    async def list_datasources(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        url = self._join_url(config.kismet_url, "/datasource/all_sources.json")
        ok, data, err = await self._http_json(url, headers, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "datasources": data if isinstance(data, list) else []}

    async def list_interfaces(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        url = self._join_url(config.kismet_url, "/datasource/list_interfaces.json")
        ok, data, err = await self._http_json(url, headers, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "interfaces": data if isinstance(data, list) else []}

    async def add_datasource(self, config: WifiMonitorConfig, source: str) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        url = self._join_url(config.kismet_url, "/datasource/add_source.cmd")
        ok, data, err = await self._http_json_post(url, headers, {"source": source}, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "result": data}

    async def set_channel(self, config: WifiMonitorConfig, uuid: str, channel: str) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        url = self._join_url(config.kismet_url, f"/datasource/by-uuid/{uuid}/set_channel.cmd")
        ok, data, err = await self._http_json_post(url, headers, {"channel": channel}, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "result": data}

    async def set_hop(self, config: WifiMonitorConfig, uuid: str, channels: list[str], rate: float) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        url = self._join_url(config.kismet_url, f"/datasource/by-uuid/{uuid}/set_channel.cmd")
        payload = {"channels": channels, "rate": rate, "shuffle": 1}
        ok, data, err = await self._http_json_post(url, headers, payload, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "result": data}

    async def open_source(self, config: WifiMonitorConfig, uuid: str) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        url = self._join_url(config.kismet_url, f"/datasource/by-uuid/{uuid}/open_source.cmd")
        ok, data, err = await self._http_json_post(url, headers, {}, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "result": data}

    async def close_source(self, config: WifiMonitorConfig, uuid: str) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        url = self._join_url(config.kismet_url, f"/datasource/by-uuid/{uuid}/close_source.cmd")
        ok, data, err = await self._http_json_post(url, headers, {}, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "result": data}

    async def list_alerts(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        url = self._join_url(config.kismet_url, "/alerts/definitions/alerts.json")
        ok, data, err = await self._http_json(url, headers, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "alerts": data if isinstance(data, list) else []}

    async def set_presence_alert(self, config: WifiMonitorConfig, alert_type: str, action: str, mac: str) -> Dict[str, Any]:
        headers = {"Accept": "application/json", **self._auth_header(config)}
        timeout_s = max(1.0, float(config.kismet_timeout_s or 3.0))
        at = "devicefound" if str(alert_type).lower() != "devicelost" else "devicelost"
        act = "add" if str(action).lower() != "remove" else "remove"
        url = self._join_url(config.kismet_url, f"/devices/alerts/mac/{at}/{act}.cmd")
        ok, data, err = await self._http_json_post(url, headers, {"mac": mac}, timeout_s)
        if not ok:
            return {"ok": False, "error": err}
        return {"ok": True, "result": data}

    async def poll(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        # Preferred path: real Kismet HTTP polling.
        if (config.kismet_url or "").strip():
            return await self._poll_kismet_http(config)

        # Fallback path: command output adapter.
        cmd = (config.kismet_cmd or "").strip()
        if not cmd:
            now = time.time()
            return {
                "rid_updates": [],
                "wifi_intel": {
                    "source": "kismet",
                    "ts": now,
                    "channels": [],
                    "top_emitters": [],
                    "frame_mix": {},
                    "anomalies": [],
                    "status": "idle_no_kismet_cmd",
                },
            }

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                shlex.split(cmd),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception as e:
            return {
                "rid_updates": [],
                "wifi_intel": {
                    "source": "kismet",
                    "ts": time.time(),
                    "channels": [],
                    "top_emitters": [],
                    "frame_mix": {},
                    "anomalies": [{"type": "provider_error", "detail": str(e)}],
                },
            }

        if proc.returncode != 0:
            return {
                "rid_updates": [],
                "wifi_intel": {
                    "source": "kismet",
                    "ts": time.time(),
                    "channels": [],
                    "top_emitters": [],
                    "frame_mix": {},
                    "anomalies": [
                        {
                            "type": "kismet_cmd_nonzero_exit",
                            "detail": (proc.stderr or proc.stdout or "").strip()[-400:],
                        }
                    ],
                },
            }

        txt = (proc.stdout or "").strip()
        if not txt:
            return {
                "rid_updates": [],
                "wifi_intel": {
                    "source": "kismet",
                    "ts": time.time(),
                    "channels": [],
                    "top_emitters": [],
                    "frame_mix": {},
                    "anomalies": [],
                    "status": "no_output",
                },
            }

        try:
            raw = json.loads(txt)
        except Exception:
            return {
                "rid_updates": [],
                "wifi_intel": {
                    "source": "kismet",
                    "ts": time.time(),
                    "channels": [],
                    "top_emitters": [],
                    "frame_mix": {},
                    "anomalies": [{"type": "invalid_json_output"}],
                },
            }

        return {
            "rid_updates": list(raw.get("rid_updates", [])),
            "wifi_intel": {
                "source": "kismet",
                "ts": raw.get("ts", time.time()),
                "channels": list(raw.get("channels", [])),
                "top_emitters": list(raw.get("top_emitters", [])),
                "frame_mix": dict(raw.get("frame_mix", {})),
                "anomalies": list(raw.get("anomalies", [])),
                "mesh_hints": list(raw.get("mesh_hints", [])),
            },
        }
