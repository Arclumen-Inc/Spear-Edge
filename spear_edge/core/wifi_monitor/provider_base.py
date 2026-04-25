from __future__ import annotations

from typing import Any, Dict

from .models import WifiMonitorConfig


class WifiProviderBase:
    name = "base"

    async def start(self, config: WifiMonitorConfig) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def poll(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        return {
            "rid_updates": [],
            "wifi_intel": {
                "channels": [],
                "top_emitters": [],
                "frame_mix": {},
                "anomalies": [],
            },
        }

    async def test_connection(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        result = await self.poll(config)
        return {"ok": True, "result": result}

    async def list_datasources(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        return {"ok": False, "error": "datasource listing not supported by provider"}

    async def list_interfaces(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        return {"ok": False, "error": "interface listing not supported by provider"}

    async def add_datasource(self, config: WifiMonitorConfig, source: str) -> Dict[str, Any]:
        return {"ok": False, "error": "add datasource not supported by provider", "source": source}

    async def set_channel(self, config: WifiMonitorConfig, uuid: str, channel: str) -> Dict[str, Any]:
        return {"ok": False, "error": "set channel not supported by provider", "uuid": uuid, "channel": channel}

    async def set_hop(self, config: WifiMonitorConfig, uuid: str, channels: list[str], rate: float) -> Dict[str, Any]:
        return {"ok": False, "error": "set hop not supported by provider", "uuid": uuid}

    async def open_source(self, config: WifiMonitorConfig, uuid: str) -> Dict[str, Any]:
        return {"ok": False, "error": "open source not supported by provider", "uuid": uuid}

    async def close_source(self, config: WifiMonitorConfig, uuid: str) -> Dict[str, Any]:
        return {"ok": False, "error": "close source not supported by provider", "uuid": uuid}

    async def list_alerts(self, config: WifiMonitorConfig) -> Dict[str, Any]:
        return {"ok": False, "error": "alerts listing not supported by provider"}

    async def set_presence_alert(self, config: WifiMonitorConfig, alert_type: str, action: str, mac: str) -> Dict[str, Any]:
        return {"ok": False, "error": "presence alerts not supported by provider"}
