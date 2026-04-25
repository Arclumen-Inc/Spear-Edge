from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WifiMonitorConfig:
    enabled: bool = False
    backend: str = "kismet"
    iface: str = "wlan1"
    channel_mode: str = "hop"
    hop_channels: List[int] = field(default_factory=lambda: [1, 6, 11, 36, 44, 149])
    poll_interval_s: float = 2.0
    kismet_cmd: str = ""
    kismet_url: str = "http://127.0.0.1:2501"
    kismet_username: str = ""
    kismet_password: str = ""
    kismet_timeout_s: float = 3.0


@dataclass
class WifiMonitorStatus:
    enabled: bool = False
    running: bool = False
    backend: str = "kismet"
    iface: str = ""
    channel_mode: str = "hop"
    last_seen_ts: Optional[float] = None
    last_decode_ts: Optional[float] = None
    last_error: str = ""
    error_count: int = 0
    rid_detections: int = 0
    wifi_updates: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self.running,
            "backend": self.backend,
            "iface": self.iface,
            "channel_mode": self.channel_mode,
            "last_seen_ts": self.last_seen_ts,
            "last_decode_ts": self.last_decode_ts,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "rid_detections": self.rid_detections,
            "wifi_updates": self.wifi_updates,
        }
