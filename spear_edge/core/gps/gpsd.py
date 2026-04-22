# spear_edge/core/gps/gpsd.py
from __future__ import annotations

import time
import threading
from typing import Optional, Dict, Any

try:
    from gps import gps, WATCH_ENABLE, WATCH_NEWSTYLE
except Exception:
    gps = None


class GpsdClient:
    """
    Lightweight GPSD reader.
    Threaded, poll-based, non-blocking for async apps.
    """

    def __init__(self, poll_interval_s: float = 1.0, host: str = "127.0.0.1", port: int = 2947):
        self.poll_interval_s = poll_interval_s
        self.host = host
        self.port = int(port)

        self.session = None
        self.last_fix: Dict[str, Any] = {}
        self.last_update = 0.0
        self.last_report_ts = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        if gps is None:
            print("[GPS] gps module not available")
            return

        self._connect()

    def _connect(self):
        try:
            self.session = gps(host=self.host, port=str(self.port), mode=WATCH_ENABLE | WATCH_NEWSTYLE)
            print(f"[GPS] Connected to gpsd at {self.host}:{self.port}")
        except Exception as e:
            print(f"[GPS] Failed to connect to gpsd at {self.host}:{self.port}: {e}")
            self.session = None

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def start(self):
        if not self.session:
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[GPS] GPS polling thread started")

    def stop(self):
        self._stop_evt.set()

    def get_status(self) -> Dict[str, Any]:
        """
        Snapshot for /status endpoint.
        """
        age = time.time() - self.last_update if self.last_update else None
        return {
            "fix": self.last_fix.get("gps_fix"),
            "lat": self.last_fix.get("gps_lat"),
            "lon": self.last_fix.get("gps_lon"),
            "alt_ft": self.last_fix.get("gps_alt_ft"),
            "heading_deg": self.last_fix.get("heading_deg"),
            "speed_mps": self.last_fix.get("speed_mps"),
            "time": self.last_fix.get("gps_time"),
            "age_s": age,
            "source": self.last_fix.get("gps_source"),
            "connected": self.session is not None,
        }

    # ------------------------------------------------------------
    # Internal loop (threaded)
    # ------------------------------------------------------------

    def _loop(self):
        while not self._stop_evt.is_set():
            if self.session is None:
                self._connect()
                if self.session is None:
                    time.sleep(max(self.poll_interval_s, 1.0))
                    continue

            try:
                report = self.session.next()
                self.last_report_ts = time.time()
            except Exception:
                self.session = None
                time.sleep(self.poll_interval_s)
                continue

            if report.get("class") != "TPV":
                time.sleep(self.poll_interval_s)
                continue

            lat = getattr(report, "lat", None)
            lon = getattr(report, "lon", None)
            alt = getattr(report, "alt", None)
            mode = getattr(report, "mode", 0)
            # Prefer true heading if available; otherwise use course-over-ground from TPV track.
            heading = getattr(report, "heading", None)
            if heading is None:
                heading = getattr(report, "track", None)
            speed = getattr(report, "speed", None)

            if lat is None or lon is None:
                time.sleep(self.poll_interval_s)
                continue

            fix = "NO FIX"
            if mode == 2:
                fix = "2D"
            elif mode >= 3:
                fix = "3D"

            self.last_fix = {
                "gps_fix": fix,
                "gps_lat": float(lat),
                "gps_lon": float(lon),
                "gps_alt_ft": float(alt * 3.28084) if alt is not None else None,
                "heading_deg": float(heading) if heading is not None else None,
                "speed_mps": float(speed) if speed is not None else None,
                "gps_source": f"{self.host}:{self.port}",
                "gps_time": time.strftime("%H:%M:%SZ", time.gmtime()),
            }
            self.last_update = time.time()

            time.sleep(self.poll_interval_s)
