from __future__ import annotations

import socket
import struct
import threading
import time
import uuid
from typing import Optional, Dict, Any, List, Tuple

import netifaces

# ---------------------------------------------------------------------
# Multicast definitions (TAK standard)
# ---------------------------------------------------------------------

# Position / sensor + general CoT multicast
POS_MC_ADDR = "239.2.3.1"
POS_MC_PORT = 6969

# All Chat
CHAT_MC_ADDR = "224.10.10.1"
CHAT_MC_PORT = 17012

MC_TTL = 1

DEFAULT_UID = "SPEAR-TRIPWIRE"
DEFAULT_CALLSIGN = "SPEAR-TRIPWIRE"


def _xml_escape(s: str) -> str:
    if not s:
        return ""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )


class CoTBroadcaster:
    """
    ATAK / WinTAK CoT broadcaster.

    - Periodic POSITION multicast (self marker)
    - On-demand CHAT multicast (All Chat)
    - RF EVENT CoT builder (truth/event)
    - Detection MARKER CoT builder (persistent rollup marker)
    """

    def __init__(
        self,
        uid: Optional[str] = None,
        callsign: Optional[str] = None,
        interval_s: float = 5.0,
    ):
        self.uid = uid or DEFAULT_UID
        self.callsign = callsign or DEFAULT_CALLSIGN
        self.interval_s = interval_s

        self._gps_cache: Dict[str, Any] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def set_identity(self, uid: Optional[str] = None, callsign: Optional[str] = None):
        if uid:
            self.uid = uid
        if callsign:
            self.callsign = callsign

    def update_gps(self, gps: Dict[str, Any]):
        self._gps_cache = dict(gps)

    def _get_interfaces(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for iface in netifaces.interfaces():
            if iface == "lo":
                continue
            try:
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET not in addrs:
                    continue
                ip = addrs[netifaces.AF_INET][0].get("addr")
                if ip:
                    out.append((iface, ip))
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # Generic CoT sender (POS multicast group)
    # ------------------------------------------------------------------

    def send_event(self, xml: str):
        data = xml.encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack("b", MC_TTL))
        try:
            for _, ip in self._get_interfaces():
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
                sock.sendto(data, (POS_MC_ADDR, POS_MC_PORT))
        finally:
            sock.close()

    # ------------------------------------------------------------------
    # POSITION CoT (self marker)
    # ------------------------------------------------------------------

    def build_current_position_xml(self) -> str:
        lat = self._gps_cache.get("gps_lat")
        lon = self._gps_cache.get("gps_lon")
        alt_ft = self._gps_cache.get("gps_alt_ft")

        if lat is None or lon is None:
            lat, lon, alt_m = 0.0, 0.0, 0.0
        else:
            alt_m = float(alt_ft or 0.0) * 0.3048

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30))

        return f"""<event version="2.0"
  uid="{self.uid}"
  type="a-f-G-E-S-E"
  how="m-p"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="{alt_m}" ce="5" le="5"/>
  <detail>
    <contact callsign="{_xml_escape(self.callsign)}"/>
    <track speed="0" course="0"/>
  </detail>
</event>"""

    # ------------------------------------------------------------------
    # CHAT CoT (All Chat)
    # ------------------------------------------------------------------

    def build_chat_cot(self, message: str, room: str = "All Chat") -> str:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 300))

        uid = f"Chat-{uuid.uuid4()}"

        msg = _xml_escape(message)
        sender = _xml_escape(self.callsign)
        room_esc = _xml_escape(room)

        # NOTE: <remarks> is included because many ATAK builds render chat text from remarks.
        return f"""<event version="2.0"
  uid="{uid}"
  type="b-t-f"
  how="h-g-i-g-o"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="0" lon="0" hae="0" ce="9999999" le="9999999"/>
  <detail>
    <__chat
      id="{room_esc}"
      chatroom="{room_esc}"
      senderCallsign="{sender}"
      message="{msg}"/>
    <remarks>{msg}</remarks>
  </detail>
</event>"""

    def send_chat(self, message: str, room: str = "All Chat"):
        xml = self.build_chat_cot(message, room).encode("utf-8")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack("b", MC_TTL))

        try:
            for _, ip in self._get_interfaces():
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
                sock.sendto(xml, (CHAT_MC_ADDR, CHAT_MC_PORT))
        finally:
            sock.close()

    # ------------------------------------------------------------------
    # RF EVENT CoT (truth / transient)
    # ------------------------------------------------------------------

    def build_rf_event_cot(
        self,
        freq_hz: float,
        delta_db: float,
        stale_s: int = 120,
    ) -> str:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))

        lat = self._gps_cache.get("gps_lat", 0.0)
        lon = self._gps_cache.get("gps_lon", 0.0)
        alt_ft = self._gps_cache.get("gps_alt_ft", 0.0)
        alt_m = float(alt_ft or 0.0) * 0.3048

        uid = f"{self.uid}-rf-{int(time.time())}"
        freq_mhz = freq_hz / 1e6

        remarks = _xml_escape(f"RF Spike {freq_mhz:.3f} MHz ?{delta_db:.1f} dB")

        return f"""<event version="2.0"
  uid="{uid}"
  type="a-u-A-E-I"
  how="m-p"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="{alt_m}" ce="20" le="20"/>
  <detail>
    <contact callsign="{_xml_escape(self.callsign)}"/>
    <remarks>{remarks}</remarks>
  </detail>
</event>"""

    # ------------------------------------------------------------------
    # Detection / Rollup marker (persistent map object)
    # ------------------------------------------------------------------

    def build_detection_marker(self, remarks: str, stale_s: int = 120) -> str:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))

        lat = self._gps_cache.get("gps_lat", 0.0)
        lon = self._gps_cache.get("gps_lon", 0.0)
        alt_ft = self._gps_cache.get("gps_alt_ft", 0.0)
        alt_m = float(alt_ft or 0.0) * 0.3048

        # Stable UID -> one marker that updates in place (MANET-safe)
        uid = f"{self.uid}-detections"

        return f"""<event version="2.0"
  uid="{uid}"
  type="a-u-E-U"
  how="m-p"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="{alt_m}" ce="20" le="20"/>
  <detail>
    <contact callsign="{_xml_escape(self.callsign)}"/>
    <remarks>{_xml_escape(remarks)}</remarks>
  </detail>
</event>"""

    def build_tai_marker(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        confidence: float,
        quality: float,
        stale_s: int = 60,
    ) -> str:
        """
        Build CoT marker for TAI (Targeted Area of Interest) from AoA fusion.
        
        Args:
            lat: Latitude of TAI center
            lon: Longitude of TAI center
            radius_m: Uncertainty radius in meters
            confidence: Average confidence (0-1)
            quality: Fusion quality metric (0-1)
            stale_s: Stale time in seconds
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))

        # Use Edge GPS altitude if available, else 0
        alt_ft = self._gps_cache.get("gps_alt_ft", 0.0)
        alt_m = float(alt_ft or 0.0) * 0.3048

        # Stable UID -> one marker that updates in place
        uid = f"{self.uid}-tai"

        # Calculate circular error (CE) and linear error (LE) from radius
        # CE/LE represent uncertainty ellipse - use radius for both
        ce = int(radius_m)
        le = int(radius_m)

        # Build remarks with TAI info
        conf_pct = int(confidence * 100)
        qual_pct = int(quality * 100)
        remarks = f"TAI (Targeted Area of Interest) - Confidence: {conf_pct}%, Quality: {qual_pct}%, Radius: {int(radius_m)}m"

        return f"""<event version="2.0"
  uid="{uid}"
  type="a-u-E-U"
  how="m-p"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="{alt_m}" ce="{ce}" le="{le}"/>
  <detail>
    <contact callsign="{_xml_escape(self.callsign)}"/>
    <remarks>{_xml_escape(remarks)}</remarks>
  </detail>
</event>"""

    def send_tai(self, lat: float, lon: float, radius_m: float, confidence: float, quality: float):
        """Send TAI marker to ATAK via CoT multicast."""
        xml = self.build_tai_marker(lat, lon, radius_m, confidence, quality)
        self.send_event(xml)
        print(f"[ATAK] TAI sent: {lat:.6f}, {lon:.6f}, radius={int(radius_m)}m, conf={confidence:.2f}")

    # ------------------------------------------------------------------
    # POSITION LOOP
    # ------------------------------------------------------------------

    def _loop(self):
        print("[COT] Position multicast loop running")
        while not self._stop_evt.is_set():
            try:
                self.send_event(self.build_current_position_xml())
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[COT] Started Tripwire CoT broadcaster")

    def stop(self):
        self._stop_evt.set()
