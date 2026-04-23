from __future__ import annotations

import logging
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


# ---------------------------------------------------------------------
# ARGB Color Helpers for CoT
# ---------------------------------------------------------------------

def rgb_to_argb_int(r: int, g: int, b: int, a: int = 255) -> int:
    """
    Convert RGBA values to signed ARGB integer for CoT.
    
    Args:
        r: Red (0-255)
        g: Green (0-255)
        b: Blue (0-255)
        a: Alpha (0-255, default 255 = opaque)
    
    Returns:
        Signed 32-bit integer in ARGB format
    """
    value = (a << 24) | (r << 16) | (g << 8) | b
    if value >= 0x80000000:
        value -= 0x100000000
    return int(value)


# Predefined CoT colors (opaque)
COT_RED = rgb_to_argb_int(255, 0, 0, 255)           # -65536
COT_GREEN = rgb_to_argb_int(0, 255, 0, 255)         # -16711936
COT_BLUE = rgb_to_argb_int(0, 0, 255, 255)          # -16776961
COT_YELLOW = rgb_to_argb_int(255, 255, 0, 255)      # -256
COT_ORANGE = rgb_to_argb_int(255, 128, 0, 255)      # -32768
COT_CYAN = rgb_to_argb_int(0, 255, 255, 255)        # -16711681
COT_MAGENTA = rgb_to_argb_int(255, 0, 255, 255)     # -65281
COT_WHITE = rgb_to_argb_int(255, 255, 255, 255)     # -1

# Semi-transparent versions (50% alpha = 128)
COT_RED_50 = rgb_to_argb_int(255, 0, 0, 128)
COT_GREEN_50 = rgb_to_argb_int(0, 255, 0, 128)
COT_BLUE_50 = rgb_to_argb_int(0, 0, 255, 128)
COT_YELLOW_50 = rgb_to_argb_int(255, 255, 0, 128)
COT_ORANGE_50 = rgb_to_argb_int(255, 128, 0, 128)

# Light transparent versions (30% alpha = 77)
COT_RED_30 = rgb_to_argb_int(255, 0, 0, 77)
COT_GREEN_30 = rgb_to_argb_int(0, 255, 0, 77)
COT_YELLOW_30 = rgb_to_argb_int(255, 255, 0, 77)


def confidence_to_color(confidence: float, opaque: bool = True) -> int:
    """
    Map confidence (0-1) to color: red (low) -> yellow (medium) -> green (high).
    
    Args:
        confidence: Value from 0.0 to 1.0
        opaque: If True, return opaque color; if False, return 50% transparent
    
    Returns:
        ARGB color integer
    """
    confidence = max(0.0, min(1.0, confidence))
    alpha = 255 if opaque else 128
    
    if confidence < 0.5:
        # Red to Yellow (0.0 -> 0.5)
        t = confidence * 2  # 0 to 1
        r = 255
        g = int(255 * t)
        b = 0
    else:
        # Yellow to Green (0.5 -> 1.0)
        t = (confidence - 0.5) * 2  # 0 to 1
        r = int(255 * (1 - t))
        g = 255
        b = 0
    
    return rgb_to_argb_int(r, g, b, alpha)


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
        self._last_mc_iface_warn_ts: float = 0.0

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
        interfaces = self._get_interfaces()
        if not interfaces:
            now = time.time()
            if now - self._last_mc_iface_warn_ts >= 60.0:
                self._last_mc_iface_warn_ts = now
                logging.getLogger(__name__).warning(
                    "[CoT] No non-loopback IPv4 interfaces for multicast; "
                    "position/events may not reach ATAK (check networking / WiFi / Ethernet)."
                )
        data = xml.encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack("b", MC_TTL))
        try:
            for _, ip in interfaces:
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
            how = "m-p"
        else:
            alt_m = float(alt_ft or 0.0) * 0.3048
            fix = str(self._gps_cache.get("gps_fix") or "").upper()
            how = "h-g-i-g-o" if fix in ("2D", "3D") else "m-p"

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30))

        return f"""<event version="2.0"
  uid="{self.uid}"
  type="a-f-G-E-S-E"
  how="{how}"
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

    # ------------------------------------------------------------------
    # Circle drawing (u-d-c-c) - for TAI uncertainty areas
    # ------------------------------------------------------------------

    def build_circle_cot(
        self,
        uid: str,
        lat: float,
        lon: float,
        radius_m: float,
        stroke_color: int = COT_RED,
        fill_color: Optional[int] = None,
        callsign: str = "Circle",
        remarks: str = "",
        stale_s: int = 120,
    ) -> str:
        """
        Build CoT circle drawing (type u-d-c-c).
        
        Args:
            uid: Unique identifier for the circle
            lat: Center latitude
            lon: Center longitude
            radius_m: Radius in meters
            stroke_color: ARGB color for outline
            fill_color: ARGB color for fill (defaults to semi-transparent stroke)
            callsign: Display name
            remarks: Description text
            stale_s: Stale time in seconds
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))
        
        if fill_color is None:
            # Make fill 30% opacity version of stroke
            fill_color = rgb_to_argb_int(
                (abs(stroke_color) >> 16) & 0xFF,
                (abs(stroke_color) >> 8) & 0xFF,
                abs(stroke_color) & 0xFF,
                77  # 30% opacity
            )
        
        return f"""<event version="2.0"
  uid="{_xml_escape(uid)}"
  type="u-d-c-c"
  how="h-e"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="0" ce="9999999" le="9999999"/>
  <detail>
    <shape>
      <ellipse major="{radius_m}" minor="{radius_m}" angle="0"/>
    </shape>
    <strokeColor value="{stroke_color}"/>
    <strokeWeight value="3.0"/>
    <fillColor value="{fill_color}"/>
    <contact callsign="{_xml_escape(callsign)}"/>
    <remarks>{_xml_escape(remarks)}</remarks>
    <labels_on value="true"/>
  </detail>
</event>"""

    # ------------------------------------------------------------------
    # Polygon drawing (u-d-f) - for TAI polygon shapes
    # ------------------------------------------------------------------

    def build_polygon_cot(
        self,
        uid: str,
        vertices: List[Tuple[float, float]],  # List of (lat, lon)
        stroke_color: int = COT_GREEN,
        fill_color: Optional[int] = None,
        callsign: str = "Polygon",
        remarks: str = "",
        closed: bool = True,
        stale_s: int = 120,
    ) -> str:
        """
        Build CoT polygon/polyline drawing (type u-d-f).
        
        Args:
            uid: Unique identifier for the polygon
            vertices: List of (lat, lon) tuples defining the shape
            stroke_color: ARGB color for outline
            fill_color: ARGB color for fill
            callsign: Display name
            remarks: Description text
            closed: If True, close the polygon; if False, draw as polyline
            stale_s: Stale time in seconds
        """
        if len(vertices) < 2:
            return ""
        
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))
        
        if fill_color is None:
            fill_color = rgb_to_argb_int(
                (abs(stroke_color) >> 16) & 0xFF,
                (abs(stroke_color) >> 8) & 0xFF,
                abs(stroke_color) & 0xFF,
                77
            )
        
        # Use centroid as the point
        center_lat = sum(v[0] for v in vertices) / len(vertices)
        center_lon = sum(v[1] for v in vertices) / len(vertices)
        
        # Build link elements for vertices
        links = []
        for lat, lon in vertices:
            links.append(f'    <link uid="{_xml_escape(uid)}" type="b-m-p-w" point="{lat},{lon}" relation="p-p"/>')
        links_xml = "\n".join(links)
        
        closed_xml = '<closed value="true"/>' if closed else ''
        
        return f"""<event version="2.0"
  uid="{_xml_escape(uid)}"
  type="u-d-f"
  how="h-e"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{center_lat}" lon="{center_lon}" hae="0" ce="9999999" le="9999999"/>
  <detail>
{links_xml}
    <strokeColor value="{stroke_color}"/>
    <strokeWeight value="2.0"/>
    <fillColor value="{fill_color}"/>
    {closed_xml}
    <contact callsign="{_xml_escape(callsign)}"/>
    <remarks>{_xml_escape(remarks)}</remarks>
  </detail>
</event>"""

    # ------------------------------------------------------------------
    # Bearing line drawing (u-rb-a) - for sensor bearing lines
    # ------------------------------------------------------------------

    def build_bearing_line_cot(
        self,
        uid: str,
        lat: float,
        lon: float,
        bearing_deg: float,
        range_m: float = 5000,
        stroke_color: int = COT_CYAN,
        callsign: str = "Bearing",
        remarks: str = "",
        stale_s: int = 120,
    ) -> str:
        """
        Build CoT range/bearing line (type u-rb-a).
        
        Args:
            uid: Unique identifier for the bearing line
            lat: Origin latitude (sensor position)
            lon: Origin longitude (sensor position)
            bearing_deg: Bearing in degrees (0=North, clockwise)
            range_m: Length of bearing line in meters
            stroke_color: ARGB color for the line
            callsign: Display name
            remarks: Description text
            stale_s: Stale time in seconds
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))
        
        return f"""<event version="2.0"
  uid="{_xml_escape(uid)}"
  type="u-rb-a"
  how="h-e"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="0" ce="9999999" le="9999999"/>
  <detail>
    <range value="{range_m}"/>
    <bearing value="{bearing_deg}"/>
    <inclination value="0"/>
    <strokeColor value="{stroke_color}"/>
    <strokeWeight value="2.0"/>
    <contact callsign="{_xml_escape(callsign)}"/>
    <remarks>{_xml_escape(remarks)}</remarks>
  </detail>
</event>"""

    # ------------------------------------------------------------------
    # Alert (b-a) - for high-confidence detections
    # ------------------------------------------------------------------

    def build_alert_cot(
        self,
        uid: str,
        lat: float,
        lon: float,
        alert_type: str = "b-a",
        callsign: str = "ALERT",
        remarks: str = "",
        color: int = COT_RED,
        stale_s: int = 300,
    ) -> str:
        """
        Build CoT alert message.
        
        Args:
            uid: Unique identifier
            lat: Alert location latitude
            lon: Alert location longitude
            alert_type: Alert CoT type (b-a, b-a-o-pan, etc.)
            callsign: Alert name
            remarks: Alert message
            color: ARGB color
            stale_s: Stale time in seconds
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))
        
        return f"""<event version="2.0"
  uid="{_xml_escape(uid)}"
  type="{alert_type}"
  how="h-e"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="0" ce="50" le="50"/>
  <detail>
    <contact callsign="{_xml_escape(callsign)}"/>
    <remarks>{_xml_escape(remarks)}</remarks>
    <color argb="{color}"/>
  </detail>
</event>"""

    def send_alert(self, lat: float, lon: float, message: str, urgent: bool = False):
        """
        Send an alert to ATAK.
        
        Args:
            lat: Alert location latitude
            lon: Alert location longitude
            message: Alert message
            urgent: If True, use "Ring the Bell" alert type
        """
        alert_type = "b-a-o-pan" if urgent else "b-a"
        callsign = "URGENT" if urgent else "ALERT"
        uid = f"{self.uid}-alert-{int(time.time())}"
        
        xml = self.build_alert_cot(uid, lat, lon, alert_type, callsign, message, COT_RED)
        self.send_event(xml)
        print(f"[ATAK] Alert sent: {message}")

    # ------------------------------------------------------------------
    # TAI Marker (enhanced with circle drawing)
    # ------------------------------------------------------------------

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
        Uses point marker with CE/LE for basic display.
        
        For visual circle, use build_tai_circle() instead.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))

        alt_ft = self._gps_cache.get("gps_alt_ft", 0.0)
        alt_m = float(alt_ft or 0.0) * 0.3048

        uid = f"{self.uid}-tai"
        ce = int(radius_m)
        le = int(radius_m)

        conf_pct = int(confidence * 100)
        qual_pct = int(quality * 100)
        remarks = f"TAI - Confidence: {conf_pct}%, Quality: {qual_pct}%, Radius: {int(radius_m)}m"

        return f"""<event version="2.0"
  uid="{uid}"
  type="a-u-E-U"
  how="m-f"
  time="{now}"
  start="{now}"
  stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="{alt_m}" ce="{ce}" le="{le}"/>
  <detail>
    <contact callsign="{_xml_escape(self.callsign)}-TAI"/>
    <remarks>{_xml_escape(remarks)}</remarks>
  </detail>
</event>"""

    def build_tai_circle(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        confidence: float,
        stale_s: int = 60,
    ) -> str:
        """
        Build CoT circle for TAI visualization.
        Draws an actual visible circle on the map.
        """
        # Color based on confidence
        stroke_color = confidence_to_color(confidence, opaque=True)
        fill_color = confidence_to_color(confidence, opaque=False)
        
        conf_pct = int(confidence * 100)
        remarks = f"TAI Area - Confidence: {conf_pct}%, Radius: {int(radius_m)}m"
        
        return self.build_circle_cot(
            uid=f"{self.uid}-tai-circle",
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            stroke_color=stroke_color,
            fill_color=fill_color,
            callsign="TAI",
            remarks=remarks,
            stale_s=stale_s,
        )

    def build_tai_polygon(
        self,
        vertices: List[Tuple[float, float]],
        confidence: float,
        stale_s: int = 60,
    ) -> str:
        """
        Build CoT polygon for TAI visualization.
        Draws the actual TAI polygon shape on the map.
        """
        stroke_color = confidence_to_color(confidence, opaque=True)
        fill_color = confidence_to_color(confidence, opaque=False)
        
        conf_pct = int(confidence * 100)
        remarks = f"TAI Polygon - Confidence: {conf_pct}%, Vertices: {len(vertices)}"
        
        return self.build_polygon_cot(
            uid=f"{self.uid}-tai-polygon",
            vertices=vertices,
            stroke_color=stroke_color,
            fill_color=fill_color,
            callsign="TAI",
            remarks=remarks,
            closed=True,
            stale_s=stale_s,
        )

    def send_tai(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        confidence: float,
        quality: float,
        polygon_vertices: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Send TAI to ATAK via CoT multicast.
        
        Sends multiple CoT messages:
        1. Point marker with CE/LE (for clients that don't render shapes)
        2. Circle drawing (visible area)
        3. Polygon drawing (if vertices provided)
        """
        # Send point marker (basic)
        xml_marker = self.build_tai_marker(lat, lon, radius_m, confidence, quality)
        self.send_event(xml_marker)
        
        # Send circle (visual)
        xml_circle = self.build_tai_circle(lat, lon, radius_m, confidence)
        self.send_event(xml_circle)
        
        # Send polygon if vertices provided
        if polygon_vertices and len(polygon_vertices) >= 3:
            xml_polygon = self.build_tai_polygon(polygon_vertices, confidence)
            self.send_event(xml_polygon)
        
        print(f"[ATAK] TAI sent: {lat:.6f}, {lon:.6f}, radius={int(radius_m)}m, conf={confidence:.2f}")

    def send_bearing_lines(
        self,
        bearings: List[Dict[str, Any]],
        range_m: float = 5000,
    ):
        """
        Send bearing lines from multiple sensors to ATAK.
        
        Args:
            bearings: List of dicts with keys: node_id, lat, lon, bearing_deg, confidence
            range_m: Length of bearing lines in meters
        """
        for b in bearings:
            node_id = b.get("node_id", "unknown")
            lat = b.get("lat") or (b.get("gps", {}).get("lat"))
            lon = b.get("lon") or (b.get("gps", {}).get("lon"))
            bearing_deg = b.get("bearing_deg")
            confidence = b.get("confidence", 0.5)
            
            if lat is None or lon is None or bearing_deg is None:
                continue
            
            # Color based on confidence
            stroke_color = confidence_to_color(confidence, opaque=True)
            
            callsign = b.get("callsign", node_id)
            remarks = f"Bearing: {bearing_deg:.1f}° from {callsign}"
            
            xml = self.build_bearing_line_cot(
                uid=f"{self.uid}-bearing-{node_id}",
                lat=lat,
                lon=lon,
                bearing_deg=bearing_deg,
                range_m=range_m,
                stroke_color=stroke_color,
                callsign=f"{callsign} BRG",
                remarks=remarks,
            )
            self.send_event(xml)
        
        if bearings:
            print(f"[ATAK] Sent {len(bearings)} bearing lines")

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
