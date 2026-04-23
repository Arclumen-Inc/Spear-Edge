"""CoT ownship position follows GpsdClient-shaped GPS updates."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from spear_edge.core.integrate.cot import CoTBroadcaster


def _point_lat_lon(xml: str) -> tuple[float, float]:
    root = ET.fromstring(xml)
    pt = root.find("point")
    assert pt is not None
    return float(pt.get("lat", "0")), float(pt.get("lon", "0"))


def test_update_gps_populates_position_cot() -> None:
    cot = CoTBroadcaster(uid="SPEAR-EDGE", callsign="SPEAR-EDGE")
    cot.update_gps(
        {
            "gps_fix": "3D",
            "gps_lat": 37.7749,
            "gps_lon": -122.4194,
            "gps_alt_ft": 100.0,
        }
    )
    xml = cot.build_current_position_xml()
    lat, lon = _point_lat_lon(xml)
    assert abs(lat - 37.7749) < 1e-6
    assert abs(lon - (-122.4194)) < 1e-6
    assert 'how="h-g-i-g-o"' in xml


def test_no_fix_uses_placeholder_and_m_p_how() -> None:
    cot = CoTBroadcaster(uid="SPEAR-EDGE", callsign="SPEAR-EDGE")
    cot.update_gps({})
    xml = cot.build_current_position_xml()
    lat, lon = _point_lat_lon(xml)
    assert lat == 0.0 and lon == 0.0
    assert 'how="m-p"' in xml


def test_2d_fix_uses_h_g_i_g_o() -> None:
    cot = CoTBroadcaster(uid="SPEAR-EDGE", callsign="SPEAR-EDGE")
    cot.update_gps(
        {
            "gps_fix": "2D",
            "gps_lat": 40.0,
            "gps_lon": -75.0,
            "gps_alt_ft": 0.0,
        }
    )
    xml = cot.build_current_position_xml()
    assert 'how="h-g-i-g-o"' in xml
