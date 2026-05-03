"""Tests for built-in ASTM / OpenDroneID Wi‑Fi beacon PCAP decode."""

import struct
import tempfile
import unittest
from pathlib import Path

from spear_edge.core.decode import rid_wifi_pcap_builtin as rw


def _basic_id_msg(uas_id: str = "SERIAL-TEST-001") -> bytes:
    """25-byte OpenDroneID Basic ID (type 0), proto 2."""
    msg = bytearray(25)
    msg[0] = (0 << 4) | 2  # type 0, proto 2
    msg[1] = (1 << 4) | 2  # ID serial, UA helicopter/multirotor
    uid = uas_id.encode("ascii")[:20]
    msg[2 : 2 + len(uid)] = uid
    return bytes(msg)


def _location_msg(lat: float = 37.7749, lon: float = -122.4194) -> bytes:
    """25-byte OpenDroneID Location (type 1), proto 2, airborne."""
    msg = bytearray(25)
    msg[0] = (1 << 4) | 2  # type 1, proto 2
    msg[1] = (2 << 4)  # status airborne, speed_mult=0, ew=0, height_type=0
    msg[2] = 90  # direction
    msg[3] = 40  # 10 m/s horizontal (40 * 0.25)
    msg[4] = 0  # vertical speed 0
    lat_i = int(round(lat * rw.LATLON_MULT))
    lon_i = int(round(lon * rw.LATLON_MULT))
    struct.pack_into("<i", msg, 5, lat_i)
    struct.pack_into("<i", msg, 9, lon_i)
    struct.pack_into("<H", msg, 13, int(round((120.0 + rw.ALT_ADDER) / rw.ALT_DIV)))
    struct.pack_into("<H", msg, 15, int(round((115.0 + rw.ALT_ADDER) / rw.ALT_DIV)))
    struct.pack_into("<H", msg, 17, int(round((50.0 + rw.ALT_ADDER) / rw.ALT_DIV)))
    msg[19] = 0x00
    msg[20] = 0x00
    struct.pack_into("<H", msg, 21, 0xFFFF)
    msg[23] = 0x00
    msg[24] = 0x00
    return bytes(msg)


class TestRidOdidPayload(unittest.TestCase):
    def test_decode_odid_wifi_vendor_inner_basic_and_location(self):
        inner = bytes([42]) + _basic_id_msg() + _location_msg()
        fields, score = rw._decode_odid_wifi_payload(inner)
        self.assertGreaterEqual(score, 2)
        self.assertEqual(fields.get("uas_id"), "SERIAL-TEST-001")
        self.assertIn("drone_position", fields)
        self.assertLess(abs(fields["drone_position"]["lat"] - 37.7749), 1e-5)
        self.assertLess(abs(fields["drone_position"]["lon"] - (-122.4194)), 1e-5)

    def test_decode_odid_message_pack(self):
        """Type 0xF message pack with two inner 25-byte messages."""
        inner_msgs = _basic_id_msg("PACK-UID") + _location_msg(40.0, -75.0)
        pack = bytearray(3 + len(inner_msgs))
        pack[0] = (0xF << 4) | 2
        pack[1] = rw.ODID_MESSAGE_SIZE
        pack[2] = 2
        pack[3:] = inner_msgs
        inner = bytes([1]) + bytes(pack)
        fields, score = rw._decode_odid_wifi_payload(inner)
        self.assertEqual(fields.get("uas_id"), "PACK-UID")
        self.assertLess(abs(fields["drone_position"]["lat"] - 40.0), 1e-5)


def _pcap_one_packet(linktype: int, pkt: bytes) -> bytes:
    # Libpcap LE microsecond: on-disk magic is d4c3b2a1 (constant 0xa1b2c3d4 LE-packed).
    gh = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype)
    ph = struct.pack("<IIII", 0, 0, len(pkt), len(pkt))
    return gh + ph + pkt


def _beacon_with_vendor_ie(inner_after_oui: bytes) -> bytes:
    """802.11 management beacon with one vendor IE 0xDD FA:0B:BC:0D."""
    fc = struct.pack("<H", 0x0080)
    # FC + Duration + DA + SA + BSSID + SeqCtrl = 24 bytes
    hdr = (
        fc
        + struct.pack("<H", 0)
        + (b"\xff" * 6)
        + bytes(range(6))
        + bytes(range(6))
        + struct.pack("<H", 0)
    )
    body = struct.pack("<QHH", 0, 100, 0)
    ie_data = bytes([0xFA, 0x0B, 0xBC, 0x0D]) + inner_after_oui
    ie = bytes([0xDD, len(ie_data)]) + ie_data
    return hdr + body + ie


class TestRidWifiPcapBuiltin(unittest.TestCase):
    def test_decode_astm_from_pcap_roundtrip_105(self):
        self._roundtrip(105)

    def test_decode_astm_from_pcap_roundtrip_radiotap(self):
        self._roundtrip(127)

    def _roundtrip(self, linktype: int) -> None:
        inner = bytes([0]) + _basic_id_msg("PCAP-ROUND") + _location_msg()
        wifi = _beacon_with_vendor_ie(inner)
        if linktype == 127:
            rt = bytearray(8)
            rt[0] = 0
            rt[1] = 0
            struct.pack_into("<H", rt, 2, 8)
            pkt = bytes(rt) + wifi
        else:
            pkt = wifi
        pcap = _pcap_one_packet(linktype, pkt)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.pcap"
            p.write_bytes(pcap)
            out = rw.decode_astm_from_pcap(p)
        self.assertIn(out["status"], ("decoded_verified", "decoded_partial"))
        self.assertEqual(out["decoded_fields"].get("uas_id"), "PCAP-ROUND")


if __name__ == "__main__":
    unittest.main()
