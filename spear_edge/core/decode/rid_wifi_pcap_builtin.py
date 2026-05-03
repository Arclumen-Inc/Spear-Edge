"""
Built-in ASTM / OpenDroneID Wi‑Fi beacon decode from PCAP or PCAPNG sidecars.

No extra Python dependencies (stdlib only). Intended for monitor-mode captures
placed next to ``samples.iq`` (see ``scripts/rid_pcap_sidecar_backend.py`` names).

Extracts Vendor Specific IE 221 (0xDD) with OUI FA:0B:BC (ASD-STAN ASTM path),
then unpacks OpenDroneID Basic ID, Location, Operator ID, System, Self ID,
and Message Pack payloads (25-byte messages, high nibble of byte 0 = type).
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROTOCOL_NAME = "remote_id"

# ASTM / ASD-STAN Wi‑Fi beacon vendor IE (common; some gear uses oui_type 0x01)
RID_OUI: Tuple[int, int, int] = (0xFA, 0x0B, 0xBC)
RID_OUI_TYPES: Tuple[int, ...] = (0x0D, 0x01)

ODID_MESSAGE_SIZE = 25
LATLON_MULT = 10_000_000
ALT_DIV = 0.5
ALT_ADDER = 1000
SPEED_DIV0 = 0.25
SPEED_DIV1 = 0.75
VSPEED_DIV = 0.5

INV_TIMESTAMP = 0xFFFF

_SIDECAR_NAMES = (
    "remote_rid.pcapng",
    "remote_rid.pcap",
    "rid_sidecar.pcapng",
    "rid_sidecar.pcap",
)


def find_rid_sidecar(iq_path: Path) -> Optional[Path]:
    parent = iq_path.parent
    for name in _SIDECAR_NAMES:
        p = parent / name
        if p.is_file():
            return p.resolve()
    return None


def decode_rid_iq_path(iq_path: Path) -> Dict[str, Any]:
    """Decode using a PCAP sidecar next to ``iq_path``, or ``no_decode`` if missing."""
    sidecar = find_rid_sidecar(iq_path)
    if sidecar is None:
        return _no_decode(
            "rid_pcap_sidecar_not_found",
            iq_path=str(iq_path.resolve()),
            searched_in=str(iq_path.parent.resolve()),
            sidecar_names=list(_SIDECAR_NAMES),
        )
    return decode_astm_from_pcap(sidecar)


def decode_astm_from_pcap(pcap_path: Path) -> Dict[str, Any]:
    """Parse ``pcap_path`` and return SPEAR ``decode/remote_id.json`` schema."""
    try:
        raw = pcap_path.read_bytes()
    except OSError as e:
        return _decode_error("pcap_read_failed", error=str(e), path=str(pcap_path))

    packets: List[Tuple[int, bytes]] = []  # (linktype, mac_payload)
    for linktype, pkt in _iter_linklayer_packets(raw):
        wifi = _extract_80211(linktype, pkt)
        if wifi:
            packets.append((linktype, wifi))

    best_fields: Dict[str, Any] = {}
    best_score = 0
    frames_with_rid = 0

    for _lt, wifi in packets:
        for payload in _iter_astm_vendor_payloads(wifi):
            frames_with_rid += 1
            fields, score = _decode_odid_wifi_payload(payload)
            if score > best_score:
                best_score = score
                best_fields = fields

    if not best_fields:
        if frames_with_rid == 0:
            return _no_decode(
                "rid_wifi_no_odid_ie",
                path=str(pcap_path),
                hint="Expected vendor IE 0xDD with OUI FA:0B:BC (ASTM Wi‑Fi beacon).",
            )
        return _no_decode(
            "rid_wifi_odid_parse_empty",
            path=str(pcap_path),
            frames_with_rid=frames_with_rid,
        )

    if best_score >= 5:
        status = "decoded_verified"
        confidence = 0.99
    elif best_score >= 2:
        status = "decoded_partial"
        confidence = 0.55
    else:
        status = "decoded_partial"
        confidence = 0.35

    validation = {
        "crc_pass": False,
        "frame_count": frames_with_rid,
    }

    return {
        "protocol": PROTOCOL_NAME,
        "status": status,
        "confidence": confidence,
        "validation": validation,
        "decoded_fields": best_fields,
        "evidence": {
            "reason": "builtin_wifi_beacon_opendroneid",
            "pcap_path": str(pcap_path.resolve()),
            "parser": "rid_wifi_pcap_builtin",
        },
        "generated_at_ts": time.time(),
    }


# --- PCAP / PCAPNG ---


def _iter_linklayer_packets(raw: bytes) -> Iterator[Tuple[int, bytes]]:
    if len(raw) >= 4 and raw[:4] == b"\n\r\r\n":
        yield from _iter_pcapng(raw)
        return
    # Classic libpcap: LE a1b2c3d4 or swapped (BE) d4c3b2a1
    if len(raw) >= 24 and raw[:4] in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4"):
        yield from _iter_classic_pcap(raw)
        return


def _iter_classic_pcap(raw: bytes) -> Iterator[Tuple[int, bytes]]:
    magic = raw[:4]
    # First 4 bytes on disk: d4c3b2a1 = LE microsecond (typical); a1b2c3d4 = swapped / BE.
    use_be = magic == b"\xa1\xb2\xc3\xd4"
    if magic not in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4"):
        return
    fmt = ">IHHIIII" if use_be else "<IHHIIII"
    _m, _vmaj, _vmin, _zone, _sig, _snap, network = struct.unpack_from(fmt, raw, 0)
    linktype = network
    pos = 24
    ph_fmt = ">IIII" if use_be else "<IIII"
    while pos + 16 <= len(raw):
        _ts_sec, _ts_usec, incl_len, _orig_len = struct.unpack_from(ph_fmt, raw, pos)
        pos += 16
        if pos + incl_len > len(raw):
            break
        pkt = raw[pos : pos + incl_len]
        pos += incl_len
        yield linktype, pkt


def _iter_pcapng(raw: bytes) -> Iterator[Tuple[int, bytes]]:
    pos = 0
    linktype = 105  # DLT_IEEE802_11 until we see an Interface Description Block
    n = len(raw)
    while pos + 8 <= n:
        block_type, block_len = struct.unpack_from("<II", raw, pos)
        if block_len < 12 or pos + block_len > n:
            break
        block = raw[pos : pos + block_len]
        if block_type == 0x00000001 and len(block) >= 12:
            linktype = struct.unpack_from("<H", block, 8)[0]
        elif block_type == 0x00000006 and len(block) >= 28:
            cap_len = struct.unpack_from("<I", block, 20)[0]
            pkt_start = 28
            if pkt_start + cap_len <= len(block):
                yield linktype, block[pkt_start : pkt_start + cap_len]
        pos += block_len


def _extract_80211(linktype: int, pkt: bytes) -> Optional[bytes]:
    if linktype == 127:  # DLT_IEEE802_11_RADIO
        if len(pkt) < 4:
            return None
        rt_len = struct.unpack_from("<H", pkt, 2)[0]
        if rt_len <= 0 or rt_len > len(pkt):
            return None
        return pkt[rt_len:]
    if linktype == 105:  # DLT_IEEE802_11
        return pkt
    return None


# --- 802.11 beacon IE scan ---


def _iter_astm_vendor_payloads(frame: bytes) -> Iterator[bytes]:
    if len(frame) < 36:
        return
    fc = struct.unpack_from("<H", frame, 0)[0]
    ftype = (fc >> 2) & 3
    subtype = (fc >> 4) & 0xF
    if ftype != 0 or subtype != 8:
        return
    body = frame[24:]
    if len(body) < 12:
        return
    ies = body[12:]
    offset = 0
    while offset + 2 <= len(ies):
        ie_id = ies[offset]
        ie_len = ies[offset + 1]
        offset += 2
        if offset + ie_len > len(ies):
            break
        data = ies[offset : offset + ie_len]
        offset += ie_len
        if ie_id != 0xDD or ie_len < 4:
            continue
        if tuple(data[:3]) != RID_OUI:
            continue
        if data[3] not in RID_OUI_TYPES:
            continue
        inner = data[4:]
        if inner:
            yield inner


# --- OpenDroneID decode ---


def _decode_odid_wifi_payload(payload: bytes) -> Tuple[Dict[str, Any], int]:
    """
    Wi‑Fi vendor payload after OUI+type: [counter byte] + ODID bytes (25-byte
    messages and/or ASTM message pack blobs longer than 25 bytes).
    """
    accum: Dict[str, Any] = {}
    score = 0
    if not payload:
        return accum, 0
    data = payload[1:] if len(payload) > 1 else b""
    if not data:
        return accum, 0

    pos = 0
    while pos < len(data):
        if pos + 1 > len(data):
            break
        b0 = data[pos]
        mtype = (b0 >> 4) & 0x0F
        if mtype == 0x0F:
            if pos + 3 > len(data):
                pos += 1
                continue
            single_size = data[pos + 1]
            npack = data[pos + 2]
            if single_size != ODID_MESSAGE_SIZE or npack < 1 or npack > 9:
                pos += 1
                continue
            total = 3 + npack * ODID_MESSAGE_SIZE
            if pos + total > len(data):
                pos += 1
                continue
            blob = data[pos : pos + total]
            _unpack_message_pack_blob(blob, accum)
            score += 2
            pos += total
        elif mtype <= 6:
            if pos + ODID_MESSAGE_SIZE > len(data):
                break
            msg = data[pos : pos + ODID_MESSAGE_SIZE]
            _decode_single_message(msg, accum)
            score += 1
            pos += ODID_MESSAGE_SIZE
        else:
            pos += 1

    return accum, score


def _unpack_message_pack_blob(blob: bytes, accum: Dict[str, Any]) -> None:
    """Decode full message pack (3-byte header + n * 25 bytes)."""
    if len(blob) < 3:
        return
    npack = blob[2]
    base = 3
    for i in range(npack):
        start = base + i * ODID_MESSAGE_SIZE
        end = start + ODID_MESSAGE_SIZE
        if end > len(blob):
            break
        sub = blob[start:end]
        _decode_single_message(sub, accum)


def _decode_single_message(msg: bytes, accum: Dict[str, Any]) -> None:
    if len(msg) != ODID_MESSAGE_SIZE:
        return
    mtype = (msg[0] >> 4) & 0x0F
    if mtype == 0:
        _parse_basic_id(msg, accum)
    elif mtype == 1:
        _parse_location(msg, accum)
    elif mtype == 3:
        _parse_self_id(msg, accum)
    elif mtype == 4:
        _parse_system(msg, accum)
    elif mtype == 5:
        _parse_operator_id(msg, accum)


def _parse_basic_id(msg: bytes, accum: Dict[str, Any]) -> None:
    id_type = (msg[1] >> 4) & 0x0F
    ua_type = msg[1] & 0x0F
    uas = msg[2:22].split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    if uas:
        accum["uas_id"] = uas
    accum["id_type"] = id_type
    accum["ua_type"] = ua_type


def _parse_location(msg: bytes, accum: Dict[str, Any]) -> None:
    b1 = msg[1]
    status = (b1 >> 4) & 0x0F
    speed_mult = b1 & 0x01
    ew_dir = (b1 >> 1) & 0x01
    height_type = (b1 >> 2) & 0x01

    direction_enc = msg[2]
    speed_h_enc = msg[3]
    speed_v_enc = int.from_bytes(msg[4:5], "little", signed=True)
    lat_i = int.from_bytes(msg[5:9], "little", signed=True)
    lon_i = int.from_bytes(msg[9:13], "little", signed=True)
    alt_baro = int.from_bytes(msg[13:15], "little")
    alt_geo = int.from_bytes(msg[15:17], "little")
    height_enc = int.from_bytes(msg[17:19], "little")

    lat = lat_i / LATLON_MULT
    lon = lon_i / LATLON_MULT
    direction = float(direction_enc + (180 if ew_dir else 0))
    if speed_mult:
        speed_h = float(speed_h_enc) * SPEED_DIV1 + (255.0 * SPEED_DIV0)
    else:
        speed_h = float(speed_h_enc) * SPEED_DIV0
    speed_v = float(speed_v_enc) * VSPEED_DIV
    alt_geo_m = float(alt_geo) * ALT_DIV - ALT_ADDER
    height_m = float(height_enc) * ALT_DIV - ALT_ADDER

    accum["drone_position"] = {
        "lat": lat,
        "lon": lon,
        "altitude_m": alt_geo_m,
    }
    accum["status"] = status
    accum["heading_deg"] = direction
    accum["speed_mps"] = speed_h
    accum["speed_vertical_mps"] = speed_v
    accum["height_type"] = height_type
    accum["height_m"] = height_m
    accum["altitude_baro_m"] = float(alt_baro) * ALT_DIV - ALT_ADDER


def _parse_self_id(msg: bytes, accum: Dict[str, Any]) -> None:
    desc = msg[2:25].split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    if desc:
        accum["self_id_description"] = desc


def _parse_system(msg: bytes, accum: Dict[str, Any]) -> None:
    op_lat_i = int.from_bytes(msg[2:6], "little", signed=True)
    op_lon_i = int.from_bytes(msg[6:10], "little", signed=True)
    op_lat = op_lat_i / LATLON_MULT
    op_lon = op_lon_i / LATLON_MULT
    op_alt_geo_enc = int.from_bytes(msg[17:19], "little")
    op_alt_m = float(op_alt_geo_enc) * ALT_DIV - ALT_ADDER
    if op_lat_i != 0 or op_lon_i != 0:
        accum["takeoff_position"] = {"lat": op_lat, "lon": op_lon, "altitude_m": op_alt_m}
        accum["operator_position"] = {"lat": op_lat, "lon": op_lon, "altitude_m": op_alt_m}


def _parse_operator_id(msg: bytes, accum: Dict[str, Any]) -> None:
    op = msg[2:22].split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    if op:
        accum["operator_id"] = op


def _no_decode(reason: str, **extra: Any) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "status": "no_decode",
        "confidence": 0.0,
        "validation": {"crc_pass": False, "frame_count": 0},
        "decoded_fields": {},
        "evidence": {"reason": reason, **extra},
        "generated_at_ts": time.time(),
    }


def _decode_error(reason: str, **extra: Any) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "status": "decode_error",
        "confidence": 0.0,
        "validation": {"crc_pass": False, "frame_count": 0},
        "decoded_fields": {},
        "evidence": {"reason": reason, **extra},
        "generated_at_ts": time.time(),
    }
