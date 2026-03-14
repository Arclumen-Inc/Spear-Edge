# Cursor on Target (CoT) and ATAK Reference Guide

This document provides a comprehensive reference for integrating with ATAK (Android Tactical Awareness Kit) version 5.2+ using the Cursor on Target (CoT) protocol.

## Overview

Cursor on Target (CoT) is an XML-based messaging format originally designed at MITRE Corp. in 2005 for exchanging tactical "who/what/where" information between military systems. ATAK uses CoT for all tactical data exchange.

## Transport Methods

### Multicast UDP (SA - Situational Awareness)

Standard multicast addresses for ATAK:

| Purpose | Address | Port |
|---------|---------|------|
| Position/Events | `239.2.3.1` | `6969` |
| Chat (All Chat) | `224.10.10.1` | `17012` |

### TAK Server

For networked deployments, CoT messages are sent via TCP/TLS to a TAK Server which relays to connected clients.

### Protocol Versions

- **XML CoT**: Traditional XML format (wide compatibility)
- **TAK Protocol v1 (Protobuf)**: Binary format for efficiency
  - Mesh format: Static header `191 1 191`
  - Stream format: Dynamic header `191`

---

## CoT Event Structure

### Basic XML Format

```xml
<?xml version="1.0" encoding="UTF-8"?>
<event version="2.0" 
       uid="unique-identifier" 
       type="a-f-G-U-C" 
       how="m-g" 
       time="2024-01-15T12:30:00.000Z" 
       start="2024-01-15T12:30:00.000Z" 
       stale="2024-01-15T12:35:00.000Z">
    <point lat="34.0522" lon="-118.2437" hae="100.0" ce="10.0" le="10.0"/>
    <detail>
        <contact callsign="UNIT-ALPHA"/>
        <remarks>Description text here</remarks>
    </detail>
</event>
```

### Event Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `version` | CoT version (always "2.0") | `version="2.0"` |
| `uid` | Globally unique identifier | `uid="SPEAR-EDGE-001"` |
| `type` | Hierarchical type code | `type="a-f-G-U-C"` |
| `how` | How coordinates were generated | `how="m-g"` |
| `time` | When event was generated (ISO 8601) | `time="2024-01-15T12:30:00Z"` |
| `start` | When event becomes relevant | `start="2024-01-15T12:30:00Z"` |
| `stale` | When event expires | `stale="2024-01-15T12:35:00Z"` |

### Point Element

| Attribute | Description | Unit |
|-----------|-------------|------|
| `lat` | Latitude | Decimal degrees |
| `lon` | Longitude | Decimal degrees |
| `hae` | Height above ellipsoid | Meters |
| `ce` | Circular error (horizontal uncertainty) | Meters |
| `le` | Linear error (vertical uncertainty) | Meters |

### How Codes

| Code | Description |
|------|-------------|
| `h-e` | Human entered |
| `h-g-i-g-o` | Human generated (chat) |
| `m-g` | Machine generated |
| `m-p` | Machine predicted |
| `m-f` | Machine fused |
| `m-s` | Machine simulated |

---

## CoT Type System

The `type` attribute uses a hierarchical, hyphen-delimited format:

```
[atoms]-[affiliation]-[battle dimension]-[function code]
```

### Affiliation Codes (Level 2)

| Code | Meaning | Color |
|------|---------|-------|
| `f` | Friendly | Blue |
| `h` | Hostile | Red |
| `u` | Unknown | Yellow |
| `n` | Neutral | Green |
| `s` | Suspect | Yellow |
| `a` | Assumed Friendly | Blue |
| `p` | Pending | Yellow |

### Battle Dimension (Level 3)

| Code | Meaning |
|------|---------|
| `A` | Air |
| `G` | Ground |
| `S` | Sea Surface |
| `U` | Subsurface |
| `P` | Space |
| `F` | Special Operations |

---

## Common CoT Types

### Entity Types (Atoms)

| Type | Description | Use Case |
|------|-------------|----------|
| `a-f-G-U-C` | Friendly ground unit (combat) | Friendly units |
| `a-f-G-E-S` | Friendly ground equipment sensor | Sensors/Edge devices |
| `a-f-G-E-S-E` | Friendly ground equipment sensor - electronic | RF sensors |
| `a-h-G` | Hostile ground | Hostile units |
| `a-u-G` | Unknown ground | Unknown contacts |
| `a-u-E-U` | Unknown entity | Detections |
| `a-u-A-E-I` | Unknown air electronic intelligence | RF detections |

### Geometry Types (Drawings)

| Type | Description | Use Case |
|------|-------------|----------|
| `u-d-p` | Point | Single location marker |
| `u-d-c-c` | Circle | TAI, uncertainty areas |
| `u-d-f` | Freeform (polygon/line) | TAI polygon, boundaries |
| `u-d-r` | Rectangle | Rectangular areas |
| `u-rb-a` | Range/Bearing line | Bearing lines from sensors |

### Chat Types

| Type | Description |
|------|-------------|
| `b-t-f` | Chat message |
| `b-t-f-d` | Chat delivery receipt |
| `b-t-f-r` | Chat read receipt |
| `b-t-f-p` | Chat pending receipt |
| `b-t-f-s` | Chat delivery failure |

### Alert Types

| Type | Description | Use Case |
|------|-------------|----------|
| `b-a` | Generic alert | General notifications |
| `b-a-o-tbl` | 911 Alert | Emergency |
| `b-a-o-can` | Cancel alert | Clear previous alert |
| `b-a-g` | GeoFence breach | Area violation |
| `b-a-o-pan` | Ring the Bell | Urgent attention needed |
| `b-a-o-opn` | Troops in Contact | Combat alert |
| `b-r-f-h-c` | CASEVAC | Medical evacuation |

### Special Types

| Type | Description |
|------|-------------|
| `b-m-r` | Route |
| `b-m-p-c` | Route control point |
| `b-m-p-w` | Route waypoint |
| `b-i-v` | Video/imagery |
| `b-i-x-i` | QuickPic |

---

## Drawing Shapes

### Circle (`u-d-c-c`)

Draw a circle on the map with specified radius:

```xml
<event version="2.0" uid="circle-001" type="u-d-c-c" how="h-e"
       time="2024-01-15T12:30:00Z" start="2024-01-15T12:30:00Z" 
       stale="2024-01-15T12:35:00Z">
    <point lat="34.0522" lon="-118.2437" hae="0" ce="9999999" le="9999999"/>
    <detail>
        <shape>
            <ellipse major="500" minor="500" angle="0"/>
        </shape>
        <strokeColor value="-65536"/>
        <strokeWeight value="3.0"/>
        <fillColor value="536870912"/>
        <contact callsign="TAI Area"/>
        <remarks>Targeted Area of Interest - 500m radius</remarks>
        <labels_on value="true"/>
    </detail>
</event>
```

**Ellipse attributes:**
- `major`: Semi-major axis in meters
- `minor`: Semi-minor axis in meters (same as major for circle)
- `angle`: Rotation angle in degrees (0 for circle)

### Ellipse (`u-d-c-c`)

For uncertainty ellipses with different axes:

```xml
<shape>
    <ellipse major="1000" minor="500" angle="45"/>
</shape>
```

### Polygon (`u-d-f`)

Draw a closed polygon using linked points:

```xml
<event version="2.0" uid="polygon-001" type="u-d-f" how="h-e"
       time="2024-01-15T12:30:00Z" start="2024-01-15T12:30:00Z"
       stale="2024-01-15T12:35:00Z">
    <point lat="34.0522" lon="-118.2437" hae="0" ce="9999999" le="9999999"/>
    <detail>
        <link uid="polygon-001" type="b-m-p-w" point="34.051,-118.242" relation="p-p"/>
        <link uid="polygon-001" type="b-m-p-w" point="34.053,-118.241" relation="p-p"/>
        <link uid="polygon-001" type="b-m-p-w" point="34.054,-118.244" relation="p-p"/>
        <link uid="polygon-001" type="b-m-p-w" point="34.052,-118.246" relation="p-p"/>
        <strokeColor value="-16711936"/>
        <strokeWeight value="2.0"/>
        <fillColor value="1342177280"/>
        <closed value="true"/>
        <contact callsign="TAI Polygon"/>
        <remarks>Targeted Area of Interest polygon</remarks>
    </detail>
</event>
```

### Line/Polyline (`u-d-f`)

Same as polygon but without `<closed value="true"/>`:

```xml
<detail>
    <link uid="line-001" type="b-m-p-w" point="34.051,-118.242" relation="p-p"/>
    <link uid="line-001" type="b-m-p-w" point="34.053,-118.241" relation="p-p"/>
    <strokeColor value="-16776961"/>
    <strokeWeight value="2.0"/>
</detail>
```

### Range/Bearing Line (`u-rb-a`)

Draw a bearing line from a position:

```xml
<event version="2.0" uid="bearing-001" type="u-rb-a" how="h-e"
       time="2024-01-15T12:30:00Z" start="2024-01-15T12:30:00Z"
       stale="2024-01-15T12:35:00Z">
    <point lat="34.0522" lon="-118.2437" hae="0" ce="9999999" le="9999999"/>
    <detail>
        <range value="5000"/>
        <bearing value="45.0"/>
        <inclination value="0"/>
        <strokeColor value="-16711681"/>
        <strokeWeight value="2.0"/>
        <contact callsign="TW-001 Bearing"/>
        <remarks>Bearing: 045° from TW-001</remarks>
    </detail>
</event>
```

---

## Color Values (ARGB)

Colors are specified as signed 32-bit integers in ARGB format:

### Common Colors

| Color | ARGB Value | Hex |
|-------|------------|-----|
| Red (opaque) | `-65536` | `#FFFF0000` |
| Green (opaque) | `-16711936` | `#FF00FF00` |
| Blue (opaque) | `-16776961` | `#FF0000FF` |
| Yellow (opaque) | `-256` | `#FFFFFF00` |
| Orange (opaque) | `-32768` | `#FFFF8000` |
| White (opaque) | `-1` | `#FFFFFFFF` |
| Black (opaque) | `-16777216` | `#FF000000` |

### Semi-Transparent Colors

| Color | ARGB Value | Hex |
|-------|------------|-----|
| Red (50%) | `2147418112` | `#80FF0000` |
| Green (50%) | `2130771712` | `#8000FF00` |
| Blue (50%) | `2130706687` | `#800000FF` |
| Yellow (50%) | `2147483392` | `#80FFFF00` |
| White (30%) | `1342177280` | `#50FFFFFF` |

### Converting Colors

**Hex to ARGB Integer:**
```python
def hex_to_argb(hex_color):
    # hex_color like "#80FF0000" (ARGB)
    value = int(hex_color[1:], 16)
    if value >= 0x80000000:
        value -= 0x100000000  # Convert to signed
    return value

# Example: "#FFFF0000" (red) -> -65536
```

**RGB to ARGB Integer:**
```python
def rgb_to_argb(r, g, b, a=255):
    value = (a << 24) | (r << 16) | (g << 8) | b
    if value >= 0x80000000:
        value -= 0x100000000
    return value

# Example: rgb_to_argb(255, 0, 0, 255) -> -65536 (red)
# Example: rgb_to_argb(255, 0, 0, 128) -> 2147418112 (semi-transparent red)
```

---

## Chat Messages

### GeoChat Format (`b-t-f`)

```xml
<event version="2.0" uid="Chat-abc123" type="b-t-f" how="h-g-i-g-o"
       time="2024-01-15T12:30:00Z" start="2024-01-15T12:30:00Z"
       stale="2024-01-15T12:35:00Z">
    <point lat="0" lon="0" hae="0" ce="9999999" le="9999999"/>
    <detail>
        <__chat id="All Chat" chatroom="All Chat" 
                senderCallsign="SPEAR-EDGE" 
                groupOwner="false">
            <chatgrp uid0="SPEAR-EDGE" uid1="All Chat" id="All Chat"/>
        </__chat>
        <link uid="SPEAR-EDGE" type="a-f-G-E-S-E" relation="p-p"/>
        <remarks source="SPEAR-EDGE" time="2024-01-15T12:30:00Z">
            Message text goes here
        </remarks>
    </detail>
</event>
```

### Simple Chat (Minimal)

```xml
<event version="2.0" uid="Chat-abc123" type="b-t-f" how="h-g-i-g-o"
       time="2024-01-15T12:30:00Z" start="2024-01-15T12:30:00Z"
       stale="2024-01-15T12:35:00Z">
    <point lat="0" lon="0" hae="0" ce="9999999" le="9999999"/>
    <detail>
        <__chat id="All Chat" chatroom="All Chat" 
                senderCallsign="SPEAR-EDGE" message="Detection alert!"/>
        <remarks>Detection alert!</remarks>
    </detail>
</event>
```

---

## Alerts

### Generic Alert (`b-a`)

```xml
<event version="2.0" uid="alert-001" type="b-a" how="h-e"
       time="2024-01-15T12:30:00Z" start="2024-01-15T12:30:00Z"
       stale="2024-01-15T12:35:00Z">
    <point lat="34.0522" lon="-118.2437" hae="0" ce="50" le="50"/>
    <detail>
        <contact callsign="RF ALERT"/>
        <remarks>High-confidence RF detection at 915 MHz</remarks>
        <color argb="-65536"/>
    </detail>
</event>
```

### Ring the Bell (`b-a-o-pan`)

Urgent attention alert:

```xml
<event version="2.0" uid="urgent-001" type="b-a-o-pan" how="h-e"
       time="2024-01-15T12:30:00Z" start="2024-01-15T12:30:00Z"
       stale="2024-01-15T12:35:00Z">
    <point lat="34.0522" lon="-118.2437" hae="0" ce="50" le="50"/>
    <detail>
        <contact callsign="URGENT"/>
        <remarks>CONFIRMED THREAT DETECTED</remarks>
    </detail>
</event>
```

---

## Best Practices

### UID Management

- Use stable UIDs for items that should update in place (e.g., `SPEAR-EDGE-tai`)
- Use random/timestamped UIDs for transient events (e.g., `Chat-{uuid}`)
- Include source identifier in UID for traceability

### Stale Times

| Event Type | Recommended Stale Time |
|------------|------------------------|
| Position updates | 30 seconds |
| Chat messages | 5 minutes |
| TAI/Detection markers | 1-2 minutes |
| Alerts | 5 minutes |
| Persistent drawings | 24 hours |

### Performance

- Send position updates every 5-10 seconds
- Batch multiple events when possible
- Use multicast for local networks, TAK Server for remote
- Consider Protobuf format for bandwidth-constrained links

### Coordinate Precision

- Use 6 decimal places for lat/lon (±0.1m precision)
- Always include `ce` and `le` for uncertainty
- Use `hae` (height above ellipsoid), not MSL

---

## Python Implementation Examples

### Sending a Circle

```python
def build_circle_cot(uid, lat, lon, radius_m, color_argb, callsign, remarks, stale_s=120):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + stale_s))
    
    return f'''<event version="2.0" uid="{uid}" type="u-d-c-c" how="h-e"
       time="{now}" start="{now}" stale="{stale}">
    <point lat="{lat}" lon="{lon}" hae="0" ce="9999999" le="9999999"/>
    <detail>
        <shape>
            <ellipse major="{radius_m}" minor="{radius_m}" angle="0"/>
        </shape>
        <strokeColor value="{color_argb}"/>
        <strokeWeight value="3.0"/>
        <fillColor value="{color_argb // 2}"/>
        <contact callsign="{callsign}"/>
        <remarks>{remarks}</remarks>
    </detail>
</event>'''
```

### Color Helper

```python
def rgb_to_argb_int(r, g, b, a=255):
    """Convert RGBA to signed ARGB integer for CoT."""
    value = (a << 24) | (r << 16) | (g << 8) | b
    if value >= 0x80000000:
        value -= 0x100000000
    return int(value)

# Predefined colors
COT_RED = rgb_to_argb_int(255, 0, 0, 255)        # -65536
COT_GREEN = rgb_to_argb_int(0, 255, 0, 255)      # -16711936
COT_BLUE = rgb_to_argb_int(0, 0, 255, 255)       # -16776961
COT_YELLOW = rgb_to_argb_int(255, 255, 0, 255)   # -256
COT_ORANGE = rgb_to_argb_int(255, 128, 0, 255)   # -32768

# Semi-transparent versions (50% alpha)
COT_RED_50 = rgb_to_argb_int(255, 0, 0, 128)     # 2147418112
COT_GREEN_50 = rgb_to_argb_int(0, 255, 0, 128)   # 2130771712
```

---

## References

- [node-cot Documentation](https://node-cot.cloudtak.io/)
- [TAK Product Center](https://tak.gov/)
- [CivTAK Community](https://www.civtak.org/)
- [MIL-STD-2525](https://www.jcs.mil/Portals/36/Documents/Doctrine/Other_Pubs/ms_2525d.pdf)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial document |
| 1.1 | 2026-03 | Added ATAK 5.2+ specifics, shape drawing examples |
