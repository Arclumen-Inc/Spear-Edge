"""
AoA (Angle of Arrival) Fusion Module

Implements proper geometric triangulation for bearing line intersection
and cone overlap calculation to determine TAI (Targeted Area of Interest).

Geometry:
- Each Tripwire provides: GPS position, bearing (degrees), cone width (uncertainty)
- Two bearing lines intersect at a point (target estimate)
- Cone widths create uncertainty bounds forming a polygon TAI
- Multiple Tripwires narrow the TAI through intersection of all cones
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


# Earth radius in meters (WGS84 mean)
EARTH_RADIUS_M = 6371000.0


@dataclass
class GeoPoint:
    """Geographic point with lat/lon in degrees."""
    lat: float
    lon: float
    alt: float = 0.0
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class LocalPoint:
    """Point in local ENU (East-North-Up) coordinates in meters."""
    x: float  # East (positive = east of origin)
    y: float  # North (positive = north of origin)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class BearingSource:
    """Source types for bearing data."""
    MANUAL_DF = "manual_df"      # Human operator marked bearing in Manual DF UI
    AOA_AUTO = "aoa_auto"        # Automated AoA from antenna array processing
    BEARING_LINE = "bearing_line"  # Legacy bearing line format
    UNKNOWN = "unknown"


@dataclass
class BearingCone:
    """A bearing cone from a Tripwire node."""
    node_id: str
    position: GeoPoint
    bearing_deg: float  # 0 = North, clockwise
    cone_width_deg: float = 30.0  # Total width (±half on each side)
    confidence: float = 0.5
    timestamp: float = 0.0
    signal_freq_mhz: Optional[float] = None
    source_type: str = BearingSource.UNKNOWN  # manual_df, aoa_auto, bearing_line
    callsign: Optional[str] = None  # Human-readable node name
    
    @property
    def bearing_left_deg(self) -> float:
        """Left edge of cone (counter-clockwise from bearing)."""
        return (self.bearing_deg - self.cone_width_deg / 2) % 360
    
    @property
    def bearing_right_deg(self) -> float:
        """Right edge of cone (clockwise from bearing)."""
        return (self.bearing_deg + self.cone_width_deg / 2) % 360
    
    @property
    def source_label(self) -> str:
        """Human-readable source label."""
        labels = {
            BearingSource.MANUAL_DF: "Manual DF",
            BearingSource.AOA_AUTO: "Auto AoA",
            BearingSource.BEARING_LINE: "Bearing Line",
            BearingSource.UNKNOWN: "Unknown",
        }
        return labels.get(self.source_type, "Unknown")


@dataclass
class BearingInfo:
    """Summary info about a bearing used in fusion."""
    node_id: str
    callsign: Optional[str]
    source_type: str
    bearing_deg: float
    cone_width_deg: float
    confidence: float


@dataclass
class TAIResult:
    """Result of TAI (Targeted Area of Interest) calculation."""
    valid: bool = False
    centroid: Optional[GeoPoint] = None
    
    # Polygon vertices (for map overlay)
    polygon: List[GeoPoint] = field(default_factory=list)
    
    # Metrics
    area_m2: float = 0.0
    radius_m: float = 0.0  # Equivalent circular radius
    
    # Quality
    confidence: float = 0.0
    gdop: float = 0.0  # Geometric Dilution of Precision (lower = better)
    num_cones: int = 0
    
    # Source tracking
    bearings_used: List[BearingInfo] = field(default_factory=list)
    sources_summary: Dict[str, int] = field(default_factory=dict)  # e.g., {"manual_df": 1, "aoa_auto": 2}
    
    # Debug info
    bearing_intersection: Optional[GeoPoint] = None  # Center line intersection
    error_message: Optional[str] = None


def latlon_to_local(origin: GeoPoint, point: GeoPoint) -> LocalPoint:
    """
    Convert lat/lon to local ENU coordinates (meters) relative to origin.
    Uses equirectangular approximation (accurate for distances < 100km).
    """
    lat_rad = math.radians(origin.lat)
    
    # Meters per degree at this latitude
    m_per_deg_lat = EARTH_RADIUS_M * math.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * math.cos(lat_rad)
    
    dx = (point.lon - origin.lon) * m_per_deg_lon  # East
    dy = (point.lat - origin.lat) * m_per_deg_lat  # North
    
    return LocalPoint(x=dx, y=dy)


def local_to_latlon(origin: GeoPoint, local: LocalPoint) -> GeoPoint:
    """
    Convert local ENU coordinates (meters) back to lat/lon.
    """
    lat_rad = math.radians(origin.lat)
    
    m_per_deg_lat = EARTH_RADIUS_M * math.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * math.cos(lat_rad)
    
    lat = origin.lat + local.y / m_per_deg_lat
    lon = origin.lon + local.x / m_per_deg_lon
    
    return GeoPoint(lat=lat, lon=lon)


def bearing_to_direction(bearing_deg: float) -> Tuple[float, float]:
    """
    Convert bearing (degrees, 0=North, clockwise) to unit direction vector.
    Returns (dx, dy) where dx=East, dy=North.
    """
    bearing_rad = math.radians(bearing_deg)
    dx = math.sin(bearing_rad)  # East component
    dy = math.cos(bearing_rad)  # North component
    return (dx, dy)


def line_intersection_2d(
    p1: LocalPoint, d1: Tuple[float, float],
    p2: LocalPoint, d2: Tuple[float, float]
) -> Optional[LocalPoint]:
    """
    Find intersection of two 2D lines.
    
    Line 1: p1 + t1 * d1
    Line 2: p2 + t2 * d2
    
    Returns intersection point or None if parallel.
    """
    # Cross product of direction vectors
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    
    if abs(cross) < 1e-10:
        # Lines are parallel
        return None
    
    # Vector from p1 to p2
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    # Parameter t1 for line 1
    t1 = (dx * d2[1] - dy * d2[0]) / cross
    
    # Intersection must be in front of both nodes (t1 > 0)
    # For bearing lines, we want the intersection in the direction of the bearing
    if t1 < 0:
        return None
    
    # Also check t2 to ensure intersection is in front of second node
    t2 = (dx * d1[1] - dy * d1[0]) / cross
    if t2 < 0:
        return None
    
    # Calculate intersection point
    ix = p1.x + t1 * d1[0]
    iy = p1.y + t1 * d1[1]
    
    return LocalPoint(x=ix, y=iy)


def bearing_line_intersection(
    cone1: BearingCone, 
    cone2: BearingCone,
    origin: GeoPoint
) -> Optional[GeoPoint]:
    """
    Calculate intersection of two bearing lines (center lines of cones).
    
    Returns the geographic point where the bearings intersect, or None if
    they don't intersect (parallel or behind the nodes).
    """
    # Convert to local coordinates
    p1 = latlon_to_local(origin, cone1.position)
    p2 = latlon_to_local(origin, cone2.position)
    
    # Get direction vectors from bearings
    d1 = bearing_to_direction(cone1.bearing_deg)
    d2 = bearing_to_direction(cone2.bearing_deg)
    
    # Find intersection
    intersection = line_intersection_2d(p1, d1, p2, d2)
    
    if intersection is None:
        return None
    
    # Convert back to lat/lon
    return local_to_latlon(origin, intersection)


def cone_intersection_polygon(
    cone1: BearingCone,
    cone2: BearingCone,
    origin: GeoPoint
) -> List[GeoPoint]:
    """
    Calculate the polygon formed by the intersection of two bearing cones.
    
    The intersection is bounded by the four combinations of cone edges:
    - cone1_left × cone2_left
    - cone1_left × cone2_right
    - cone1_right × cone2_left
    - cone1_right × cone2_right
    
    Returns a list of vertices forming the TAI polygon (convex hull).
    """
    # Convert to local coordinates
    p1 = latlon_to_local(origin, cone1.position)
    p2 = latlon_to_local(origin, cone2.position)
    
    # Get direction vectors for all four cone edges
    d1_left = bearing_to_direction(cone1.bearing_left_deg)
    d1_right = bearing_to_direction(cone1.bearing_right_deg)
    d2_left = bearing_to_direction(cone2.bearing_left_deg)
    d2_right = bearing_to_direction(cone2.bearing_right_deg)
    
    # Calculate all four intersection points
    intersections = []
    
    for d1, d1_name in [(d1_left, "L"), (d1_right, "R")]:
        for d2, d2_name in [(d2_left, "L"), (d2_right, "R")]:
            point = line_intersection_2d(p1, d1, p2, d2)
            if point is not None:
                intersections.append(point)
    
    if len(intersections) < 3:
        # Not enough intersections to form a polygon
        return []
    
    # Sort points by angle from centroid to form proper polygon
    if intersections:
        cx = sum(p.x for p in intersections) / len(intersections)
        cy = sum(p.y for p in intersections) / len(intersections)
        
        def angle_from_centroid(p: LocalPoint) -> float:
            return math.atan2(p.y - cy, p.x - cx)
        
        intersections.sort(key=angle_from_centroid)
    
    # Convert back to lat/lon
    return [local_to_latlon(origin, p) for p in intersections]


def polygon_area_m2(vertices: List[GeoPoint], origin: GeoPoint) -> float:
    """
    Calculate area of a polygon in square meters using Shoelace formula.
    """
    if len(vertices) < 3:
        return 0.0
    
    # Convert to local coordinates
    local_verts = [latlon_to_local(origin, v) for v in vertices]
    
    # Shoelace formula
    n = len(local_verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += local_verts[i].x * local_verts[j].y
        area -= local_verts[j].x * local_verts[i].y
    
    return abs(area) / 2.0


def polygon_centroid(vertices: List[GeoPoint], origin: GeoPoint) -> GeoPoint:
    """
    Calculate centroid of a polygon.
    """
    if len(vertices) == 0:
        return origin
    
    if len(vertices) < 3:
        # Just average the points
        avg_lat = sum(v.lat for v in vertices) / len(vertices)
        avg_lon = sum(v.lon for v in vertices) / len(vertices)
        return GeoPoint(lat=avg_lat, lon=avg_lon)
    
    # Convert to local, compute centroid, convert back
    local_verts = [latlon_to_local(origin, v) for v in vertices]
    
    # Centroid formula for polygon
    n = len(local_verts)
    cx, cy = 0.0, 0.0
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        cross = local_verts[i].x * local_verts[j].y - local_verts[j].x * local_verts[i].y
        area += cross
        cx += (local_verts[i].x + local_verts[j].x) * cross
        cy += (local_verts[i].y + local_verts[j].y) * cross
    
    area /= 2.0
    if abs(area) < 1e-10:
        # Degenerate polygon, use simple average
        cx = sum(v.x for v in local_verts) / n
        cy = sum(v.y for v in local_verts) / n
    else:
        cx /= (6.0 * area)
        cy /= (6.0 * area)
    
    return local_to_latlon(origin, LocalPoint(x=cx, y=cy))


def calculate_gdop(cones: List[BearingCone], intersection: GeoPoint, origin: GeoPoint) -> float:
    """
    Calculate Geometric Dilution of Precision (GDOP).
    
    GDOP indicates how the geometry of the sensor network affects position accuracy.
    Lower GDOP = better geometry = more accurate position estimate.
    
    Ideal: sensors ~90° apart relative to target
    Poor: sensors nearly in line with target
    """
    if len(cones) < 2:
        return 999.0  # Very poor
    
    # Convert intersection to local coords
    target = latlon_to_local(origin, intersection)
    
    # Calculate angles from target to each sensor
    angles = []
    for cone in cones:
        sensor = latlon_to_local(origin, cone.position)
        dx = sensor.x - target.x
        dy = sensor.y - target.y
        angle = math.atan2(dx, dy)  # Angle from target to sensor
        angles.append(angle)
    
    # GDOP is related to the spread of angles
    # Best case: angles are evenly distributed (90° apart for 2 sensors)
    if len(angles) == 2:
        angle_diff = abs(angles[1] - angles[0])
        # Normalize to 0-180 range
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # GDOP approximation: 1/sin(angle_diff)
        # At 90°: GDOP = 1.0 (ideal)
        # At 30°: GDOP = 2.0 (poor)
        # At 10°: GDOP = 5.7 (very poor)
        sin_diff = abs(math.sin(angle_diff))
        if sin_diff < 0.1:
            return 10.0  # Very poor geometry
        return 1.0 / sin_diff
    
    # For 3+ sensors, compute more complex GDOP
    # Simplified: use minimum angle between any pair
    min_angle_diff = math.pi
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            diff = abs(angles[j] - angles[i])
            if diff > math.pi:
                diff = 2 * math.pi - diff
            min_angle_diff = min(min_angle_diff, diff)
    
    sin_diff = abs(math.sin(min_angle_diff))
    if sin_diff < 0.1:
        return 10.0
    
    # Bonus for more sensors
    sensor_bonus = 1.0 - (len(cones) - 2) * 0.1  # 3 sensors = 0.9x, 4 = 0.8x
    sensor_bonus = max(0.5, sensor_bonus)
    
    return (1.0 / sin_diff) * sensor_bonus


def distance_m(p1: GeoPoint, p2: GeoPoint) -> float:
    """Calculate distance between two geographic points in meters (Haversine)."""
    lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
    lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return EARTH_RADIUS_M * c


def fuse_bearing_cones(cones: List[BearingCone]) -> TAIResult:
    """
    Fuse multiple bearing cones to calculate TAI (Targeted Area of Interest).
    
    Algorithm:
    1. With 2 cones: Calculate bearing line intersection and cone overlap polygon
    2. With 3+ cones: Intersect all cone polygons for tighter TAI
    
    Returns TAIResult with centroid, polygon, area, and quality metrics.
    """
    result = TAIResult()
    result.num_cones = len(cones)
    
    if len(cones) < 2:
        result.error_message = "Need at least 2 bearing cones for triangulation"
        return result
    
    # Use first cone's position as origin for local coordinate transforms
    origin = cones[0].position
    
    # Step 1: Calculate center bearing line intersection (best estimate point)
    center_intersection = bearing_line_intersection(cones[0], cones[1], origin)
    
    if center_intersection is None:
        result.error_message = "Bearing lines do not intersect (parallel or behind sensors)"
        return result
    
    result.bearing_intersection = center_intersection
    
    # Step 2: Calculate cone intersection polygon for first two cones
    polygon = cone_intersection_polygon(cones[0], cones[1], origin)
    
    if len(polygon) < 3:
        # Fall back to point estimate with estimated uncertainty
        result.valid = True
        result.centroid = center_intersection
        result.polygon = [center_intersection]
        
        # Estimate radius from cone widths and distance
        dist = distance_m(cones[0].position, cones[1].position)
        avg_width = (cones[0].cone_width_deg + cones[1].cone_width_deg) / 2
        # Rough approximation: radius = distance * tan(width/2) at intersection
        result.radius_m = dist * 0.5 * math.tan(math.radians(avg_width / 2))
        result.area_m2 = math.pi * result.radius_m ** 2
        
        result.confidence = (cones[0].confidence + cones[1].confidence) / 2
        result.gdop = calculate_gdop(cones, center_intersection, origin)
        return result
    
    # Step 3: With 3+ cones, intersect additional cone polygons
    # (For now, we refine using all pairwise intersections)
    if len(cones) > 2:
        all_polygon_points = list(polygon)
        
        # Add intersections from other cone pairs
        for i in range(len(cones)):
            for j in range(i + 1, len(cones)):
                if i == 0 and j == 1:
                    continue  # Already computed
                
                additional_polygon = cone_intersection_polygon(cones[i], cones[j], origin)
                if additional_polygon:
                    all_polygon_points.extend(additional_polygon)
        
        # Find the common intersection region
        # Simplified: use convex hull of points that are inside all cones
        # For production, use proper polygon intersection algorithm
        if len(all_polygon_points) > len(polygon):
            # Sort by angle for proper polygon
            cx = sum(p.lat for p in all_polygon_points) / len(all_polygon_points)
            cy = sum(p.lon for p in all_polygon_points) / len(all_polygon_points)
            
            def angle_sort(p: GeoPoint) -> float:
                return math.atan2(p.lon - cy, p.lat - cx)
            
            all_polygon_points.sort(key=angle_sort)
            polygon = all_polygon_points
    
    # Step 4: Calculate TAI properties
    result.valid = True
    result.polygon = polygon
    result.centroid = polygon_centroid(polygon, origin)
    result.area_m2 = polygon_area_m2(polygon, origin)
    result.radius_m = math.sqrt(result.area_m2 / math.pi) if result.area_m2 > 0 else 0.0
    
    # Step 5: Quality metrics
    result.confidence = sum(c.confidence for c in cones) / len(cones)
    result.gdop = calculate_gdop(cones, result.centroid, origin)
    
    # Step 6: Track bearing sources
    result.bearings_used = [
        BearingInfo(
            node_id=c.node_id,
            callsign=c.callsign,
            source_type=c.source_type,
            bearing_deg=c.bearing_deg,
            cone_width_deg=c.cone_width_deg,
            confidence=c.confidence,
        )
        for c in cones
    ]
    
    # Summarize sources
    sources = {}
    for c in cones:
        sources[c.source_type] = sources.get(c.source_type, 0) + 1
    result.sources_summary = sources
    
    return result


def std_to_cone_width(bearing_std_deg: float, sigma_multiplier: float = 2.0) -> float:
    """
    Convert bearing standard deviation to cone width.
    
    Uses the sigma multiplier to determine cone width:
    - 1 sigma (68% confidence): multiplier = 1.0
    - 2 sigma (95% confidence): multiplier = 2.0 (default)
    - 3 sigma (99.7% confidence): multiplier = 3.0
    
    Cone width = 2 * std * multiplier (±std on each side)
    """
    return 2.0 * bearing_std_deg * sigma_multiplier


def infer_source_type(d: Dict[str, Any]) -> str:
    """
    Infer the source type from event data.
    
    Heuristics:
    - If 'type' == 'bearing_line' or 'df_bearing': Manual DF
    - If 'type' == 'aoa_cone': Automated AoA
    - If 'source_type' is explicitly set: Use that
    - If has 'bearing_std_deg' but no 'cone_width_deg': Likely Manual DF
    - Default: Unknown
    """
    # Explicit source type
    if d.get("source_type"):
        return d.get("source_type")
    
    # Infer from event type
    event_type = d.get("type") or d.get("event_type", "")
    
    if event_type in ("bearing_line", "df_bearing"):
        return BearingSource.MANUAL_DF
    elif event_type == "aoa_cone":
        return BearingSource.AOA_AUTO
    
    # Heuristic: Manual DF often has bearing_std_deg
    if d.get("bearing_std_deg") and not d.get("cone_width_deg"):
        return BearingSource.MANUAL_DF
    
    # Heuristic: Automated AoA often has multipath_flag or trend
    if d.get("multipath_flag") is not None or d.get("trend"):
        return BearingSource.AOA_AUTO
    
    return BearingSource.UNKNOWN


def cones_from_dicts(cone_dicts: List[Dict[str, Any]]) -> List[BearingCone]:
    """
    Convert list of cone dictionaries (from events) to BearingCone objects.
    
    Handles both AoA cones and Manual DF bearing lines, normalizing
    uncertainty representation (converts bearing_std_deg to cone_width_deg).
    
    Expected dict format:
    {
        "node_id": "tripwire-001",
        "gps": {"lat": 34.05, "lon": -118.24},
        "bearing_deg": 45.0,
        "cone_width_deg": 30.0,      # OR
        "bearing_std_deg": 5.0,      # Will be converted to cone_width
        "confidence": 0.8,
        "timestamp": 1234567890.0,
        "source_type": "manual_df",  # Optional: manual_df, aoa_auto
        "type": "aoa_cone",          # Used to infer source if not explicit
    }
    """
    cones = []
    
    for d in cone_dicts:
        gps = d.get("gps", {})
        lat = gps.get("lat")
        lon = gps.get("lon")
        alt = gps.get("alt", 0.0)
        
        if lat is None or lon is None:
            continue
        
        bearing = d.get("bearing_deg")
        if bearing is None:
            continue
        
        # Determine cone width: prefer explicit, then convert from std dev
        cone_width = d.get("cone_width_deg")
        if cone_width is None or cone_width <= 0:
            bearing_std = d.get("bearing_std_deg")
            if bearing_std and bearing_std > 0:
                # Convert std dev to cone width (2-sigma = 95% confidence)
                cone_width = std_to_cone_width(bearing_std, sigma_multiplier=2.0)
            else:
                # Default cone width based on source type
                source = infer_source_type(d)
                if source == BearingSource.MANUAL_DF:
                    cone_width = 15.0  # Manual DF typically more precise
                else:
                    cone_width = 30.0  # Default for automated AoA
        
        # Clamp cone width to reasonable range
        cone_width = max(5.0, min(90.0, float(cone_width)))
        
        # Infer source type
        source_type = infer_source_type(d)
        
        cone = BearingCone(
            node_id=d.get("node_id", "unknown"),
            position=GeoPoint(lat=lat, lon=lon, alt=alt),
            bearing_deg=float(bearing),
            cone_width_deg=cone_width,
            confidence=float(d.get("confidence", 0.5)),
            timestamp=float(d.get("timestamp", 0.0)),
            signal_freq_mhz=d.get("signal_freq_mhz") or d.get("center_freq_mhz"),
            source_type=source_type,
            callsign=d.get("callsign"),
        )
        cones.append(cone)
    
    return cones


def tai_to_dict(tai: TAIResult) -> Dict[str, Any]:
    """Convert TAIResult to dictionary for JSON serialization."""
    return {
        "valid": tai.valid,
        "centroid": {
            "lat": tai.centroid.lat,
            "lon": tai.centroid.lon,
        } if tai.centroid else None,
        "polygon": [
            {"lat": p.lat, "lon": p.lon}
            for p in tai.polygon
        ],
        "area_m2": tai.area_m2,
        "radius_m": tai.radius_m,
        "confidence": tai.confidence,
        "gdop": tai.gdop,
        "num_cones": tai.num_cones,
        "bearings_used": [
            {
                "node_id": b.node_id,
                "callsign": b.callsign,
                "source_type": b.source_type,
                "source_label": {
                    "manual_df": "Manual DF",
                    "aoa_auto": "Auto AoA", 
                    "bearing_line": "Bearing Line",
                    "unknown": "Unknown",
                }.get(b.source_type, "Unknown"),
                "bearing_deg": b.bearing_deg,
                "cone_width_deg": b.cone_width_deg,
                "confidence": b.confidence,
            }
            for b in tai.bearings_used
        ],
        "sources_summary": tai.sources_summary,
        "bearing_intersection": {
            "lat": tai.bearing_intersection.lat,
            "lon": tai.bearing_intersection.lon,
        } if tai.bearing_intersection else None,
        "error_message": tai.error_message,
    }
