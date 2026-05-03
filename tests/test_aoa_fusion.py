#!/usr/bin/env python3
"""
Test script for AoA Fusion visualization.

Simulates 3 Tripwire nodes sending AoA cone events that converge to a target location.
This allows testing the AoA fusion panel without needing physical Tripwire devices.

Usage:
    python test_aoa_fusion.py [--host HOST] [--port PORT] [--interval SECONDS]

Example:
    python test_aoa_fusion.py --host localhost --port 8080 --interval 2
"""

import argparse
import asyncio
import json
import math
import time
from typing import Dict, Any, Tuple
import websockets
import requests


# Default Edge server settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
DEFAULT_INTERVAL = 2.0  # seconds between updates

# Simulated node configurations
# Nodes are positioned on an east-west line (horizontal)
# RF source is to the north (above the nodes)
NODES = [
    {
        "node_id": "tripwire-01",
        "callsign": "TW-01",
        "gps": {"lat": 37.7749, "lon": -122.4244, "alt": 10.0},  # West node
    },
    {
        "node_id": "tripwire-02",
        "callsign": "TW-02",
        "gps": {"lat": 37.7749, "lon": -122.4144, "alt": 12.0},  # Center node (same latitude)
    },
    {
        "node_id": "tripwire-03",
        "callsign": "TW-03",
        "gps": {"lat": 37.7749, "lon": -122.4044, "alt": 10.0},  # East node
    },
]

# Target location (where the transmission is coming from) - North of the nodes
TARGET_LAT = 37.7849  # ~1km north of nodes
TARGET_LON = -122.4144  # Same longitude as center node


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing from point 1 to point 2 in degrees (0-360).
    Returns bearing in degrees, where 0 is North, 90 is East, etc.
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360
    
    return bearing_deg


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS points in kilometers."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def create_aoa_cone_event(node: Dict[str, Any], bearing: float, target_distance_km: float) -> Dict[str, Any]:
    """
    Create an AoA cone event for a node.
    
    Args:
        node: Node configuration dict
        bearing: Bearing to target in degrees
        target_distance_km: Distance to target in km (affects confidence)
    
    Returns:
        AoA cone event dict
    """
    # Simulate confidence based on distance (closer = higher confidence)
    base_confidence = max(0.5, 1.0 - (target_distance_km / 10.0))
    confidence = base_confidence + (math.sin(time.time()) * 0.1)  # Add some variation
    
    # Cone width varies with confidence (better confidence = narrower cone)
    cone_width = 45.0 - (confidence * 20.0)  # Range: 25-45 degrees
    
    # Add small random variation to bearing (±2 degrees)
    bearing_variation = (math.sin(time.time() * 0.5) * 2.0)
    bearing_with_variation = bearing + bearing_variation
    
    # Simulate signal strength (stronger when closer)
    strength_db = -40.0 - (target_distance_km * 2.0)
    delta_db = 10.0 + (confidence * 5.0)
    noise_db = -60.0
    
    event = {
        "type": "aoa_cone",
        "event_type": "aoa_cone",
        "node_id": node["node_id"],
        "callsign": node["callsign"],
        "bearing_deg": bearing_with_variation,
        "cone_width_deg": cone_width,
        "confidence": min(1.0, max(0.0, confidence)),
        "trend": "stable",  # or "closing", "opening"
        "timestamp": time.time(),
        "multipath_flag": False,
        "strength_db": strength_db,
        "delta_db": delta_db,
        "noise_db": noise_db,
        "center_freq_mhz": 915.0,  # Sub-GHz frequency
        "status": "active",
        "squelch_passed": True,
        "bearing_history": [
            {
                "bearing": bearing_with_variation,
                "cone_width": cone_width,
                "confidence": confidence,
                "timestamp": time.time()
            }
        ]
    }
    
    return event


async def register_node_websocket(node: Dict[str, Any], host: str, port: int) -> None:
    """
    Register a node via WebSocket hello message.
    This makes the node appear in the Tripwire Nodes panel.
    """
    uri = f"ws://{host}:{port}/ws/tripwire_link"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send hello message
            hello_msg = {
                "type": "hello",
                "node_id": node["node_id"],
                "callsign": node["callsign"],
                "gps": node["gps"],
                "meta": {
                    "sdr_driver": "bladeRF",
                    "version": "2.0"
                },
                "system_time": time.time()
            }
            
            await websocket.send(json.dumps(hello_msg))
            print(f"[{node['node_id']}] Registered via WebSocket")
            
            # Keep connection alive with periodic heartbeats
            while True:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                heartbeat = {
                    "type": "heartbeat",
                    "node_id": node["node_id"],
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(heartbeat))
                
    except Exception as e:
        print(f"[{node['node_id']}] WebSocket error: {e}")


def send_aoa_event(node: Dict[str, Any], event: Dict[str, Any], host: str, port: int) -> bool:
    """
    Send AoA cone event via HTTP POST.
    
    Returns:
        True if successful, False otherwise
    """
    url = f"http://{host}:{port}/api/tripwire/event"
    
    try:
        response = requests.post(url, json=event, timeout=2.0)
        if response.status_code == 200:
            return True
        else:
            print(f"[{node['node_id']}] HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"[{node['node_id']}] Error sending event: {e}")
        return False


async def simulate_node(node: Dict[str, Any], host: str, port: int, interval: float) -> None:
    """
    Simulate a single node sending AoA cone events.
    
    Args:
        node: Node configuration
        host: Edge server host
        port: Edge server port
        interval: Seconds between events
    """
    node_gps = node["gps"]
    
    print(f"[{node['node_id']}] Starting simulation...")
    print(f"[{node['node_id']}] Node GPS: {node_gps['lat']:.6f}, {node_gps['lon']:.6f}")
    
    while True:
        # Calculate bearing to target
        bearing = calculate_bearing(
            node_gps["lat"], node_gps["lon"],
            TARGET_LAT, TARGET_LON
        )
        
        # Calculate distance to target
        distance_km = calculate_distance_km(
            node_gps["lat"], node_gps["lon"],
            TARGET_LAT, TARGET_LON
        )
        
        # Create AoA cone event
        event = create_aoa_cone_event(node, bearing, distance_km)
        
        # Send event
        success = send_aoa_event(node, event, host, port)
        
        if success:
            print(f"[{node['node_id']}] Sent AoA cone: bearing={bearing:.1f}°, "
                  f"width={event['cone_width_deg']:.1f}°, "
                  f"conf={event['confidence']:.2f}, "
                  f"distance={distance_km:.2f}km")
        else:
            print(f"[{node['node_id']}] Failed to send event")
        
        await asyncio.sleep(interval)


async def main():
    parser = argparse.ArgumentParser(description="Simulate AoA cone events from 3 Tripwire nodes")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Edge server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Edge server port (default: {DEFAULT_PORT})")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, 
                       help=f"Interval between events in seconds (default: {DEFAULT_INTERVAL})")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AoA Fusion Test Script")
    print("=" * 70)
    print(f"Edge Server: http://{args.host}:{args.port}")
    print(f"Update Interval: {args.interval} seconds")
    print(f"Target Location: {TARGET_LAT:.6f}, {TARGET_LON:.6f}")
    print()
    print("Simulating 3 Tripwire nodes:")
    for node in NODES:
        gps = node["gps"]
        distance = calculate_distance_km(gps["lat"], gps["lon"], TARGET_LAT, TARGET_LON)
        bearing = calculate_bearing(gps["lat"], gps["lon"], TARGET_LAT, TARGET_LON)
        print(f"  {node['node_id']} ({node['callsign']}): "
              f"GPS={gps['lat']:.6f},{gps['lon']:.6f}, "
              f"Distance={distance:.2f}km, Bearing={bearing:.1f}°")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    # Start WebSocket registration tasks (one per node)
    ws_tasks = []
    for node in NODES:
        task = asyncio.create_task(register_node_websocket(node, args.host, args.port))
        ws_tasks.append(task)
        await asyncio.sleep(0.5)  # Stagger connections
    
    # Start simulation tasks (one per node)
    sim_tasks = []
    for node in NODES:
        task = asyncio.create_task(simulate_node(node, args.host, args.port, args.interval))
        sim_tasks.append(task)
        await asyncio.sleep(0.1)  # Stagger start times
    
    try:
        # Wait for all tasks
        await asyncio.gather(*ws_tasks, *sim_tasks)
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
        for task in ws_tasks + sim_tasks:
            task.cancel()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
