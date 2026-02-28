# AoA Fusion Test Script

This script simulates 3 Tripwire nodes sending AoA cone events to test the AoA fusion visualization panel without needing physical hardware.

## Features

- Simulates 3 Tripwire nodes positioned in a triangle around a target location
- Sends AoA cone events via HTTP POST to `/api/tripwire/event`
- Registers nodes via WebSocket so they appear in the Tripwire Nodes panel
- Calculates realistic bearings from each node to the target
- Includes confidence, cone width, and signal strength based on distance
- Adds realistic variation to simulate real-world conditions

## Prerequisites

Install required Python packages:

```bash
pip install requests websockets
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage (defaults to localhost:8080, 2 second interval):

```bash
python test_aoa_fusion.py
```

Custom host and port:

```bash
python test_aoa_fusion.py --host 192.168.1.100 --port 8080
```

Custom update interval:

```bash
python test_aoa_fusion.py --interval 1.0  # Update every 1 second
```

## Configuration

The script simulates 3 nodes positioned on an east-west line:

- **Node 1 (TW-01)**: West node (left)
- **Node 2 (TW-02)**: Center node
- **Node 3 (TW-03)**: East node (right)

**Target Location**: ~1km north of the center node (all nodes point north toward the target)

This creates a realistic triangulation scenario where nodes are spread horizontally and the RF source is to the north, providing good intersection angles for accurate fusion.

You can modify the node positions and target location in the script:

```python
NODES = [
    {
        "node_id": "tripwire-01",
        "callsign": "TW-01",
        "gps": {"lat": 37.7749, "lon": -122.4194, "alt": 10.0},
    },
    # ... more nodes
]

TARGET_LAT = 37.7799
TARGET_LON = -122.4144
```

## What It Does

1. **Registers Nodes**: Connects to Edge via WebSocket and sends hello messages so nodes appear in the UI
2. **Sends AoA Events**: Periodically sends AoA cone events with:
   - Bearing calculated from node GPS to target
   - Cone width based on confidence (25-45 degrees)
   - Confidence based on distance (closer = higher)
   - Signal strength and noise floor
   - Small random variations for realism

3. **Maintains Connections**: Keeps WebSocket connections alive with heartbeats

## Expected Behavior

When running the script, you should see:

1. **Tripwire Nodes Panel**: All 3 nodes appear as "Online" with GPS fix
2. **AoA Fusion Panel**: 
   - Canvas shows 3 nodes as colored points
   - Bearing lines extend from each node toward the intersection point
   - Cone sectors show uncertainty areas
   - Intersection point (fused location) is displayed
   - Fused coordinates and quality metric are shown

## Stopping the Script

Press `Ctrl+C` to stop. The script will gracefully close all connections.

## Troubleshooting

**Nodes don't appear in UI:**
- Check that Edge server is running
- Verify host/port are correct
- Check WebSocket connection logs

**No fusion visualization:**
- Ensure nodes have GPS data (should be automatic)
- Check that AoA events are being received (check Edge logs)
- Verify `/api/tripwire/aoa-fusion` endpoint is accessible

**Events not being received:**
- Check Edge server logs for errors
- Verify API endpoint is correct
- Check network connectivity
