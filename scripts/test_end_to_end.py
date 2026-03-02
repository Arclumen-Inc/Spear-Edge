#!/usr/bin/env python3
"""
End-to-end test: Simulate Tripwire cue → Capture → Classification

Usage:
    python3 scripts/test_end_to_end.py
"""

import requests
import json
import time
import sys
from pathlib import Path

# Edge API endpoint
EDGE_URL = "http://127.0.0.1:8080"

def check_edge_running():
    """Check if Edge is running."""
    try:
        response = requests.get(f"{EDGE_URL}/api/status", timeout=2)
        if response.status_code == 200:
            print("✓ Edge is running")
            return True
    except Exception as e:
        print(f"✗ Edge is not running: {e}")
        print("  Start Edge with: python3 -m spear_edge.main")
        return False
    return False

def get_edge_mode():
    """Get current Edge mode."""
    try:
        response = requests.get(f"{EDGE_URL}/api/mode", timeout=2)
        if response.status_code == 200:
            mode = response.json().get("mode", "unknown")
            print(f"Current mode: {mode}")
            return mode
    except Exception as e:
        print(f"Error getting mode: {e}")
        return None

def set_edge_mode(mode):
    """Set Edge mode."""
    try:
        response = requests.post(
            f"{EDGE_URL}/api/mode",
            json={"mode": mode},
            timeout=2
        )
        if response.status_code == 200:
            print(f"✓ Mode set to: {mode}")
            return True
    except Exception as e:
        print(f"Error setting mode: {e}")
        return False
    return False

def send_tripwire_cue(freq_hz=915000000, classification="FHSS-like"):
    """Send a Tripwire cue event."""
    payload = {
        "schema": "spear.tripwire.event.v1",
        "type": "confirmed_event",  # Must be confirmed_event to trigger capture
        "stage": "confirmed",  # Must be "confirmed" to trigger capture
        "node_id": "test-node-01",
        "callsign": "Test-Node",
        "freq_hz": freq_hz,
        "bandwidth_hz": 2000000.0,
        "confidence": 0.95,
        "classification": classification,
        "scan_plan": "fhss_track",
        "timestamp": time.time(),
        "delta_db": 15.5,
        "level_db": -45.0
    }
    
    print(f"\n📡 Sending Tripwire cue:")
    print(f"  Frequency: {freq_hz/1e6:.3f} MHz")
    print(f"  Classification: {classification}")
    print(f"  Stage: confirmed")
    print(f"  Confidence: 0.95")
    
    try:
        response = requests.post(
            f"{EDGE_URL}/api/tripwire/event",
            json=payload,
            timeout=10
        )
        
        result = response.json()
        print(f"\n📥 Response:")
        print(json.dumps(result, indent=2))
        
        if result.get("accepted"):
            action = result.get("action", "unknown")
            if action == "auto_capture_started":
                print("\n✓ Capture started successfully!")
                return True
            elif action == "queued_for_operator":
                print("\n⚠ Capture queued (manual mode)")
                return True
            elif action == "rejected":
                reason = result.get("reason", "unknown")
                print(f"\n✗ Capture rejected: {reason}")
                return False
        else:
            print(f"\n✗ Request not accepted")
            return False
            
    except Exception as e:
        print(f"\n✗ Error sending cue: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_captures():
    """Check recent captures."""
    try:
        response = requests.get(f"{EDGE_URL}/api/captures?limit=5", timeout=2)
        if response.status_code == 200:
            captures = response.json().get("captures", [])
            if captures:
                print(f"\n📦 Recent captures ({len(captures)}):")
                for cap in captures[:3]:
                    status = cap.get("status", "unknown")
                    freq = cap.get("freq_hz", 0) / 1e6
                    reason = cap.get("reason", "unknown")
                    print(f"  - {freq:.3f} MHz, {reason}, status: {status}")
                return captures[0] if captures else None
            else:
                print("\n📦 No captures found")
                return None
    except Exception as e:
        print(f"Error checking captures: {e}")
        return None

def main():
    print("=" * 60)
    print("SPEAR-Edge End-to-End Test")
    print("=" * 60)
    
    # Check if Edge is running
    if not check_edge_running():
        sys.exit(1)
    
    # Check current mode
    mode = get_edge_mode()
    if mode != "armed":
        print(f"\n⚠ Edge is in '{mode}' mode. Need 'armed' mode for auto-capture.")
        print("Setting mode to 'armed'...")
        if not set_edge_mode("armed"):
            print("Failed to set mode. Continuing anyway...")
    
    # Wait a moment for mode to take effect
    time.sleep(1)
    
    # Send Tripwire cue at 915 MHz
    print("\n" + "=" * 60)
    success = send_tripwire_cue(freq_hz=915000000, classification="FHSS-like")
    
    if success:
        print("\n⏳ Waiting 6 seconds for capture to complete...")
        time.sleep(6)
        
        # Check captures
        capture = check_captures()
        
        if capture:
            artifact_path = capture.get("artifact_path")
            if artifact_path:
                print(f"\n✓ Capture artifact: {artifact_path}")
                
                # Check if classification was performed
                capture_dir = Path(artifact_path)
                if capture_dir.exists():
                    capture_json = capture_dir / "capture.json"
                    if capture_json.exists():
                        with open(capture_json, 'r') as f:
                            cap_data = json.load(f)
                            ml_classification = cap_data.get("ml_classification")
                            if ml_classification:
                                print(f"\n🤖 ML Classification:")
                                print(f"  Label: {ml_classification.get('label', 'unknown')}")
                                print(f"  Confidence: {ml_classification.get('confidence', 0):.2%}")
                                print(f"  Signal Type: {ml_classification.get('signal_type', 'unknown')}")
                            else:
                                print("\n⚠ No ML classification found in capture")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
