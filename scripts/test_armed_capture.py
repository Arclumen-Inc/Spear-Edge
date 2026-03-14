#!/usr/bin/env python3
"""
Test script for Armed mode capture.

This script:
1. Sets Edge to Armed mode
2. Configures SDR to 100 MHz, 2.0 MS/s
3. Starts the live scan (so capture can resume after)
4. Injects a Tripwire event that triggers auto-capture

Run from another SSH session while Edge is running.

Usage:
    python3 scripts/test_armed_capture.py
"""

import requests
import time
import json

EDGE_URL = "http://localhost:8080"

def check_edge_status():
    """Check if Edge is running and get current status."""
    try:
        r = requests.get(f"{EDGE_URL}/health/status", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Cannot connect to Edge: {e}")
        return None

def set_mode(mode: str):
    """Set Edge mode (manual/armed)."""
    r = requests.post(f"{EDGE_URL}/api/edge/mode/{mode}", timeout=5)
    r.raise_for_status()
    print(f"[OK] Mode set to: {mode}")
    return r.json()

def get_mode():
    """Get current Edge mode."""
    r = requests.get(f"{EDGE_URL}/api/edge/mode", timeout=5)
    r.raise_for_status()
    return r.json().get("mode", "unknown")

def configure_sdr(center_freq_hz: int, sample_rate_sps: int, fft_size: int = 4096):
    """Configure SDR and start live scan."""
    payload = {
        "center_freq_hz": center_freq_hz,
        "sample_rate_sps": sample_rate_sps,
        "fft_size": fft_size,
        "fps": 15.0
    }
    r = requests.post(f"{EDGE_URL}/live/start", json=payload, timeout=10)
    r.raise_for_status()
    print(f"[OK] SDR configured: {center_freq_hz/1e6:.3f} MHz, {sample_rate_sps/1e6:.1f} MS/s, FFT {fft_size}")
    return r.json()

def get_auto_policy():
    """Get current auto-capture policy."""
    r = requests.get(f"{EDGE_URL}/live/auto/policy", timeout=5)
    r.raise_for_status()
    return r.json()

def set_auto_policy(min_confidence: float = None):
    """Update auto-capture policy."""
    payload = {}
    if min_confidence is not None:
        payload["min_confidence"] = min_confidence
    r = requests.post(f"{EDGE_URL}/live/auto/policy", json=payload, timeout=5)
    r.raise_for_status()
    print(f"[OK] Auto policy updated: {r.json()}")
    return r.json()

def inject_tripwire_event(freq_hz: float, confidence: float = 0.95):
    """
    Inject a Tripwire event that will trigger auto-capture.
    
    Uses fhss_cluster event type (v2.0 actionable) with high confidence.
    """
    event = {
        "type": "fhss_cluster",  # v2.0 actionable event type
        "node_id": "test-tripwire-001",
        "callsign": "TEST-TW",
        "freq_hz": freq_hz,
        "bandwidth_hz": 1_000_000,  # 1 MHz
        "confidence": confidence,
        "classification": "unknown",
        "scan_plan": "test_scan",
        "timestamp": time.time(),
        # v2.0 FHSS cluster fields
        "hop_count": 5,
        "span_mhz": 2.0,
        "unique_buckets": 3,
        # Optional GPS (for AoA/TAI)
        "gps_lat": 32.0,
        "gps_lon": -117.0,
    }
    
    print(f"[INFO] Injecting event: {event['type']} @ {freq_hz/1e6:.3f} MHz (conf={confidence})")
    
    r = requests.post(f"{EDGE_URL}/api/tripwire/event", json=event, timeout=10)
    r.raise_for_status()
    result = r.json()
    print(f"[RESULT] {json.dumps(result, indent=2)}")
    return result

def get_captures(limit: int = 5):
    """Get recent captures."""
    r = requests.get(f"{EDGE_URL}/live/captures", params={"limit": limit}, timeout=5)
    r.raise_for_status()
    return r.json().get("captures", [])

def main():
    print("=" * 60)
    print("SPEAR Edge Armed Mode Capture Test")
    print("=" * 60)
    
    # 1. Check Edge is running
    print("\n[1] Checking Edge status...")
    status = check_edge_status()
    if not status:
        print("[FAIL] Edge is not running. Start it first.")
        return 1
    print(f"    Status: {json.dumps(status, indent=4)}")
    
    # 2. Configure SDR (100 MHz, 2.0 MS/s, FFT 4096)
    print("\n[2] Configuring SDR...")
    try:
        configure_sdr(
            center_freq_hz=100_000_000,  # 100 MHz
            sample_rate_sps=2_000_000,   # 2.0 MS/s
            fft_size=4096
        )
    except Exception as e:
        print(f"[ERROR] Failed to configure SDR: {e}")
        return 1
    
    # Wait for scan to start
    time.sleep(1)
    
    # 3. Check/set auto policy (ensure min_confidence allows our test)
    print("\n[3] Checking auto-capture policy...")
    policy = get_auto_policy()
    print(f"    Current policy: {json.dumps(policy, indent=4)}")
    
    # 4. Set mode to Armed
    print("\n[4] Setting mode to ARMED...")
    current_mode = get_mode()
    print(f"    Current mode: {current_mode}")
    if current_mode != "armed":
        set_mode("armed")
    else:
        print("    Already in armed mode")
    
    # Verify mode
    time.sleep(0.5)
    current_mode = get_mode()
    print(f"    Mode after set: {current_mode}")
    
    if current_mode != "armed":
        print("[FAIL] Could not set armed mode")
        return 1
    
    # 5. Get initial capture count
    print("\n[5] Getting initial capture count...")
    initial_captures = get_captures(limit=10)
    initial_count = len(initial_captures)
    print(f"    Initial captures: {initial_count}")
    
    # 6. Inject Tripwire event
    print("\n[6] Injecting Tripwire event...")
    result = inject_tripwire_event(
        freq_hz=100_000_000,  # 100 MHz (same as SDR center)
        confidence=0.95      # High confidence to pass policy
    )
    
    action = result.get("action", "unknown")
    if action == "auto_capture_started":
        print("[OK] Auto-capture started!")
    elif action == "rejected":
        print(f"[WARN] Event rejected: {result.get('reason', 'unknown')}")
        return 1
    elif action == "queued_for_operator":
        print("[WARN] Not in armed mode - event queued for operator")
        return 1
    else:
        print(f"[INFO] Action: {action}")
    
    # 7. Wait for capture to complete
    print("\n[7] Waiting for capture to complete (5-10 seconds)...")
    for i in range(20):
        time.sleep(1)
        status = check_edge_status()
        mode = status.get("mode", "unknown") if status else "unknown"
        print(f"    [{i+1}s] Mode: {mode}")
        
        # Check if capture completed (mode goes back to armed)
        if mode == "armed" and i > 3:
            print("    Capture appears complete (mode returned to armed)")
            break
    
    # 8. Check for new captures
    print("\n[8] Checking for new captures...")
    time.sleep(2)  # Wait a bit more for capture to be logged
    final_captures = get_captures(limit=10)
    final_count = len(final_captures)
    new_captures = final_count - initial_count
    
    print(f"    Final captures: {final_count}")
    print(f"    New captures: {new_captures}")
    
    if new_captures > 0:
        print("\n[SUCCESS] Capture completed!")
        print(f"    Latest capture: {json.dumps(final_captures[0], indent=4)}")
        return 0
    else:
        print("\n[WARN] No new captures detected")
        print("    Check Edge logs for capture status")
        return 1

if __name__ == "__main__":
    exit(main())
