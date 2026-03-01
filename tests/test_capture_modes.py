#!/usr/bin/env python3
"""
Test script for manual and armed (cued) capture modes.

Usage:
    python3 test_capture_modes.py [--manual] [--armed] [--all]
    
    --manual: Test manual capture only
    --armed:  Test armed capture only  
    --all:    Test both (default)
"""

import asyncio
import json
import time
import sys
import argparse
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Install with: pip3 install requests")
    sys.exit(1)


# Configuration
BASE_URL = "http://localhost:8000"
MANUAL_CAPTURE_ENDPOINT = f"{BASE_URL}/api/capture/start"
TRIPWIRE_EVENT_ENDPOINT = f"{BASE_URL}/api/tripwire/event"
MODE_ENDPOINT = f"{BASE_URL}/api/live/mode"
SDR_INFO_ENDPOINT = f"{BASE_URL}/api/live/sdr/info"


class CaptureTester:
    """Test harness for capture modes."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.manual_endpoint = f"{base_url}/api/capture/start"
        self.tripwire_endpoint = f"{base_url}/api/tripwire/event"
        self.mode_endpoint = f"{base_url}/api/edge/mode"  # Correct endpoint
        self.sdr_info_endpoint = f"{base_url}/api/live/sdr/info"
        
    def check_server(self) -> bool:
        """Check if server is running."""
        try:
            # Try multiple endpoints to verify server is up
            response = requests.get(f"{self.base_url}/api/live/sdr/info", timeout=2)
            if response.status_code == 200:
                return True
            # Fallback: just check if we get any response (not connection refused)
            response = requests.get(f"{self.base_url}/", timeout=2)
            return response.status_code in (200, 404)  # 404 means server is up, route just doesn't exist
        except requests.exceptions.ConnectionRefusedError:
            print(f"❌ Server not reachable at {self.base_url}: Connection refused")
            return False
        except Exception as e:
            print(f"❌ Server not reachable at {self.base_url}: {e}")
            return False
    
    def get_current_mode(self) -> Optional[str]:
        """Get current Edge mode."""
        try:
            response = requests.get(self.mode_endpoint, timeout=2)
            if response.status_code == 200:
                data = response.json()
                return data.get("mode", "unknown")
        except Exception as e:
            print(f"⚠️  Could not get current mode: {e}")
        return None
    
    def set_mode(self, mode: str) -> bool:
        """Set Edge mode (manual or armed)."""
        try:
            # Endpoint uses path parameter: /api/edge/mode/{mode}
            response = requests.post(
                f"{self.mode_endpoint}/{mode}",
                timeout=2
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    print(f"✅ Mode set to: {mode}")
                    return True
                else:
                    print(f"❌ Failed to set mode: {result.get('error', 'unknown')}")
                    return False
            else:
                print(f"❌ Failed to set mode: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error setting mode: {e}")
            return False
    
    def get_sdr_info(self) -> Optional[Dict[str, Any]]:
        """Get current SDR configuration."""
        try:
            response = requests.get(self.sdr_info_endpoint, timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️  Could not get SDR info: {e}")
        return None
    
    def test_manual_capture(self, freq_hz: float = 915_000_000, duration_s: float = 2.0) -> bool:
        """Test manual capture."""
        print("\n" + "="*70)
        print("TEST 1: Manual Capture")
        print("="*70)
        
        # Get SDR config for defaults
        sdr_info = self.get_sdr_info()
        sample_rate = 10_000_000  # Default
        if sdr_info and "sample_rate_sps" in sdr_info:
            sample_rate = sdr_info["sample_rate_sps"]
        
        print(f"\n📡 Capture Parameters:")
        print(f"   Frequency: {freq_hz/1e6:.3f} MHz")
        print(f"   Sample Rate: {sample_rate/1e6:.2f} MS/s")
        print(f"   Duration: {duration_s}s")
        print(f"   Gain Mode: manual")
        print(f"   Gain: 50 dB")
        
        # Create manual capture request
        payload = {
            "reason": "manual",
            "freq_hz": freq_hz,
            "sample_rate_sps": sample_rate,
            "bandwidth_hz": int(sample_rate * 0.8),  # 80% of sample rate
            "gain_mode": "manual",
            "gain_db": 50.0,
            "rx_channel": 0,
            "duration_s": duration_s,
            "source_node": None,
            "scan_plan": None,
            "classification": None,
        }
        
        print(f"\n📤 Sending manual capture request...")
        try:
            response = requests.post(
                self.manual_endpoint,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {json.dumps(result, indent=2)}")
                
                if result.get("accepted"):
                    print(f"✅ Manual capture queued successfully!")
                    print(f"⏳ Waiting {duration_s + 2:.1f}s for capture to complete...")
                    time.sleep(duration_s + 2)
                    print(f"✅ Manual capture test completed")
                    return True
                else:
                    print(f"❌ Capture was not accepted: {result.get('action', 'unknown')}")
                    return False
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error during manual capture: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_armed_capture(self, freq_hz: float = 915_000_000) -> bool:
        """Test armed (cued) capture via Tripwire event."""
        print("\n" + "="*70)
        print("TEST 2: Armed Capture (Tripwire Cued)")
        print("="*70)
        
        # First, ensure mode is "armed"
        current_mode = self.get_current_mode()
        if current_mode != "armed":
            print(f"⚠️  Current mode is '{current_mode}', setting to 'armed'...")
            if not self.set_mode("armed"):
                print("❌ Failed to set mode to 'armed'. Cannot test armed capture.")
                return False
            time.sleep(0.5)  # Brief delay for mode change
        
        print(f"\n📡 Event Parameters:")
        print(f"   Frequency: {freq_hz/1e6:.3f} MHz")
        print(f"   Type: fhss_cluster (v2.0 actionable event)")
        print(f"   Stage: confirmed (v1.1 format)")
        print(f"   Confidence: 0.95 (above 0.90 threshold)")
        
        # Create Tripwire event that should trigger capture
        # Using v2.0 format with fhss_cluster (actionable)
        payload = {
            "schema": "tripwire.event.v2.0",
            "type": "fhss_cluster",  # v2.0 actionable event type
            "stage": "confirmed",  # v1.1 compatibility
            "node_id": "test_node_001",
            "callsign": "TEST-001",
            "freq_hz": freq_hz,
            "bandwidth_hz": 5_000_000,
            "confidence": 0.95,  # Above 0.90 threshold
            "confidence_source": "ml_classifier",
            "classification": "unknown",
            "scan_plan": "targeted",
            "timestamp": time.time(),
            "delta_db": 15.0,
            "level_db": -45.0,
            "hop_count": 5,
            "span_mhz": 2.5,
            "unique_buckets": 12,
        }
        
        print(f"\n📤 Sending Tripwire event (should trigger capture)...")
        try:
            response = requests.post(
                self.tripwire_endpoint,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {json.dumps(result, indent=2)}")
                
                action = result.get("action", "")
                if action == "auto_capture_started":
                    print(f"✅ Armed capture triggered successfully!")
                    print(f"⏳ Waiting 7s for capture to complete...")
                    time.sleep(7)
                    print(f"✅ Armed capture test completed")
                    return True
                elif action == "rejected":
                    reason = result.get("reason", "unknown")
                    print(f"⚠️  Capture was rejected: {reason}")
                    print(f"   This may be due to cooldown or policy restrictions")
                    return False
                elif action == "queued_for_operator":
                    print(f"⚠️  Event queued for operator (mode may not be 'armed')")
                    return False
                else:
                    print(f"⚠️  Unexpected action: {action}")
                    return False
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error during armed capture: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_armed_capture_rejection(self) -> bool:
        """Test that advisory events are rejected."""
        print("\n" + "="*70)
        print("TEST 3: Armed Capture Rejection (Advisory Event)")
        print("="*70)
        
        # Ensure mode is "armed"
        current_mode = self.get_current_mode()
        if current_mode != "armed":
            print(f"⚠️  Current mode is '{current_mode}', setting to 'armed'...")
            if not self.set_mode("armed"):
                return False
            time.sleep(0.5)
        
        # Create advisory-only event (should NOT trigger capture)
        payload = {
            "schema": "tripwire.event.v1.1",
            "type": "rf_cue",  # Advisory only, never actionable
            "stage": "cue",  # Advisory stage
            "node_id": "test_node_001",
            "freq_hz": 915_000_000,
            "confidence": 0.95,
            "timestamp": time.time(),
        }
        
        print(f"\n📤 Sending advisory event (rf_cue - should NOT trigger capture)...")
        try:
            response = requests.post(
                self.tripwire_endpoint,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                action = result.get("action", "")
                
                if action == "rejected" and result.get("reason") == "cue_only":
                    print(f"✅ Advisory event correctly rejected (cue_only)")
                    return True
                elif action == "queued_for_operator":
                    print(f"✅ Advisory event queued for operator (correct behavior)")
                    return True
                else:
                    print(f"⚠️  Unexpected action: {action}")
                    return False
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def run_all_tests(self, test_manual: bool = True, test_armed: bool = True) -> bool:
        """Run all capture tests."""
        print("\n" + "="*70)
        print("CAPTURE MODES TEST SUITE")
        print("="*70)
        
        # Check server
        if not self.check_server():
            print("\n❌ Server is not running. Please start the SPEAR-Edge server first.")
            return False
        
        print("✅ Server is reachable")
        
        results = []
        
        # Test manual capture
        if test_manual:
            results.append(("Manual Capture", self.test_manual_capture()))
            time.sleep(2)  # Brief delay between tests
        
        # Test armed capture rejection (advisory event)
        if test_armed:
            results.append(("Armed Capture Rejection", self.test_armed_capture_rejection()))
            time.sleep(2)
        
        # Test armed capture (actionable event)
        if test_armed:
            results.append(("Armed Capture", self.test_armed_capture()))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        all_passed = True
        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {test_name}")
            if not passed:
                all_passed = False
        
        print("="*70)
        if all_passed:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test manual and armed capture modes")
    parser.add_argument("--manual", action="store_true", help="Test manual capture only")
    parser.add_argument("--armed", action="store_true", help="Test armed capture only")
    parser.add_argument("--all", action="store_true", help="Test both (default)")
    parser.add_argument("--url", type=str, default=BASE_URL, help=f"Base URL (default: {BASE_URL})")
    
    args = parser.parse_args()
    
    # Determine which tests to run
    test_manual = args.manual or (not args.armed and not args.manual)
    test_armed = args.armed or (not args.manual and not args.armed)
    
    if args.all:
        test_manual = True
        test_armed = True
    
    # Run tests
    tester = CaptureTester(base_url=args.url)
    success = tester.run_all_tests(test_manual=test_manual, test_armed=test_armed)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
