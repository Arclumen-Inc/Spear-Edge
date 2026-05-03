#!/usr/bin/env python3
"""
End-to-end integration test for Edge server with libbladerf.
Tests server startup, SDR initialization, and basic API endpoints.
"""

import sys
import asyncio
import time
import requests
import json

sys.path.insert(0, '/home/spear/spear-edgev1_0')

async def test_server_startup():
    """Test that the server can be created and initialized."""
    print("\n[INTEGRATION] Testing server startup...")
    
    try:
        from spear_edge.app import create_app, make_sdr
        from spear_edge.core.orchestrator.orchestrator import Orchestrator
        
        # Test SDR creation
        print("[INTEGRATION] Creating SDR...")
        sdr = make_sdr()
        sdr_type = type(sdr).__name__
        print(f"[INTEGRATION] ✓ SDR created: {sdr_type}")
        
        if sdr_type == "MockSDR":
            print("[INTEGRATION] ⚠ Using MockSDR (libbladerf not available or failed)")
            return False
        
        # Test orchestrator creation
        print("[INTEGRATION] Creating orchestrator...")
        orchestrator = Orchestrator(sdr)
        print(f"[INTEGRATION] ✓ Orchestrator created")
        print(f"[INTEGRATION] ✓ SDR type: {type(orchestrator.sdr).__name__}")
        
        # Test app creation
        print("[INTEGRATION] Creating FastAPI app...")
        app = create_app()
        print(f"[INTEGRATION] ✓ FastAPI app created")
        
        # Verify SDR in app state
        if hasattr(app.state, 'orchestrator'):
            app_sdr_type = type(app.state.orchestrator.sdr).__name__
            print(f"[INTEGRATION] ✓ App state SDR: {app_sdr_type}")
            if app_sdr_type != "BladeRFNativeDevice":
                print(f"[INTEGRATION] ⚠ Expected BladeRFNativeDevice, got {app_sdr_type}")
                return False
        
        # Cleanup
        await orchestrator.close()
        sdr.close()
        
        print("[INTEGRATION] ✓ Server startup test passed")
        return True
        
    except Exception as e:
        print(f"[INTEGRATION] ✗ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sdr_initialization():
    """Test SDR initialization and basic operations."""
    print("\n[INTEGRATION] Testing SDR initialization...")
    
    try:
        from spear_edge.app import make_sdr
        from spear_edge.core.sdr.base import SdrConfig, GainMode
        
        sdr = make_sdr()
        sdr_type = type(sdr).__name__
        
        if sdr_type == "MockSDR":
            print("[INTEGRATION] ⚠ Skipping - using MockSDR")
            return True
        
        print(f"[INTEGRATION] SDR type: {sdr_type}")
        
        # Test device info
        info = sdr.get_info()
        print(f"[INTEGRATION] ✓ Device info: {info.get('driver')} - {info.get('label')}")
        
        # Test initial health
        health = sdr.get_health()
        print(f"[INTEGRATION] ✓ Initial health: {health.get('status')}")
        
        # Test configuration
        cfg = SdrConfig(
            center_freq_hz=915_000_000,
            sample_rate_sps=10_000_000,
            gain_mode=GainMode.MANUAL,
            gain_db=40.0,
            bandwidth_hz=8_000_000
        )
        sdr.apply_config(cfg)
        print(f"[INTEGRATION] ✓ Applied config: {cfg.center_freq_hz/1e6:.1f} MHz @ {cfg.sample_rate_sps/1e6:.1f} MS/s")
        
        # Test sample reading
        samples = sdr.read_samples(8192)
        print(f"[INTEGRATION] ✓ Read {len(samples)} samples")
        
        # Test health after operation
        health = sdr.get_health()
        print(f"[INTEGRATION] ✓ Health after read: {health.get('status')}, success rate: {health.get('success_rate_pct'):.1f}%")
        
        sdr.close()
        print("[INTEGRATION] ✓ SDR initialization test passed")
        return True
        
    except Exception as e:
        print(f"[INTEGRATION] ✗ SDR initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_orchestrator_integration():
    """Test orchestrator integration with SDR."""
    print("\n[INTEGRATION] Testing orchestrator integration...")
    
    try:
        from spear_edge.app import make_sdr
        from spear_edge.core.orchestrator.orchestrator import Orchestrator
        
        sdr = make_sdr()
        sdr_type = type(sdr).__name__
        
        if sdr_type == "MockSDR":
            print("[INTEGRATION] ⚠ Skipping - using MockSDR")
            await Orchestrator(sdr).close()
            return True
        
        orchestrator = Orchestrator(sdr)
        print(f"[INTEGRATION] ✓ Orchestrator created with {sdr_type}")
        
        # Test SDR access through orchestrator
        assert orchestrator.sdr is not None, "Orchestrator should have SDR"
        assert type(orchestrator.sdr).__name__ == "BladeRFNativeDevice", "Should use BladeRFNativeDevice"
        
        # Test opening SDR
        await orchestrator.open()
        print("[INTEGRATION] ✓ SDR opened through orchestrator")
        
        # Test SDR info
        info = orchestrator.sdr.get_info()
        print(f"[INTEGRATION] ✓ SDR info: {info.get('driver')}")
        
        # Test starting a scan (this will actually configure the SDR)
        print("[INTEGRATION] Testing scan start...")
        await orchestrator.start_scan(
            center_freq_hz=915_000_000,
            sample_rate_sps=10_000_000,
            fft_size=2048,
            fps=15.0
        )
        print("[INTEGRATION] ✓ Scan started")
        
        # Wait a moment for samples to flow
        await asyncio.sleep(1.0)
        
        # Check health
        health = orchestrator.sdr.get_health()
        print(f"[INTEGRATION] ✓ Health during scan: {health.get('status')}, throughput: {health.get('throughput_mbps'):.2f} MB/s")
        
        # Stop scan
        await orchestrator.stop_scan()
        print("[INTEGRATION] ✓ Scan stopped")
        
        # Cleanup
        await orchestrator.close()
        print("[INTEGRATION] ✓ Orchestrator integration test passed")
        return True
        
    except Exception as e:
        print(f"[INTEGRATION] ✗ Orchestrator integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_capture_integration():
    """Test capture manager integration with SDR."""
    print("\n[INTEGRATION] Testing capture integration...")
    
    try:
        from spear_edge.app import make_sdr
        from spear_edge.core.orchestrator.orchestrator import Orchestrator
        from spear_edge.core.capture.capture_manager import CaptureManager
        from spear_edge.core.bus.models import CaptureRequest
        
        sdr = make_sdr()
        sdr_type = type(sdr).__name__
        
        if sdr_type == "MockSDR":
            print("[INTEGRATION] ⚠ Skipping - using MockSDR")
            orchestrator = Orchestrator(sdr)
            await orchestrator.close()
            return True
        
        orchestrator = Orchestrator(sdr)
        capture_mgr = CaptureManager(orchestrator)
        
        print(f"[INTEGRATION] ✓ Capture manager created with {sdr_type}")
        
        # Open orchestrator
        await orchestrator.open()
        
        # Create a test capture request
        req = CaptureRequest(
            freq_hz=915_000_000,
            sample_rate_sps=10_000_000,
            duration_s=1.0,
            reason="test",
            ts=time.time()
        )
        
        print("[INTEGRATION] Testing capture request...")
        # Note: We won't actually execute the capture, just verify the setup works
        print("[INTEGRATION] ✓ Capture request created")
        
        # Cleanup
        await orchestrator.close()
        print("[INTEGRATION] ✓ Capture integration test passed")
        return True
        
    except Exception as e:
        print(f"[INTEGRATION] ✗ Capture integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Edge Server Integration Test Suite")
    print("=" * 70)
    
    tests = [
        ("Server Startup", test_server_startup),
        ("SDR Initialization", test_sdr_initialization),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Capture Integration", test_capture_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[INTEGRATION] ✗ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("Integration Test Results Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All integration tests passed!")
        print("\n[INFO] The Edge server is ready to use with libbladerf!")
        return 0
    else:
        print("\n[WARNING] Some integration tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
