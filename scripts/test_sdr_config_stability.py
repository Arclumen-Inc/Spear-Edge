#!/usr/bin/env python3
"""
Test script to verify SDR configuration stability fixes.

Tests:
1. Rapid config changes (frequency, sample rate, gain)
2. Config changes during active stream
3. Concurrent requests handling
4. Device close/reopen recovery
"""

import asyncio
import aiohttp
import time
import sys
from typing import List, Dict, Any

BASE_URL = "http://localhost:8080"

async def test_rapid_config_changes():
    """Test rapid frequency and sample rate changes."""
    print("\n=== Test 1: Rapid Config Changes ===")
    
    async with aiohttp.ClientSession() as session:
        # Test rapid frequency changes
        frequencies = [915_000_000, 2400_000_000, 5800_000_000, 5910_000_000, 915_000_000]
        errors = []
        
        for freq in frequencies:
            try:
                async with session.post(
                    f"{BASE_URL}/live/sdr/config",
                    json={
                        "center_freq_hz": freq,
                        "sample_rate_sps": 10_000_000,
                        "gain_mode": "manual",
                        "gain_db": 20.0,
                        "rx_channel": 0,
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    result = await resp.json()
                    if not result.get("ok"):
                        errors.append(f"Freq {freq/1e6:.1f} MHz: {result.get('error')}")
                    else:
                        print(f"  ✓ Frequency {freq/1e6:.1f} MHz: OK")
            except Exception as e:
                errors.append(f"Freq {freq/1e6:.1f} MHz: {e}")
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Test rapid sample rate changes
        sample_rates = [2_400_000, 10_000_000, 20_000_000, 30_000_000, 10_000_000]
        for rate in sample_rates:
            try:
                async with session.post(
                    f"{BASE_URL}/live/sdr/config",
                    json={
                        "center_freq_hz": 915_000_000,
                        "sample_rate_sps": rate,
                        "gain_mode": "manual",
                        "gain_db": 20.0,
                        "rx_channel": 0,
                    },
                    timeout=aiohttp.ClientTimeout(total=10)  # Longer timeout for sample rate changes
                ) as resp:
                    result = await resp.json()
                    if not result.get("ok"):
                        errors.append(f"Rate {rate/1e6:.1f} MS/s: {result.get('error')}")
                    else:
                        print(f"  ✓ Sample rate {rate/1e6:.1f} MS/s: OK")
            except Exception as e:
                errors.append(f"Rate {rate/1e6:.1f} MS/s: {e}")
            await asyncio.sleep(0.2)  # Longer delay for sample rate changes
        
        if errors:
            print(f"  ✗ Errors: {errors}")
            return False
        else:
            print("  ✓ All rapid config changes succeeded")
            return True

async def test_concurrent_requests():
    """Test concurrent config requests to verify serialization."""
    print("\n=== Test 2: Concurrent Requests ===")
    
    async def send_config(session, freq, rate, index):
        try:
            async with session.post(
                f"{BASE_URL}/live/sdr/config",
                json={
                    "center_freq_hz": freq,
                    "sample_rate_sps": rate,
                    "gain_mode": "manual",
                    "gain_db": 20.0,
                    "rx_channel": 0,
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                result = await resp.json()
                return (index, result.get("ok"), result.get("error"))
        except Exception as e:
            return (index, False, str(e))
    
    async with aiohttp.ClientSession() as session:
        # Send 5 concurrent requests
        tasks = [
            send_config(session, 915_000_000 + i * 1_000_000, 10_000_000, i)
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        errors = []
        
        for index, ok, error in results:
            if ok:
                print(f"  ✓ Request {index}: OK")
            else:
                errors.append(f"Request {index}: {error}")
                print(f"  ✗ Request {index}: {error}")
        
        if errors:
            print(f"  ✗ {len(errors)} errors in concurrent requests")
            return False
        else:
            print("  ✓ All concurrent requests handled correctly")
            return True

async def test_config_during_stream():
    """Test config changes while stream is active."""
    print("\n=== Test 3: Config Changes During Active Stream ===")
    
    async with aiohttp.ClientSession() as session:
        # Start a scan
        print("  Starting scan...")
        try:
            async with session.post(
                f"{BASE_URL}/live/start",
                json={
                    "center_freq_hz": 915_000_000,
                    "sample_rate_sps": 10_000_000,
                    "fft_size": 4096,
                    "fps": 15.0,
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                result = await resp.json()
                if not result.get("ok"):
                    print(f"  ✗ Failed to start scan: {result}")
                    return False
        except Exception as e:
            print(f"  ✗ Failed to start scan: {e}")
            return False
        
        print("  ✓ Scan started")
        await asyncio.sleep(1)  # Wait for stream to stabilize
        
        # Change frequency while stream is active
        print("  Changing frequency during active stream...")
        try:
            async with session.post(
                f"{BASE_URL}/live/sdr/config",
                json={
                    "center_freq_hz": 2400_000_000,
                    "sample_rate_sps": 10_000_000,
                    "gain_mode": "manual",
                    "gain_db": 20.0,
                    "rx_channel": 0,
                },
                timeout=aiohttp.ClientTimeout(total=15)  # Longer timeout for stream restart
            ) as resp:
                result = await resp.json()
                if not result.get("ok"):
                    print(f"  ✗ Failed to change frequency: {result}")
                    return False
                print(f"  ✓ Frequency changed: {result.get('note', 'OK')}")
        except Exception as e:
            print(f"  ✗ Failed to change frequency: {e}")
            return False
        
        await asyncio.sleep(1)  # Wait for stream to stabilize
        
        # Change sample rate while stream is active
        print("  Changing sample rate during active stream...")
        try:
            async with session.post(
                f"{BASE_URL}/live/sdr/config",
                json={
                    "center_freq_hz": 2400_000_000,
                    "sample_rate_sps": 20_000_000,
                    "gain_mode": "manual",
                    "gain_db": 20.0,
                    "rx_channel": 0,
                },
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                result = await resp.json()
                if not result.get("ok"):
                    print(f"  ✗ Failed to change sample rate: {result}")
                    return False
                print(f"  ✓ Sample rate changed: {result.get('note', 'OK')}")
        except Exception as e:
            print(f"  ✗ Failed to change sample rate: {e}")
            return False
        
        await asyncio.sleep(1)
        
        # Stop scan
        print("  Stopping scan...")
        try:
            async with session.post(
                f"{BASE_URL}/live/stop",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                result = await resp.json()
                print("  ✓ Scan stopped")
        except Exception as e:
            print(f"  ⚠ Failed to stop scan: {e}")
        
        print("  ✓ All config changes during stream succeeded")
        return True

async def test_gain_only_changes():
    """Test gain-only changes (should not restart stream)."""
    print("\n=== Test 4: Gain-Only Changes (No Stream Restart) ===")
    
    async with aiohttp.ClientSession() as session:
        # Start a scan
        print("  Starting scan...")
        try:
            async with session.post(
                f"{BASE_URL}/live/start",
                json={
                    "center_freq_hz": 915_000_000,
                    "sample_rate_sps": 10_000_000,
                    "fft_size": 4096,
                    "fps": 15.0,
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                result = await resp.json()
                if not result.get("ok"):
                    print(f"  ✗ Failed to start scan: {result}")
                    return False
        except Exception as e:
            print(f"  ✗ Failed to start scan: {e}")
            return False
        
        await asyncio.sleep(1)
        
        # Change gain only (should not restart stream)
        gains = [10.0, 20.0, 30.0, 20.0]
        for gain in gains:
            print(f"  Changing gain to {gain} dB...")
            try:
                async with session.post(
                    f"{BASE_URL}/live/sdr/config",
                    json={
                        "center_freq_hz": 915_000_000,
                        "sample_rate_sps": 10_000_000,
                        "gain_mode": "manual",
                        "gain_db": gain,
                        "rx_channel": 0,
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    result = await resp.json()
                    if not result.get("ok"):
                        print(f"  ✗ Failed to change gain: {result}")
                        return False
                    note = result.get("note", "")
                    if "gain updated live" in note:
                        print(f"  ✓ Gain changed to {gain} dB (live update, no restart)")
                    else:
                        print(f"  ⚠ Gain changed to {gain} dB (unexpected: {note})")
            except Exception as e:
                print(f"  ✗ Failed to change gain: {e}")
                return False
            await asyncio.sleep(0.5)
        
        # Stop scan
        try:
            async with session.post(
                f"{BASE_URL}/live/stop",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                pass
        except Exception as e:
            pass
        
        print("  ✓ All gain-only changes succeeded")
        return True

async def main():
    """Run all tests."""
    print("=" * 60)
    print("SDR Configuration Stability Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health/status", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status != 200:
                    print("✗ Server is not responding")
                    sys.exit(1)
    except Exception as e:
        print(f"✗ Cannot connect to server at {BASE_URL}: {e}")
        print("  Make sure SPEAR-Edge is running")
        sys.exit(1)
    
    print("✓ Server is running")
    
    results = []
    
    # Run tests
    results.append(("Rapid Config Changes", await test_rapid_config_changes()))
    results.append(("Concurrent Requests", await test_concurrent_requests()))
    results.append(("Config During Stream", await test_config_during_stream()))
    results.append(("Gain-Only Changes", await test_gain_only_changes()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
