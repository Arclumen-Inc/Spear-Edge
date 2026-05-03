#!/usr/bin/env python3
"""
Test script to investigate timeout issues with different channel configurations:
- Single RX on channel 0
- Single RX on channel 1
- Dual RX (both channels)

This will help identify if timeouts are channel-specific or configuration-related.
"""

import sys
import time
import numpy as np
sys.path.insert(0, '/home/spear/spear-edgev1_0')

from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice, BLADERF_CHANNEL_RX, BLADERF_RX_X1, BLADERF_RX_X2
from spear_edge.core.sdr.base import SdrConfig, GainMode

# Import libbladerf constants
import ctypes
_libbladerf = None
try:
    _libbladerf = ctypes.CDLL("/usr/local/lib/libbladeRF.so.2")
except OSError:
    try:
        _libbladerf = ctypes.CDLL("libbladeRF.so.2")
    except OSError:
        _libbladerf = ctypes.CDLL("libbladeRF.so")

BLADERF_ERR_TIMEOUT = -1
BLADERF_FORMAT_SC16_Q11 = 0x0001

def test_single_rx_channel(channel: int, num_reads: int = 1000, read_size: int = 8192):
    """Test single RX on specified channel (0 or 1)."""
    print(f"\n{'='*70}")
    print(f"TEST: Single RX on Channel {channel}")
    print(f"{'='*70}")
    
    try:
        sdr = BladeRFNativeDevice()
        
        # Configure for single channel
        # Note: We need to manually configure since the class is hardcoded to ch0
        # We'll use the tune() method but then manually reconfigure the channel
        sdr.rx_channel = channel
        sdr.max_rx_channels = 1
        
        # Tune to a test frequency
        center_freq = 915_000_000  # 915 MHz
        sample_rate = 4_000_000    # 4 MS/s
        bandwidth = 4_000_000
        
        print(f"Configuring: {center_freq/1e6:.1f} MHz @ {sample_rate/1e6:.1f} MS/s")
        
        # For channel 1, we need to manually configure since tune() uses rx_channel
        # First set the channel, then tune
        if channel == 1:
            # Manually configure channel 1
            ch1 = BLADERF_CHANNEL_RX(1)
            _libbladerf.bladerf_set_sample_rate(sdr.dev, ch1, sample_rate, None)
            _libbladerf.bladerf_set_bandwidth(sdr.dev, ch1, bandwidth, None)
            _libbladerf.bladerf_set_frequency(sdr.dev, ch1, center_freq)
            _libbladerf.bladerf_set_gain_mode(sdr.dev, ch1, 0)  # Manual gain
            _libbladerf.bladerf_set_gain(sdr.dev, ch1, 30)
            
            # Deactivate any existing stream
            if sdr._stream_active:
                sdr._deactivate_stream()
            
            # Setup stream for channel 1
            num_buffers = 64
            buffer_size = 131072
            num_transfers = 16
            stream_timeout_ms = 5000
            
            ret = _libbladerf.bladerf_sync_config(
                sdr.dev,
                BLADERF_RX_X1,
                BLADERF_FORMAT_SC16_Q11,
                num_buffers,
                buffer_size,
                num_transfers,
                stream_timeout_ms
            )
            
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                raise RuntimeError(f"Failed to configure RX stream for ch1: {error_str}")
            
            # Enable RX module for channel 1
            ret = _libbladerf.bladerf_enable_module(sdr.dev, ch1, True)
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                raise RuntimeError(f"Failed to enable RX module for ch1: {error_str}")
            
            sdr._stream_active = True
            sdr._stream_configured = True
            sdr.rx_channel = 1
        else:
            # Use normal tune() for channel 0
            sdr.tune(center_freq, sample_rate, bandwidth)
            sdr.set_gain(30.0)
            sdr.set_gain_mode(GainMode.MANUAL)
        
        # Wait for stream to stabilize
        time.sleep(0.5)
        
        # Test reads
        print(f"\nPerforming {num_reads} reads of {read_size} samples each...")
        stats = {
            "total": 0,
            "successful": 0,
            "timeouts": 0,
            "errors": 0,
            "empty": 0,
            "read_times": [],
        }
        
        start_time = time.time()
        
        for i in range(num_reads):
            stats["total"] += 1
            read_start = time.perf_counter_ns()
            
            samples = sdr.read_samples(read_size)
            
            read_time_ms = (time.perf_counter_ns() - read_start) / 1_000_000
            stats["read_times"].append(read_time_ms)
            
            if samples.size == 0:
                stats["empty"] += 1
                # For channel 1, we need to track timeouts differently
                # The health stats might not be accurate for manual channel 1 setup
                # We'll track based on read time instead
                if read_time_ms >= 100:
                    stats["timeouts"] += 1
            else:
                stats["successful"] += 1
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{num_reads} reads ({rate:.1f} reads/sec)", end='\r')
        
        elapsed = time.time() - start_time
        
        # Calculate statistics
        read_times = np.array(stats["read_times"])
        avg_read_time = np.mean(read_times) if len(read_times) > 0 else 0
        max_read_time = np.max(read_times) if len(read_times) > 0 else 0
        min_read_time = np.min(read_times) if len(read_times) > 0 else 0
        p95_read_time = np.percentile(read_times, 95) if len(read_times) > 0 else 0
        p99_read_time = np.percentile(read_times, 99) if len(read_times) > 0 else 0
        
        # Get final health (may not be accurate for channel 1)
        health = sdr.get_health()
        
        print(f"\n\nResults:")
        print(f"  Total reads: {stats['total']}")
        print(f"  Successful: {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
        print(f"  Empty reads: {stats['empty']} ({stats['empty']/stats['total']*100:.1f}%)")
        print(f"  Timeouts (estimated): {stats['timeouts']}")
        if channel == 0:
            print(f"  Timeouts (from health): {health['timeouts']}")
            print(f"  Errors (from health): {health['errors']}")
        print(f"\n  Read time statistics:")
        print(f"    Average: {avg_read_time:.2f} ms")
        print(f"    Min: {min_read_time:.2f} ms")
        print(f"    Max: {max_read_time:.2f} ms")
        print(f"    95th percentile: {p95_read_time:.2f} ms")
        print(f"    99th percentile: {p99_read_time:.2f} ms")
        print(f"    Reads > 100ms: {np.sum(read_times > 100)} ({np.sum(read_times > 100)/len(read_times)*100:.1f}%)")
        print(f"    Reads > 200ms: {np.sum(read_times > 200)} ({np.sum(read_times > 200)/len(read_times)*100:.1f}%)")
        
        sdr.close()
        return stats, read_times
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_dual_rx(num_reads: int = 1000, read_size: int = 8192):
    """Test dual RX (both channels simultaneously)."""
    print(f"\n{'='*70}")
    print(f"TEST: Dual RX (Channels 0 and 1)")
    print(f"{'='*70}")
    
    try:
        sdr = BladeRFNativeDevice()
        
        # Configure for dual channel
        # Note: This requires modifying the stream setup to use BLADERF_RX_X2
        # For now, we'll test if we can enable both channels
        
        # Tune both channels
        center_freq = 915_000_000  # 915 MHz
        sample_rate = 4_000_000    # 4 MS/s
        bandwidth = 4_000_000
        
        print(f"Configuring: {center_freq/1e6:.1f} MHz @ {sample_rate/1e6:.1f} MS/s")
        
        # Configure channel 0
        ch0 = BLADERF_CHANNEL_RX(0)
        _libbladerf.bladerf_set_sample_rate(sdr.dev, ch0, sample_rate, None)
        _libbladerf.bladerf_set_bandwidth(sdr.dev, ch0, bandwidth, None)
        _libbladerf.bladerf_set_frequency(sdr.dev, ch0, center_freq)
        _libbladerf.bladerf_set_gain_mode(sdr.dev, ch0, 0)  # Manual gain
        _libbladerf.bladerf_set_gain(sdr.dev, ch0, 30)
        
        # Configure channel 1
        ch1 = BLADERF_CHANNEL_RX(1)
        _libbladerf.bladerf_set_sample_rate(sdr.dev, ch1, sample_rate, None)
        _libbladerf.bladerf_set_bandwidth(sdr.dev, ch1, bandwidth, None)
        _libbladerf.bladerf_set_frequency(sdr.dev, ch1, center_freq)
        _libbladerf.bladerf_set_gain_mode(sdr.dev, ch1, 0)  # Manual gain
        _libbladerf.bladerf_set_gain(sdr.dev, ch1, 30)
        
        # Setup dual-channel stream
        num_buffers = 64
        buffer_size = 131072
        num_transfers = 16
        stream_timeout_ms = 5000
        
        ret = _libbladerf.bladerf_sync_config(
            sdr.dev,
            BLADERF_RX_X2,              # Dual-channel layout
            BLADERF_FORMAT_SC16_Q11,
            num_buffers,
            buffer_size,
            num_transfers,
            stream_timeout_ms
        )
        
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"ERROR: Failed to configure dual RX stream: {error_str} (code: {ret})")
            sdr.close()
            return None, None
        
        # Enable both RX modules
        _libbladerf.bladerf_enable_module(sdr.dev, ch0, True)
        _libbladerf.bladerf_enable_module(sdr.dev, ch1, True)
        
        time.sleep(0.5)
        
        print(f"\nPerforming {num_reads} reads of {read_size} samples each (dual channel)...")
        stats = {
            "total": 0,
            "successful": 0,
            "timeouts": 0,
            "errors": 0,
            "empty": 0,
            "read_times": [],
        }
        
        start_time = time.time()
        
        # For dual RX, buffer size is 2x (interleaved I/Q from both channels)
        buf_size = read_size * 2 * 2  # samples * 2 channels * 2 (I/Q)
        buf = (ctypes.c_int16 * buf_size)()
        
        for i in range(num_reads):
            stats["total"] += 1
            read_start = time.perf_counter_ns()
            
            # Read dual channel samples
            timeout_ms = 100
            ret = _libbladerf.bladerf_sync_rx(
                sdr.dev,
                buf,
                read_size,  # Samples per channel
                None,
                timeout_ms
            )
            
            read_time_ms = (time.perf_counter_ns() - read_start) / 1_000_000
            stats["read_times"].append(read_time_ms)
            
            if ret != 0:
                stats["empty"] += 1
                if ret == BLADERF_ERR_TIMEOUT:
                    stats["timeouts"] += 1
                else:
                    stats["errors"] += 1
            else:
                stats["successful"] += 1
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{num_reads} reads ({rate:.1f} reads/sec)", end='\r')
        
        elapsed = time.time() - start_time
        
        # Calculate statistics
        read_times = np.array(stats["read_times"])
        avg_read_time = np.mean(read_times) if len(read_times) > 0 else 0
        max_read_time = np.max(read_times) if len(read_times) > 0 else 0
        min_read_time = np.min(read_times) if len(read_times) > 0 else 0
        p95_read_time = np.percentile(read_times, 95) if len(read_times) > 0 else 0
        p99_read_time = np.percentile(read_times, 99) if len(read_times) > 0 else 0
        
        print(f"\n\nResults:")
        print(f"  Total reads: {stats['total']}")
        print(f"  Successful: {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
        print(f"  Empty reads: {stats['empty']} ({stats['empty']/stats['total']*100:.1f}%)")
        print(f"  Timeouts: {stats['timeouts']}")
        print(f"  Errors: {stats['errors']}")
        print(f"\n  Read time statistics:")
        print(f"    Average: {avg_read_time:.2f} ms")
        print(f"    Min: {min_read_time:.2f} ms")
        print(f"    Max: {max_read_time:.2f} ms")
        print(f"    95th percentile: {p95_read_time:.2f} ms")
        print(f"    99th percentile: {p99_read_time:.2f} ms")
        print(f"    Reads > 100ms: {np.sum(read_times > 100)} ({np.sum(read_times > 100)/len(read_times)*100:.1f}%)")
        print(f"    Reads > 200ms: {np.sum(read_times > 200)} ({np.sum(read_times > 200)/len(read_times)*100:.1f}%)")
        
        # Disable modules
        _libbladerf.bladerf_enable_module(sdr.dev, ch0, False)
        _libbladerf.bladerf_enable_module(sdr.dev, ch1, False)
        
        sdr.close()
        return stats, read_times
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    print("="*70)
    print("bladeRF Channel Configuration Timeout Investigation")
    print("="*70)
    print("\nThis script tests:")
    print("  1. Single RX on Channel 0")
    print("  2. Single RX on Channel 1")
    print("  3. Dual RX (both channels)")
    print("\nEach test performs 1000 reads and analyzes timeout patterns.")
    
    results = {}
    
    # Test 1: Single RX Channel 0
    results['ch0'] = test_single_rx_channel(0, num_reads=1000, read_size=8192)
    time.sleep(1)
    
    # Test 2: Single RX Channel 1
    results['ch1'] = test_single_rx_channel(1, num_reads=1000, read_size=8192)
    time.sleep(1)
    
    # Test 3: Dual RX
    results['dual'] = test_dual_rx(num_reads=1000, read_size=8192)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for test_name, (stats, read_times) in results.items():
        if stats is None:
            print(f"\n{test_name}: FAILED")
            continue
        
        print(f"\n{test_name.upper()}:")
        print(f"  Success rate: {stats['successful']/stats['total']*100:.1f}%")
        print(f"  Timeout rate: {stats['timeouts']/stats['total']*100:.1f}%")
        if read_times is not None and len(read_times) > 0:
            print(f"  Avg read time: {np.mean(read_times):.2f} ms")
            print(f"  Max read time: {np.max(read_times):.2f} ms")
            print(f"  Reads > 100ms: {np.sum(read_times > 100)} ({np.sum(read_times > 100)/len(read_times)*100:.1f}%)")


if __name__ == "__main__":
    main()
