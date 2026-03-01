#!/usr/bin/env python3
"""
Standalone bladeRF noise floor test using libbladerf directly.
This tests the hardware outside of SPEAR-Edge to verify actual noise floor behavior.
"""

import ctypes
import numpy as np
import sys
import time

# Try to load libbladerf (same logic as SPEAR-Edge)
try:
    _libbladerf = ctypes.CDLL("/usr/local/lib/libbladeRF.so.2")
    print("[OK] Loaded libbladeRF.so.2 from /usr/local/lib")
except OSError:
    try:
        _libbladerf = ctypes.CDLL("libbladeRF.so.2")
        print("[OK] Loaded libbladeRF.so.2 from system")
    except OSError:
        try:
            _libbladerf = ctypes.CDLL("libbladeRF.so")
            print("[OK] Loaded libbladeRF.so from system")
        except OSError:
            print("ERROR: Could not load libbladeRF. Is it installed?")
            print("Tried: /usr/local/lib/libbladeRF.so.2, libbladeRF.so.2, libbladeRF.so")
            sys.exit(1)

# Constants from libbladerf.h (matching SPEAR-Edge encoding)
def BLADERF_CHANNEL_RX(ch: int) -> int:
    """Encode RX channel: ((ch) << 1) | 0x0 (matches header macro)"""
    return ((ch) << 1) | 0x0

BLADERF_GAIN_MGC = 0
BLADERF_GAIN_AGC = 1
BLADERF_FORMAT_SC16_Q11 = 0x0001

# Function signatures
_libbladerf.bladerf_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
_libbladerf.bladerf_open.restype = ctypes.c_int

_libbladerf.bladerf_close.argtypes = [ctypes.c_void_p]
_libbladerf.bladerf_close.restype = None

_libbladerf.bladerf_set_sample_rate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_libbladerf.bladerf_set_sample_rate.restype = ctypes.c_int

_libbladerf.bladerf_set_bandwidth.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_libbladerf.bladerf_set_bandwidth.restype = ctypes.c_int

_libbladerf.bladerf_set_frequency.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
_libbladerf.bladerf_set_frequency.restype = ctypes.c_int

_libbladerf.bladerf_set_gain_mode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_libbladerf.bladerf_set_gain_mode.restype = ctypes.c_int

_libbladerf.bladerf_set_gain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_libbladerf.bladerf_set_gain.restype = ctypes.c_int

_libbladerf.bladerf_get_gain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_libbladerf.bladerf_get_gain.restype = ctypes.c_int

_libbladerf.bladerf_enable_module.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
_libbladerf.bladerf_enable_module.restype = ctypes.c_int

_libbladerf.bladerf_sync_config.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint16
]
_libbladerf.bladerf_sync_config.restype = ctypes.c_int

_libbladerf.bladerf_sync_rx.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32
]
_libbladerf.bladerf_sync_rx.restype = ctypes.c_int

_libbladerf.bladerf_strerror.argtypes = [ctypes.c_int]
_libbladerf.bladerf_strerror.restype = ctypes.c_char_p


def test_noise_floor(center_freq_hz=915000000, sample_rate_sps=30000000, gain_db=0, num_reads=100):
    """Test noise floor at specified settings."""
    
    print(f"\n{'='*70}")
    print(f"bladeRF Noise Floor Test")
    print(f"{'='*70}")
    print(f"Frequency: {center_freq_hz/1e6:.3f} MHz")
    print(f"Sample Rate: {sample_rate_sps/1e6:.3f} MS/s")
    print(f"Gain: {gain_db} dB")
    print(f"Reads: {num_reads}")
    print(f"{'='*70}\n")
    
    # Open device
    dev_ptr = ctypes.c_void_p()
    ret = _libbladerf.bladerf_open(ctypes.byref(dev_ptr), None)
    if ret != 0:
        error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
        print(f"ERROR: Failed to open bladeRF: {error_str} (code: {ret})")
        return
    dev = dev_ptr.value
    print(f"[OK] Opened bladeRF device")
    
    try:
        ch = BLADERF_CHANNEL_RX(0)  # Channel 0 = 0x00
        
        # Configure RF parameters (CRITICAL ORDER)
        print(f"\n[1/7] Setting gain mode to MANUAL...")
        ret = _libbladerf.bladerf_set_gain_mode(dev, ch, BLADERF_GAIN_MGC)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  WARNING: Failed to set gain mode: {error_str}")
        else:
            print(f"  [OK] Gain mode set to MANUAL")
        
        print(f"[2/7] Setting sample rate to {sample_rate_sps/1e6:.3f} MS/s...")
        actual_rate = ctypes.c_uint32()
        ret = _libbladerf.bladerf_set_sample_rate(dev, ch, sample_rate_sps, ctypes.byref(actual_rate))
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  ERROR: Failed to set sample rate: {error_str}")
            return
        print(f"  [OK] Sample rate: {actual_rate.value/1e6:.3f} MS/s (requested: {sample_rate_sps/1e6:.3f} MS/s)")
        
        print(f"[3/7] Setting bandwidth...")
        actual_bw = ctypes.c_uint32()
        ret = _libbladerf.bladerf_set_bandwidth(dev, ch, sample_rate_sps, ctypes.byref(actual_bw))
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  WARNING: Failed to set bandwidth: {error_str}")
        else:
            print(f"  [OK] Bandwidth: {actual_bw.value/1e6:.3f} MHz")
        
        print(f"[4/7] Setting frequency to {center_freq_hz/1e6:.3f} MHz...")
        ret = _libbladerf.bladerf_set_frequency(dev, ch, center_freq_hz)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  ERROR: Failed to set frequency: {error_str}")
            return
        print(f"  [OK] Frequency set")
        
        print(f"[5/7] Enabling RX module...")
        ret = _libbladerf.bladerf_enable_module(dev, ch, True)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  ERROR: Failed to enable RX: {error_str}")
            return
        print(f"  [OK] RX module enabled")
        
        print(f"[6/7] Configuring stream...")
        # Configure sync interface
        # Format: SC16_Q11, num_buffers=16, buffer_size=8192, num_transfers=8, timeout_ms=3500
        ret = _libbladerf.bladerf_sync_config(
            dev, ch, BLADERF_FORMAT_SC16_Q11, 16, 8192, 8, 3500, 0
        )
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  ERROR: Failed to configure stream: {error_str}")
            return
        print(f"  [OK] Stream configured")
        
        print(f"[7/7] Setting gain to {gain_db} dB (AFTER stream setup)...")
        gain_val = int(gain_db)
        print(f"  DEBUG: gain_db={gain_db}, gain_val={gain_val}, ch={ch}")
        ret = _libbladerf.bladerf_set_gain(dev, ch, gain_val)
        print(f"  DEBUG: bladerf_set_gain returned: {ret}")
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  WARNING: Failed to set gain: {error_str}")
        else:
            # Small delay to ensure gain is applied
            time.sleep(0.05)  # Longer delay
            # Read back gain (using pointer argument)
            gain_ptr = ctypes.c_int()
            ret_get = _libbladerf.bladerf_get_gain(dev, ch, ctypes.byref(gain_ptr))
            if ret_get == 0:
                applied_gain = gain_ptr.value
                print(f"  [OK] Gain requested: {gain_db} dB (int={gain_val}), hardware applied: {applied_gain} dB")
                if applied_gain != gain_val:
                    print(f"  ⚠️  WARNING: Gain mismatch! Requested {gain_val} but got {applied_gain}")
            else:
                error_str = _libbladerf.bladerf_strerror(ret_get).decode('utf-8', errors='ignore')
                print(f"  WARNING: Could not read back gain: {error_str}")
        
        # Read samples and analyze
        print(f"\n{'='*70}")
        print(f"Reading {num_reads} buffers of 8192 samples each...")
        print(f"{'='*70}\n")
        
        read_size = 8192  # Power-of-two
        buffer_size = read_size * 2 * 2  # 2 channels (I/Q) * 2 bytes (int16)
        buf = (ctypes.c_int16 * (read_size * 2))()  # I/Q interleaved
        
        all_raw_max = []
        all_raw_mean = []
        all_raw_std = []
        all_rail_frac = []
        all_normalized_mean = []
        all_normalized_max = []
        
        scale_q11 = 1.0 / 2048.0
        scale_int16 = 1.0 / 32768.0
        
        for i in range(num_reads):
            ret = _libbladerf.bladerf_sync_rx(dev, ctypes.byref(buf), read_size, None, 3500)
            if ret != 0:
                if ret == -1:  # BLADERF_ERR_TIMEOUT
                    print(f"  [WARN] Read {i+1}/{num_reads}: Timeout (not critical)")
                    continue
                else:
                    error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                    print(f"  [ERROR] Read {i+1}/{num_reads}: {error_str}")
                    continue
            
            # Convert to numpy array
            arr = np.frombuffer(buf, dtype=np.int16)
            
            # Extract I/Q (interleaved: I, Q, I, Q, ...)
            i_samples = arr[0::2]
            q_samples = arr[1::2]
            
            # Raw statistics
            raw = np.abs(arr)
            raw_max = int(raw.max())
            raw_min = int(raw.min())
            raw_mean = float(np.mean(raw))
            raw_std = float(np.std(raw))
            
            # Q11 rail detection
            rail_frac = float(np.mean(raw >= 2047))
            
            # Normalize with Q11 scaling
            i_norm = i_samples.astype(np.float32) * scale_q11
            q_norm = q_samples.astype(np.float32) * scale_q11
            iq_norm = i_norm + 1j * q_norm
            normalized_mean = float(np.mean(np.abs(iq_norm)))
            normalized_max = float(np.max(np.abs(iq_norm)))
            
            all_raw_max.append(raw_max)
            all_raw_mean.append(raw_mean)
            all_raw_std.append(raw_std)
            all_rail_frac.append(rail_frac)
            all_normalized_mean.append(normalized_mean)
            all_normalized_max.append(normalized_max)
            
            if (i + 1) % 10 == 0:
                print(f"  Read {i+1}/{num_reads}: raw_max={raw_max}, rail_frac={rail_frac*100:.2f}%, "
                      f"norm_mean={normalized_mean:.4f}, norm_max={normalized_max:.4f}")
        
        # Final statistics
        print(f"\n{'='*70}")
        print(f"Results (averaged over {num_reads} reads):")
        print(f"{'='*70}")
        print(f"Raw Samples (int16):")
        print(f"  Max:     {np.mean(all_raw_max):.1f} (range: {np.min(all_raw_max)} - {np.max(all_raw_max)})")
        print(f"  Mean:    {np.mean(all_raw_mean):.1f}")
        print(f"  Std:     {np.mean(all_raw_std):.1f}")
        print(f"  Rail fraction (>=2047): {np.mean(all_rail_frac)*100:.3f}%")
        print(f"\nNormalized Samples (Q11 scaling, 1/2048):")
        print(f"  Mean magnitude: {np.mean(all_normalized_mean):.4f}")
        print(f"  Max magnitude:  {np.mean(all_normalized_max):.4f}")
        print(f"\nQ11 Format Analysis:")
        print(f"  Expected range: [-2048, 2047]")
        print(f"  Actual max seen: {np.max(all_raw_max)}")
        if np.max(all_raw_max) > 2047:
            print(f"  ⚠️  WARNING: Values exceed Q11 range!")
        print(f"\nNoise Floor Estimate:")
        # Simple FFT to estimate noise floor
        print(f"  (Run with FFT analysis for dBFS estimate)")
        print(f"{'='*70}\n")
        
    finally:
        print(f"Closing device...")
        _libbladerf.bladerf_close(dev)
        print(f"[OK] Device closed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test bladeRF noise floor")
    parser.add_argument("--freq", type=float, default=915.0, help="Center frequency in MHz (default: 915.0)")
    parser.add_argument("--rate", type=float, default=30.0, help="Sample rate in MS/s (default: 30.0)")
    parser.add_argument("--gain", type=float, default=0.0, help="Gain in dB (default: 0.0)")
    parser.add_argument("--reads", type=int, default=100, help="Number of reads (default: 100)")
    
    args = parser.parse_args()
    
    test_noise_floor(
        center_freq_hz=int(args.freq * 1e6),
        sample_rate_sps=int(args.rate * 1e6),
        gain_db=args.gain,
        num_reads=args.reads
    )
