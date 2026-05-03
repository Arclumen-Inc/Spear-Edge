#!/usr/bin/env python3
"""
Diagnose frequency readback bug and Edge display issues.
Compares raw libbladeRF data with Edge's processing pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import ctypes
from spear_edge.core.sdr.bladerf_native import (
    BladeRFNativeDevice, 
    _libbladerf,
    BLADERF_CHANNEL_RX
)
from spear_edge.core.sdr.base import SdrConfig, GainMode
from spear_edge.core.scan.scan_task import ScanTask
from spear_edge.core.scan.ring_buffer import IQRingBuffer

def diagnose_freq_and_display():
    """Diagnose frequency readback and display pipeline."""
    print("=" * 80)
    print("FREQUENCY READBACK & DISPLAY DIAGNOSTIC")
    print("=" * 80)
    
    # Edge's exact configuration
    center_freq = 5910_000_000  # 5910 MHz
    sample_rate = 40_000_000    # 40 MS/s
    bandwidth = 40_000_000      # 40 MHz
    gain = 12.0                 # 12 dB
    fft_size = 65536
    rx_channel = 0
    
    print(f"\n[CONFIG] Edge Configuration:")
    print(f"  Center Frequency: {center_freq / 1e6:.3f} MHz")
    print(f"  Sample Rate: {sample_rate / 1e6:.2f} MS/s")
    print(f"  Bandwidth: {bandwidth / 1e6:.2f} MHz")
    print(f"  Gain: {gain} dB")
    print(f"  FFT Size: {fft_size}")
    
    sdr = BladeRFNativeDevice()
    
    try:
        print(f"\n[1] Opening bladeRF device...")
        sdr.open()
        print("    ✓ Device opened")
        
        print(f"\n[2] Applying Edge configuration...")
        config = SdrConfig(
            center_freq_hz=center_freq,
            sample_rate_sps=sample_rate,
            bandwidth_hz=bandwidth,
            gain_mode=GainMode.MANUAL,
            gain_db=gain,
            rx_channel=rx_channel,
        )
        sdr.apply_config(config)
        print("    ✓ Configuration applied")
        time.sleep(0.5)
        
        print(f"\n[3] FREQUENCY READBACK INVESTIGATION...")
        ch = BLADERF_CHANNEL_RX(rx_channel)
        
        # Try different readback methods
        print(f"\n  Method 1: Using ctypes.c_uint64 (current method)...")
        actual_freq_1 = ctypes.c_uint64()
        ret1 = _libbladerf.bladerf_get_frequency(sdr.dev, ch, ctypes.byref(actual_freq_1))
        if ret1 == 0:
            print(f"    Result: {actual_freq_1.value / 1e6:.6f} MHz (return code: {ret1})")
            print(f"    Raw value: {actual_freq_1.value}")
        else:
            error_str = _libbladerf.bladerf_strerror(ret1).decode('utf-8', errors='ignore')
            print(f"    ERROR: {error_str} (code: {ret1})")
        
        print(f"\n  Method 2: Using ctypes.POINTER(ctypes.c_uint64)...")
        actual_freq_2 = (ctypes.c_uint64 * 1)()
        ret2 = _libbladerf.bladerf_get_frequency(sdr.dev, ch, actual_freq_2)
        if ret2 == 0:
            print(f"    Result: {actual_freq_2[0] / 1e6:.6f} MHz (return code: {ret2})")
            print(f"    Raw value: {actual_freq_2[0]}")
        else:
            error_str = _libbladerf.bladerf_strerror(ret2).decode('utf-8', errors='ignore')
            print(f"    ERROR: {error_str} (code: {ret2})")
        
        print(f"\n  Method 3: Using ctypes.c_uint32 (wrong type, but test)...")
        actual_freq_3 = ctypes.c_uint32()
        ret3 = _libbladerf.bladerf_get_frequency(sdr.dev, ch, ctypes.byref(actual_freq_3))
        if ret3 == 0:
            print(f"    Result: {actual_freq_3.value / 1e6:.6f} MHz (return code: {ret3})")
            print(f"    Raw value: {actual_freq_3.value}")
            print(f"    ⚠️  This is uint32 - might truncate high frequencies!")
        else:
            error_str = _libbladerf.bladerf_strerror(ret3).decode('utf-8', errors='ignore')
            print(f"    ERROR: {error_str} (code: {ret3})")
        
        # Check internal state
        print(f"\n  Internal state (sdr.center_freq_hz): {sdr.center_freq_hz / 1e6:.6f} MHz")
        
        if not sdr._stream_active:
            print("\n[ERROR] Stream not active!")
            return
        
        print(f"\n[4] READING SAMPLES AND TESTING EDGE PROCESSING PIPELINE...")
        read_size = 196608
        print(f"  Reading {read_size} samples...")
        
        all_samples = []
        for i in range(3):
            iq = sdr.read_samples(read_size)
            if len(iq) > 0:
                all_samples.append(iq)
                print(f"    Chunk {i+1}: {len(iq)} samples")
        
        if not all_samples:
            print("\n[ERROR] No samples received!")
            return
        
        combined_iq = np.concatenate(all_samples)
        print(f"  Total samples: {len(combined_iq)}")
        
        # Test Edge's processing pipeline
        print(f"\n[5] TESTING EDGE'S FFT PROCESSING PIPELINE...")
        
        # Create ring buffer and scan task (matching Edge)
        ring_size = int(sample_rate * 0.3)  # 0.3 seconds
        ring = IQRingBuffer(ring_size)
        
        # Push samples to ring
        chunk_size = min(len(combined_iq), 262144)
        for i in range(0, len(combined_iq), chunk_size):
            chunk = combined_iq[i:i+chunk_size]
            if len(chunk) > 0:
                ring.push(chunk)
        
        # Create scan task (matching Edge)
        scan_task = ScanTask(
            ring=ring,
            center_freq_hz=center_freq,
            sample_rate_sps=sample_rate,
            fft_size=fft_size,
            fps=15.0,
            calibration_offset_db=0.0
        )
        
        # Process frames (matching Edge's loop)
        print(f"  Processing FFT frames (matching Edge's scan_task)...")
        
        frames_processed = []
        for frame_num in range(5):
            iq = ring.pop(fft_size)
            if iq.size < fft_size:
                print(f"    Frame {frame_num+1}: Not enough samples ({iq.size} < {fft_size})")
                break
            
            # This is what Edge does internally - we'll simulate it
            # Edge uses _work_iq, but we'll do it directly
            work_iq = iq[:fft_size].copy()
            
            # DC removal (Edge has it disabled by default)
            from spear_edge.settings import settings
            if settings.DC_REMOVAL:
                dc_offset = np.mean(work_iq)
                work_iq -= dc_offset
            
            # Window and FFT (matching Edge exactly)
            window = np.hanning(fft_size).astype(np.float32)
            windowed = work_iq * window
            window_sum = np.sum(window)
            
            fft = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
            mag = np.abs(fft) / window_sum
            spec_db = 20.0 * np.log10(mag + 1e-12)
            
            # Edge's noise floor calculation (wideband detection)
            exclude_pct = 0.05
            exclude_count = int(len(spec_db) * exclude_pct)
            center_spectrum = spec_db[exclude_count:-exclude_count]
            
            # Wideband detection (matching Edge)
            noise_floor_2nd = np.percentile(center_spectrum, 2)
            signal_threshold = noise_floor_2nd + 3.0
            signal_bins = np.sum(center_spectrum > signal_threshold)
            signal_pct = (signal_bins / len(center_spectrum)) * 100
            is_wideband = signal_pct > 20.0
            
            if is_wideband:
                noise_floor = np.percentile(center_spectrum, 2)  # 2nd percentile for wideband
            else:
                noise_floor = np.percentile(center_spectrum, 10)  # 10th percentile for narrowband
            
            # Edge's smoothing
            if scan_task._fft_smoothed is None:
                scan_task._fft_smoothed = spec_db.astype(np.float32)
            else:
                alpha = scan_task._fft_smooth_alpha
                scan_task._fft_smoothed = (alpha * spec_db + (1.0 - alpha) * scan_task._fft_smoothed).astype(np.float32)
            
            # Edge's edge bin zeroing
            edge_zero_pct = 0.025
            edge_zero_count = int(len(scan_task._fft_smoothed) * edge_zero_pct)
            if edge_zero_count > 0:
                edge_zero_value = noise_floor - 10.0
                scan_task._fft_smoothed[:edge_zero_count] = edge_zero_value
                scan_task._fft_smoothed[-edge_zero_count:] = edge_zero_value
            
            # Find peak
            peak_idx = np.argmax(scan_task._fft_smoothed)
            peak_power = scan_task._fft_smoothed[peak_idx]
            
            # Frequency axis
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
            freqs_hz = freqs + center_freq
            peak_freq = freqs_hz[peak_idx] / 1e6
            
            frames_processed.append({
                'frame': frame_num + 1,
                'peak_power': peak_power,
                'peak_freq': peak_freq,
                'peak_bin': peak_idx,
                'noise_floor': noise_floor,
                'snr': peak_power - noise_floor,
                'is_wideband': is_wideband,
                'signal_pct': signal_pct
            })
            
            print(f"    Frame {frame_num+1}: peak={peak_power:.1f} dBFS @ {peak_freq:.3f} MHz, "
                  f"floor={noise_floor:.1f} dBFS, SNR={peak_power - noise_floor:.1f} dB, "
                  f"wideband={is_wideband}")
        
        if frames_processed:
            print(f"\n[6] EDGE PROCESSING SUMMARY:")
            avg_peak = np.mean([f['peak_power'] for f in frames_processed])
            avg_floor = np.mean([f['noise_floor'] for f in frames_processed])
            avg_snr = np.mean([f['snr'] for f in frames_processed])
            
            print(f"  Average peak: {avg_peak:.1f} dBFS")
            print(f"  Average noise floor: {avg_floor:.1f} dBFS")
            print(f"  Average SNR: {avg_snr:.1f} dB")
            print(f"  Wideband: {frames_processed[0]['is_wideband']}")
            print(f"  Signal bins: {frames_processed[0]['signal_pct']:.1f}%")
            
            # Check if signal would be visible in UI
            # UI uses 70 dB range by default, or 35 dB for wideband when floor < -75 dBFS
            ui_range = 35.0 if avg_floor < -75.0 else 70.0
            ui_max = avg_floor + ui_range
            ui_min = avg_floor
            
            print(f"\n  UI Display Range (estimated):")
            print(f"    Range: {ui_range} dB")
            print(f"    Min: {ui_min:.1f} dBFS")
            print(f"    Max: {ui_max:.1f} dBFS")
            print(f"    Peak position: {avg_peak:.1f} dBFS")
            
            if avg_peak > ui_max:
                print(f"    ⚠️  PEAK IS ABOVE DISPLAY RANGE (clipped at top)")
            elif avg_peak < ui_min:
                print(f"    ⚠️  PEAK IS BELOW DISPLAY RANGE (not visible)")
            else:
                peak_position_pct = ((avg_peak - ui_min) / (ui_max - ui_min)) * 100
                print(f"    ✓ Peak is in display range ({peak_position_pct:.1f}% from bottom)")
                if peak_position_pct < 10:
                    print(f"    ⚠️  Peak is very close to bottom - might appear flat!")
        
        print("\n" + "=" * 80)
        print("Diagnostic complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[7] Closing device...")
        sdr.close()
        print("    ✓ Device closed")

if __name__ == "__main__":
    diagnose_freq_and_display()
