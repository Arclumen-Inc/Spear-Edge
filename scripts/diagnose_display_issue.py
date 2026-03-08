#!/usr/bin/env python3
"""
Diagnose why VTX signal isn't visible in Edge UI despite being present in data.
Compares raw FFT with Edge's processed output.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice
from spear_edge.core.sdr.base import SdrConfig, GainMode
from spear_edge.core.scan.scan_task import ScanTask
from spear_edge.core.scan.ring_buffer import IQRingBuffer

def diagnose_display():
    """Diagnose display pipeline issue."""
    print("=" * 80)
    print("EDGE DISPLAY PIPELINE DIAGNOSTIC")
    print("=" * 80)
    
    center_freq = 5910_000_000
    sample_rate = 40_000_000
    bandwidth = 40_000_000
    gain = 12.0
    fft_size = 65536
    
    sdr = BladeRFNativeDevice()
    
    try:
        sdr.open()
        config = SdrConfig(
            center_freq_hz=center_freq,
            sample_rate_sps=sample_rate,
            bandwidth_hz=bandwidth,
            gain_mode=GainMode.MANUAL,
            gain_db=gain,
            rx_channel=0,
        )
        sdr.apply_config(config)
        time.sleep(0.5)
        
        print(f"\n[1] Reading samples...")
        read_size = 196608
        all_samples = []
        for i in range(3):
            iq = sdr.read_samples(read_size)
            if len(iq) > 0:
                all_samples.append(iq)
        combined_iq = np.concatenate(all_samples)
        print(f"  Total: {len(combined_iq)} samples")
        
        print(f"\n[2] Testing Edge's FFT processing pipeline...")
        
        # Create Edge's processing pipeline
        ring = IQRingBuffer(int(sample_rate * 0.3))
        for i in range(0, len(combined_iq), 262144):
            chunk = combined_iq[i:i+262144]
            if len(chunk) > 0:
                ring.push(chunk)
        
        scan_task = ScanTask(
            ring=ring,
            center_freq_hz=center_freq,
            sample_rate_sps=sample_rate,
            fft_size=fft_size,
            fps=15.0,
            calibration_offset_db=0.0
        )
        
        # Subscribe to frames to see what Edge sends to UI
        frames_received = []
        def frame_handler(frame):
            frames_received.append(frame)
        
        scan_task.subscribe(frame_handler)
        
        # Process a few frames
        print(f"  Processing 5 frames through Edge's pipeline...")
        for i in range(5):
            iq = ring.pop(fft_size)
            if iq.size < fft_size:
                break
            
            # Edge's internal processing (simplified)
            work_iq = iq[:fft_size].copy()
            
            # DC removal (disabled by default)
            from spear_edge.settings import settings
            if settings.DC_REMOVAL:
                dc_offset = np.mean(work_iq)
                work_iq -= dc_offset
            
            # Window and FFT
            window = np.hanning(fft_size).astype(np.float32)
            windowed = work_iq * window
            window_sum = np.sum(window)
            
            fft = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
            mag = np.abs(fft) / window_sum
            spec_db = 20.0 * np.log10(mag + 1e-12)
            
            # Edge's noise floor and smoothing
            exclude_pct = 0.05
            exclude_count = int(len(spec_db) * exclude_pct)
            center_spectrum = spec_db[exclude_count:-exclude_count]
            
            noise_floor_2nd = np.percentile(center_spectrum, 2)
            signal_threshold = noise_floor_2nd + 3.0
            signal_bins = np.sum(center_spectrum > signal_threshold)
            signal_pct = (signal_bins / len(center_spectrum)) * 100
            is_wideband = signal_pct > 20.0
            
            noise_floor = noise_floor_2nd if is_wideband else np.percentile(center_spectrum, 10)
            
            # Smoothing
            if scan_task._fft_smoothed is None:
                scan_task._fft_smoothed = spec_db.astype(np.float32)
            else:
                alpha = scan_task._fft_smooth_alpha
                scan_task._fft_smoothed = (alpha * spec_db + (1.0 - alpha) * scan_task._fft_smoothed).astype(np.float32)
            
            # Edge bin zeroing
            edge_zero_pct = 0.025
            edge_zero_count = int(len(scan_task._fft_smoothed) * edge_zero_pct)
            if edge_zero_count > 0:
                edge_zero_value = noise_floor - 10.0
                scan_task._fft_smoothed[:edge_zero_count] = edge_zero_value
                scan_task._fft_smoothed[-edge_zero_count:] = edge_zero_value
            
            # Create frame (what Edge sends to UI)
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
            freqs_hz = freqs + center_freq
            
            frame = {
                'power_dbfs': scan_task._fft_smoothed.tolist(),  # Smoothed for FFT line
                'power_inst_dbfs': spec_db.tolist(),  # Instant for waterfall
                'freqs_hz': freqs_hz.tolist(),
                'noise_floor_dbfs': noise_floor,
                'center_freq_hz': center_freq,
            }
            
            frame_handler(frame)
            
            peak_idx = np.argmax(scan_task._fft_smoothed)
            peak_power = scan_task._fft_smoothed[peak_idx]
            peak_freq = freqs_hz[peak_idx] / 1e6
            
            print(f"    Frame {i+1}: peak={peak_power:.1f} dBFS @ {peak_freq:.3f} MHz, "
                  f"floor={noise_floor:.1f} dBFS, SNR={peak_power - noise_floor:.1f} dB")
        
        if frames_received:
            print(f"\n[3] ANALYZING WHAT EDGE SENDS TO UI...")
            last_frame = frames_received[-1]
            
            power_dbfs = np.array(last_frame['power_dbfs'])
            power_inst = np.array(last_frame['power_inst_dbfs'])
            noise_floor = last_frame['noise_floor_dbfs']
            
            print(f"\n  FFT Data (power_dbfs - what UI uses for FFT line):")
            print(f"    Length: {len(power_dbfs)}")
            print(f"    Min: {np.min(power_dbfs):.1f} dBFS")
            print(f"    Max: {np.max(power_dbfs):.1f} dBFS")
            print(f"    Mean: {np.mean(power_dbfs):.1f} dBFS")
            print(f"    Noise floor: {noise_floor:.1f} dBFS")
            
            peak_idx = np.argmax(power_dbfs)
            peak_power = power_dbfs[peak_idx]
            freqs_hz = np.array(last_frame['freqs_hz'])
            peak_freq = freqs_hz[peak_idx] / 1e6
            
            print(f"    Peak: {peak_power:.1f} dBFS @ {peak_freq:.3f} MHz (bin {peak_idx})")
            print(f"    SNR: {peak_power - noise_floor:.1f} dB")
            
            # Check UI display range
            print(f"\n  UI Display Range Calculation:")
            print(f"    Noise floor: {noise_floor:.1f} dBFS")
            
            # UI logic: 35 dB range if floor < -75, else 70 dB
            ui_range = 35.0 if noise_floor < -75.0 else 70.0
            ui_min = noise_floor
            ui_max = noise_floor + ui_range
            
            # Clamp to reference level
            if ui_max > -20.0:
                ui_max = -20.0
                ui_min = ui_max - ui_range
            
            print(f"    Range: {ui_range} dB")
            print(f"    Min: {ui_min:.1f} dBFS")
            print(f"    Max: {ui_max:.1f} dBFS")
            print(f"    Peak position: {peak_power:.1f} dBFS")
            
            if peak_power > ui_max:
                print(f"    ⚠️  PEAK IS ABOVE DISPLAY RANGE!")
            elif peak_power < ui_min:
                print(f"    ⚠️  PEAK IS BELOW DISPLAY RANGE!")
            else:
                peak_pos_pct = ((peak_power - ui_min) / (ui_max - ui_min)) * 100
                print(f"    ✓ Peak is in range ({peak_pos_pct:.1f}% from bottom)")
                if peak_pos_pct < 5:
                    print(f"    ⚠️  Peak is very close to bottom - might appear flat!")
            
            # Check if signal is spread out (wideband)
            signal_above_floor = power_dbfs > (noise_floor + 3.0)
            signal_bins_count = np.sum(signal_above_floor)
            signal_pct = (signal_bins_count / len(power_dbfs)) * 100
            
            print(f"\n  Wideband Signal Analysis:")
            print(f"    Bins > noise+3dB: {signal_bins_count} ({signal_pct:.1f}%)")
            print(f"    If wideband, signal power is spread across many bins")
            print(f"    Individual bin peaks may be low even if total power is high")
            
            # Show top 10 bins
            print(f"\n  Top 10 Bins (after Edge processing):")
            top_indices = np.argsort(power_dbfs)[::-1][:10]
            for i, idx in enumerate(top_indices):
                freq_mhz = freqs_hz[idx] / 1e6
                print(f"    {i+1}. {power_dbfs[idx]:.1f} dBFS @ {freq_mhz:.3f} MHz (bin {idx})")
        
        print("\n" + "=" * 80)
        print("Diagnostic complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sdr.close()

if __name__ == "__main__":
    diagnose_display()
