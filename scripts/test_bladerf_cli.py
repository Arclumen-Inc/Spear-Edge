#!/usr/bin/env python3
"""
Test bladeRF using official bladeRF CLI tools.
Captures samples and analyzes them to verify signal presence.
"""

import subprocess
import sys
import os
import numpy as np
import struct
from pathlib import Path

def run_bladerf_cli_command(cmd):
    """Run a bladeRF CLI command and return output."""
    try:
        result = subprocess.run(
            ['bladeRF-cli', '-e', cmd],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError:
        print("ERROR: bladeRF-cli not found. Is it installed?")
        return False, "", "bladeRF-cli not found"
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_bladerf_cli():
    """Test bladeRF using official CLI tools."""
    print("=" * 70)
    print("BladeRF CLI Test (Official bladeRF Tools)")
    print("=" * 70)
    
    # Check if bladeRF CLI is available
    print("\n[1] Checking bladeRF CLI availability...")
    success, stdout, stderr = run_bladerf_cli_command("version")
    if not success:
        print(f"    ✗ bladeRF CLI not available: {stderr}")
        print("    Install with: sudo apt install bladerf")
        return False
    print(f"    ✓ bladeRF CLI available")
    if stdout:
        print(f"    {stdout.strip()}")
    
    # Get device info
    print("\n[2] Getting device information...")
    success, stdout, stderr = run_bladerf_cli_command("print \"Device info\"")
    if success and stdout:
        print(f"    {stdout.strip()}")
    
    # Configure for VTX frequency
    center_freq = 5915_000_000  # 5915 MHz
    sample_rate = 40_000_000    # 40 MS/s
    bandwidth = 40_000_000      # 40 MHz
    gain = 40                   # 40 dB
    num_samples = 262144        # 256k samples
    
    print(f"\n[3] Configuring bladeRF...")
    print(f"    Center Frequency: {center_freq / 1e6:.3f} MHz")
    print(f"    Sample Rate: {sample_rate / 1e6:.2f} MS/s")
    print(f"    Bandwidth: {bandwidth / 1e6:.2f} MHz")
    print(f"    Gain: {gain} dB")
    print(f"    Samples: {num_samples}")
    
    # Build configuration commands
    config_commands = [
        f"set frequency rx {center_freq}",
        f"set samplerate rx {sample_rate}",
        f"set bandwidth rx {bandwidth}",
        f"set gain rx {gain}",
    ]
    
    for cmd in config_commands:
        success, stdout, stderr = run_bladerf_cli_command(cmd)
        if not success:
            print(f"    ✗ Failed: {cmd}")
            print(f"    Error: {stderr}")
            return False
        print(f"    ✓ {cmd}")
    
    # Capture samples to file
    output_file = "/tmp/bladerf_test.bin"
    print(f"\n[4] Capturing {num_samples} samples to {output_file}...")
    
    capture_cmd = f"rx config file={output_file} format=bin n={num_samples}"
    success, stdout, stderr = run_bladerf_cli_command(capture_cmd)
    if not success:
        print(f"    ✗ Failed to configure RX: {stderr}")
        return False
    print(f"    ✓ RX configured")
    
    # Start capture
    print("    Starting capture...")
    success, stdout, stderr = run_bladerf_cli_command("rx start")
    if not success:
        print(f"    ✗ Failed to start RX: {stderr}")
        return False
    print(f"    ✓ RX started")
    
    # Wait for capture to complete
    print("    Waiting for capture to complete...")
    success, stdout, stderr = run_bladerf_cli_command("rx wait")
    if not success:
        print(f"    ✗ Capture failed: {stderr}")
        return False
    print(f"    ✓ Capture completed")
    
    # Stop RX
    run_bladerf_cli_command("rx stop")
    
    # Check if file exists and has data
    if not os.path.exists(output_file):
        print(f"\n[5] ✗ ERROR: Output file {output_file} not found!")
        return False
    
    file_size = os.path.getsize(output_file)
    expected_size = num_samples * 4  # 2 bytes I + 2 bytes Q per sample
    print(f"\n[5] Analyzing captured data...")
    print(f"    File size: {file_size} bytes (expected: {expected_size} bytes)")
    
    if file_size < expected_size:
        print(f"    ⚠ WARNING: File size is smaller than expected!")
    
    # Read and analyze the binary file
    # Format: SC16 Q11 (signed 16-bit I/Q interleaved)
    try:
        with open(output_file, 'rb') as f:
            data = f.read()
        
        # Convert to numpy array (int16 I/Q pairs)
        samples_int16 = np.frombuffer(data, dtype=np.int16)
        
        # Reshape to complex (I/Q pairs)
        if len(samples_int16) % 2 != 0:
            samples_int16 = samples_int16[:-1]  # Remove last sample if odd
        
        i_samples = samples_int16[0::2].astype(np.float32)
        q_samples = samples_int16[1::2].astype(np.float32)
        
        # Scale from Q11 format: divide by 2048
        i_samples = i_samples / 2048.0
        q_samples = q_samples / 2048.0
        
        # Create complex array
        iq = i_samples + 1j * q_samples
        
        print(f"    ✓ Read {len(iq)} complex samples")
        
        # Calculate statistics
        power = np.mean(np.abs(iq) ** 2)
        power_db = 10 * np.log10(power + 1e-12)
        max_mag = np.max(np.abs(iq))
        mean_mag = np.mean(np.abs(iq))
        std_mag = np.std(np.abs(iq))
        
        print(f"\n[6] Sample Statistics:")
        print(f"    Power: {power_db:.2f} dB")
        print(f"    Max magnitude: {max_mag:.6f}")
        print(f"    Mean magnitude: {mean_mag:.6f}")
        print(f"    Std magnitude: {std_mag:.6f}")
        
        # FFT analysis
        print(f"\n[7] FFT Analysis (looking for signal at 5915 MHz)...")
        
        fft_sizes = [1024, 2048, 4096]
        for fft_size in fft_sizes:
            if len(iq) < fft_size:
                continue
            
            # Use first fft_size samples
            iq_segment = iq[:fft_size]
            
            # Apply window
            window = np.hanning(len(iq_segment))
            windowed = iq_segment * window
            window_sum = np.sum(window)
            
            # FFT
            fft = np.fft.fftshift(np.fft.fft(windowed))
            mag = np.abs(fft) / window_sum
            spec_db = 20.0 * np.log10(mag + 1e-12)
            
            # Frequency axis
            freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_segment), d=1.0 / sample_rate))
            freqs_hz = freqs + center_freq
            
            # Find peak
            peak_idx = np.argmax(spec_db)
            peak_power = spec_db[peak_idx]
            peak_freq = freqs_hz[peak_idx] / 1e6
            
            # Noise floor (10th percentile, excluding edges)
            exclude_pct = 0.05
            exclude_count = int(len(spec_db) * exclude_pct)
            center_spectrum = spec_db[exclude_count:-exclude_count] if exclude_count > 0 else spec_db
            noise_floor = np.percentile(center_spectrum, 10)
            
            # SNR
            snr = peak_power - noise_floor
            
            # Check if signal is near 5915 MHz
            target_freq = 5915.0
            freq_bin_width = (sample_rate / fft_size) / 1e6  # MHz per bin
            distance_from_target = abs(peak_freq - target_freq)
            
            # Check for wideband signal
            signal_threshold = noise_floor + 3.0
            signal_bins = np.sum(spec_db > signal_threshold)
            signal_pct = (signal_bins / len(spec_db)) * 100
            
            print(f"\n    FFT Size: {fft_size}")
            print(f"      Peak: {peak_power:.1f} dBFS @ {peak_freq:.3f} MHz (bin {peak_idx})")
            print(f"      Noise floor: {noise_floor:.1f} dBFS")
            print(f"      SNR: {snr:.1f} dB")
            print(f"      Distance from 5915 MHz: {distance_from_target:.3f} MHz")
            print(f"      Bin width: {freq_bin_width:.3f} MHz")
            print(f"      Signal detected near 5915 MHz: {distance_from_target < freq_bin_width * 10}")
            print(f"      Bins above noise+3dB: {signal_bins} ({signal_pct:.1f}%)")
            if signal_pct > 20:
                print(f"      → WIDEBAND SIGNAL DETECTED ({signal_pct:.1f}% of spectrum)")
            elif signal_pct > 5:
                print(f"      → Possible wideband signal ({signal_pct:.1f}% of spectrum)")
            else:
                print(f"      → Narrowband or no signal ({signal_pct:.1f}% of spectrum)")
        
        # Clean up
        try:
            os.remove(output_file)
            print(f"\n[8] Cleaned up {output_file}")
        except:
            pass
        
        print("\n" + "=" * 70)
        print("Test complete!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Exception analyzing data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bladerf_cli()
    sys.exit(0 if success else 1)
