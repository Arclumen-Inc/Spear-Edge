#!/usr/bin/env python3
"""
Test FFT normalization formulas to match SDR++ behavior.
Tests different normalization approaches and compares results.
"""

import ctypes
import numpy as np
import sys
import time

# Load libbladeRF
try:
    _libbladerf = ctypes.CDLL("/usr/local/lib/libbladeRF.so.2")
    print("[OK] Loaded libbladeRF.so.2")
except OSError:
    try:
        _libbladerf = ctypes.CDLL("libbladeRF.so.2")
        print("[OK] Loaded libbladeRF.so.2 from system")
    except OSError:
        print("[ERROR] Could not load libbladeRF")
        sys.exit(1)

# Constants
BLADERF_CHANNEL_RX = lambda ch: ((ch) << 1) | 0x0
BLADERF_FORMAT_SC16_Q11 = 0x0001
BLADERF_GAIN_MGC = 0

# Setup function signatures
_libbladerf.bladerf_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
_libbladerf.bladerf_open.restype = ctypes.c_int
_libbladerf.bladerf_close.argtypes = [ctypes.c_void_p]
_libbladerf.bladerf_close.restype = None
_libbladerf.bladerf_set_gain_mode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_libbladerf.bladerf_set_gain_mode.restype = ctypes.c_int
_libbladerf.bladerf_set_sample_rate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_libbladerf.bladerf_set_sample_rate.restype = ctypes.c_int
_libbladerf.bladerf_set_bandwidth.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_libbladerf.bladerf_set_bandwidth.restype = ctypes.c_int
_libbladerf.bladerf_set_frequency.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
_libbladerf.bladerf_set_frequency.restype = ctypes.c_int
_libbladerf.bladerf_set_gain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_libbladerf.bladerf_set_gain.restype = ctypes.c_int
_libbladerf.bladerf_get_gain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_libbladerf.bladerf_get_gain.restype = ctypes.c_int
_libbladerf.bladerf_enable_module.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
_libbladerf.bladerf_enable_module.restype = ctypes.c_int
_libbladerf.bladerf_sync_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
_libbladerf.bladerf_sync_config.restype = ctypes.c_int
_libbladerf.bladerf_sync_rx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
_libbladerf.bladerf_sync_rx.restype = ctypes.c_int
_libbladerf.bladerf_strerror.argtypes = [ctypes.c_int]
_libbladerf.bladerf_strerror.restype = ctypes.c_char_p

def test_fft_normalization(center_freq_hz=915000000, sample_rate_sps=20000000, gain_db=0, fft_size=4096, num_reads=10):
    """Test different FFT normalization formulas."""
    
    print("="*70)
    print("FFT Normalization Test")
    print("="*70)
    print(f"Frequency: {center_freq_hz/1e6:.3f} MHz")
    print(f"Sample Rate: {sample_rate_sps/1e6:.2f} MS/s")
    print(f"Gain: {gain_db} dB")
    print(f"FFT Size: {fft_size}")
    print(f"Reads: {num_reads}")
    print()
    
    # Open device
    dev_ptr = ctypes.c_void_p()
    ret = _libbladerf.bladerf_open(ctypes.byref(dev_ptr), None)
    if ret != 0:
        error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
        print(f"[ERROR] Failed to open device: {error_str}")
        return
    dev = dev_ptr.value
    print("[OK] Device opened")
    
    try:
        ch = BLADERF_CHANNEL_RX(0)
        
        # Configure device
        print("\n[1/6] Setting gain mode to MANUAL...")
        ret = _libbladerf.bladerf_set_gain_mode(dev, ch, BLADERF_GAIN_MGC)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  [ERROR] {error_str}")
            return
        print("  [OK]")
        
        print("[2/6] Setting sample rate...")
        actual_rate = ctypes.c_uint32()
        ret = _libbladerf.bladerf_set_sample_rate(dev, ch, sample_rate_sps, ctypes.byref(actual_rate))
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  [ERROR] {error_str}")
            return
        print(f"  [OK] {actual_rate.value/1e6:.3f} MS/s")
        
        print("[3/6] Setting bandwidth...")
        actual_bw = ctypes.c_uint32()
        ret = _libbladerf.bladerf_set_bandwidth(dev, ch, sample_rate_sps, ctypes.byref(actual_bw))
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  [ERROR] {error_str}")
            return
        print(f"  [OK] {actual_bw.value/1e6:.3f} MHz")
        
        print("[4/6] Setting frequency...")
        ret = _libbladerf.bladerf_set_frequency(dev, ch, center_freq_hz)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  [ERROR] {error_str}")
            return
        print("  [OK]")
        
        print("[5/6] Enabling RX module...")
        ret = _libbladerf.bladerf_enable_module(dev, ch, True)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  [ERROR] {error_str}")
            return
        print("  [OK]")
        
        print("[6/6] Configuring stream...")
        ret = _libbladerf.bladerf_sync_config(dev, ch, BLADERF_FORMAT_SC16_Q11, 16, 8192, 8, 3500, 0)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  [ERROR] {error_str}")
            return
        print("  [OK]")
        
        # Set gain AFTER stream setup
        print(f"\n[GAIN] Setting gain to {gain_db} dB...")
        gain_int = int(gain_db)
        ret = _libbladerf.bladerf_set_gain(dev, ch, gain_int)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            print(f"  [WARNING] {error_str}")
        else:
            time.sleep(0.05)
            gain_ptr = ctypes.c_int()
            ret_get = _libbladerf.bladerf_get_gain(dev, ch, ctypes.byref(gain_ptr))
            if ret_get == 0:
                applied = gain_ptr.value
                print(f"  [OK] Gain set to {applied} dB")
            else:
                print(f"  [OK] Gain set (verification failed)")
        
        # Wait for stream to stabilize
        print("\n[WAIT] Waiting for stream to stabilize...")
        time.sleep(0.5)
        
        # Read samples and test FFT normalization
        print(f"\n[READ] Reading {num_reads} buffers for FFT analysis...")
        print("="*70)
        
        # Window
        win = np.hanning(fft_size).astype(np.float32)
        win_energy = float(np.sum(win * win))
        coherent_gain = float(np.sum(win)) / fft_size
        
        print(f"Window: Hann, energy={win_energy:.2f}, coherent_gain={coherent_gain:.4f}")
        print()
        
        all_fft_results = {
            'mag_norm': [],
            'coherent_norm': [],  # SDR++ style: divide by sum(window)
            'power_norm': [],
            'power_win_norm': [],
        }
        
        read_size = 8192  # Power of two
        scale = 1.0 / 32768.0  # int16 scaling (matches settings default)
        
        for read_num in range(num_reads):
            # Read samples
            buf_size = read_size * 2  # I/Q pairs
            buf = (ctypes.c_int16 * buf_size)()
            ret = _libbladerf.bladerf_sync_rx(dev, buf, read_size, None, 250)
            if ret != 0:
                print(f"  Read {read_num+1}/{num_reads}: ERROR (code {ret})")
                continue
            
            # Convert to complex
            arr = np.frombuffer(buf, dtype=np.int16)
            i = arr[0::2].astype(np.float32) * scale
            q = arr[1::2].astype(np.float32) * scale
            iq = (i + 1j * q).astype(np.complex64)
            
            # Use first fft_size samples
            if len(iq) < fft_size:
                continue
            iq = iq[:fft_size]
            
            # Remove DC
            iq = iq - np.mean(iq)
            
            # Window and FFT
            x = iq * win
            X = np.fft.fftshift(np.fft.fft(x, n=fft_size))
            
            # Test different normalizations
            eps = 1e-12
            window_sum = float(np.sum(win))  # Sum of window for coherent gain normalization
            
            # 1. Magnitude normalization (GNU Radio standard) - divide by N
            mag = np.abs(X) / fft_size
            spec_db_mag = 20.0 * np.log10(mag + eps)
            noise_floor_mag = float(np.percentile(spec_db_mag, 10))
            all_fft_results['mag_norm'].append(noise_floor_mag)
            
            # 2. Coherent gain normalization (SDR++ style) - divide by sum(window)
            mag_coherent = np.abs(X) / window_sum
            spec_db_coherent = 20.0 * np.log10(mag_coherent + eps)
            noise_floor_coherent = float(np.percentile(spec_db_coherent, 10))
            all_fft_results['coherent_norm'].append(noise_floor_coherent)
            
            # 3. Power normalization (no window energy)
            P = (np.abs(X) ** 2) / fft_size
            P = np.clip(P, eps, None)
            spec_db_power = 10.0 * np.log10(P)
            noise_floor_power = float(np.percentile(spec_db_power, 10))
            all_fft_results['power_norm'].append(noise_floor_power)
            
            # 4. Power normalization with window energy
            P_win = (np.abs(X) ** 2) / (fft_size * win_energy)
            P_win = np.clip(P_win, eps, None)
            spec_db_power_win = 10.0 * np.log10(P_win)
            noise_floor_power_win = float(np.percentile(spec_db_power_win, 10))
            all_fft_results['power_win_norm'].append(noise_floor_power_win)
            
            if (read_num + 1) % 5 == 0:
                print(f"  Read {read_num+1}/{num_reads}: mag={noise_floor_mag:.1f} dBFS, "
                      f"coherent={noise_floor_coherent:.1f} dBFS, "
                      f"power={noise_floor_power:.1f} dBFS, power_win={noise_floor_power_win:.1f} dBFS")
        
        # Results
        print("\n" + "="*70)
        print("Results (averaged over all reads):")
        print("="*70)
        
        method_names = {
            'mag_norm': 'Magnitude norm (divide by N)',
            'coherent_norm': 'Coherent gain norm (SDR++ style, divide by sum(window))',
            'power_norm': 'Power norm (divide by N)',
            'power_win_norm': 'Power+window norm (divide by N*win_energy)',
        }
        
        for method, values in all_fft_results.items():
            if values:
                mean_floor = np.mean(values)
                std_floor = np.std(values)
                method_label = method_names.get(method, method)
                print(f"\n{method_label}:")
                print(f"  Noise floor: {mean_floor:.1f} ± {std_floor:.1f} dBFS")
                print(f"  Range: {np.min(values):.1f} to {np.max(values):.1f} dBFS")
                print(f"  Target: -95 dBFS")
                print(f"  Difference: {mean_floor - (-95):.1f} dB")
        
        # Theoretical calculations
        print("\n" + "="*70)
        print("Theoretical Noise Floors:")
        print("="*70)
        window_sum = float(np.sum(win))
        print(f"Magnitude norm (divide by N): -10*log10({fft_size}) = {-10*np.log10(fft_size):.1f} dBFS")
        print(f"Coherent norm (divide by sum(window)={window_sum:.1f}): 20*log10(2) - 10*log10({fft_size}) = {20*np.log10(2) - 10*np.log10(fft_size):.1f} dBFS")
        print(f"Power norm: -10*log10({fft_size}) = {-10*np.log10(fft_size):.1f} dBFS")
        print(f"Power+win norm: -10*log10({win_energy}) = {-10*np.log10(win_energy):.1f} dBFS")
        print(f"\nTarget (SDR++): -95 dBFS")
        
    finally:
        print("\n[CLOSE] Closing device...")
        _libbladerf.bladerf_close(dev)
        print("[OK] Device closed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test FFT normalization formulas")
    parser.add_argument("--freq", type=float, default=915.0, help="Center frequency in MHz (default: 915.0)")
    parser.add_argument("--rate", type=float, default=20.0, help="Sample rate in MS/s (default: 20.0)")
    parser.add_argument("--gain", type=float, default=0.0, help="Gain in dB (default: 0.0)")
    parser.add_argument("--fft", type=int, default=4096, help="FFT size (default: 4096)")
    parser.add_argument("--reads", type=int, default=10, help="Number of reads (default: 10)")
    
    args = parser.parse_args()
    
    test_fft_normalization(
        center_freq_hz=int(args.freq * 1e6),
        sample_rate_sps=int(args.rate * 1e6),
        gain_db=args.gain,
        fft_size=args.fft,
        num_reads=args.reads
    )
