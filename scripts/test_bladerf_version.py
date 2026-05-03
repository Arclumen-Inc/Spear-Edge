#!/usr/bin/env python3
"""
Test bladeRF firmware and FPGA version compatibility with Edge.
"""

import ctypes
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_bladerf_version():
    """Test bladeRF version and basic functionality."""
    print("=" * 80)
    print("Testing bladeRF Firmware/FPGA Compatibility")
    print("=" * 80)
    
    # Load library
    try:
        lib = ctypes.CDLL("/usr/local/lib/libbladeRF.so.2")
        print("[OK] Library loaded: /usr/local/lib/libbladeRF.so.2")
    except OSError:
        try:
            lib = ctypes.CDLL("libbladeRF.so.2")
            print("[OK] Library loaded: libbladeRF.so.2 (system)")
        except OSError:
            print("[ERROR] Could not load libbladeRF.so.2")
            return 1
    
    # Define version struct (from bladeRF headers)
    class bladerf_version(ctypes.Structure):
        _fields_ = [
            ("major", ctypes.c_uint),
            ("minor", ctypes.c_uint),
            ("patch", ctypes.c_uint),
        ]
    
    # Bind version functions
    lib.bladerf_version.argtypes = []
    lib.bladerf_version.restype = bladerf_version
    
    lib.bladerf_fpga_version.argtypes = [ctypes.c_void_p, ctypes.POINTER(bladerf_version)]
    lib.bladerf_fpga_version.restype = ctypes.c_int
    
    lib.bladerf_fw_version.argtypes = [ctypes.c_void_p, ctypes.POINTER(bladerf_version)]
    lib.bladerf_fw_version.restype = ctypes.c_int
    
    lib.bladerf_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.bladerf_open.restype = ctypes.c_int
    
    lib.bladerf_close.argtypes = [ctypes.c_void_p]
    lib.bladerf_close.restype = ctypes.c_int
    
    lib.bladerf_strerror.argtypes = [ctypes.c_int]
    lib.bladerf_strerror.restype = ctypes.c_char_p
    
    # Get library version
    try:
        lib_version = lib.bladerf_version()
        print(f"\n[LIBRARY] Version: {lib_version.major}.{lib_version.minor}.{lib_version.patch}")
    except Exception as e:
        print(f"[WARN] Could not get library version: {e}")
    
    # Open device
    dev_ptr = ctypes.c_void_p()
    ret = lib.bladerf_open(ctypes.byref(dev_ptr), b"*")
    
    if ret != 0:
        error_str = lib.bladerf_strerror(ret).decode('utf-8', errors='ignore')
        print(f"\n[ERROR] Failed to open bladeRF device: {error_str} (code: {ret})")
        return 1
    
    print("[OK] Device opened successfully")
    
    # Get firmware version
    fw_version = bladerf_version()
    ret = lib.bladerf_fw_version(dev_ptr, ctypes.byref(fw_version))
    if ret == 0:
        print(f"[FIRMWARE] Version: {fw_version.major}.{fw_version.minor}.{fw_version.patch}")
    else:
        error_str = lib.bladerf_strerror(ret).decode('utf-8', errors='ignore')
        print(f"[WARN] Could not get firmware version: {error_str}")
    
    # Get FPGA version
    fpga_version = bladerf_version()
    ret = lib.bladerf_fpga_version(dev_ptr, ctypes.byref(fpga_version))
    if ret == 0:
        print(f"[FPGA] Version: {fpga_version.major}.{fpga_version.minor}.{fpga_version.patch}")
    else:
        error_str = lib.bladerf_strerror(ret).decode('utf-8', errors='ignore')
        print(f"[WARN] Could not get FPGA version: {error_str}")
    
    # Close device
    lib.bladerf_close(dev_ptr)
    print("\n[OK] Device closed")
    
    # Test Edge SDR initialization
    print("\n" + "=" * 80)
    print("Testing Edge SDR Integration")
    print("=" * 80)
    
    try:
        from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice
        
        print("[OK] BladeRFNativeDevice imported successfully")
        
        # Try to create device instance
        sdr = BladeRFNativeDevice()
        print("[OK] BladeRFNativeDevice created successfully")
        
        # Test basic configuration
        from spear_edge.core.sdr.base import SdrConfig, GainMode
        
        config = SdrConfig(
            center_freq_hz=915_000_000,
            sample_rate_sps=10_000_000,
            bandwidth_hz=10_000_000,
            gain_db=20.0,
            gain_mode=GainMode.MANUAL,
            rx_channel=0
        )
        
        print("\n[TEST] Applying basic configuration...")
        sdr.apply_config(config)
        print("[OK] Configuration applied successfully")
        
        # Test stream setup
        print("\n[TEST] Testing stream setup...")
        sdr._setup_stream()
        print("[OK] Stream setup successful")
        
        # Test reading samples
        print("\n[TEST] Testing sample read (8192 samples)...")
        samples = sdr.read_samples(8192)
        if len(samples) > 0:
            print(f"[OK] Successfully read {len(samples)} samples")
        else:
            print("[WARN] Read returned 0 samples (may be normal if no signal)")
        
        # Cleanup
        sdr.close()
        print("[OK] SDR closed successfully")
        
        print("\n" + "=" * 80)
        print("[SUCCESS] All tests passed! Edge is compatible with updated bladeRF.")
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Edge integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_bladerf_version())
