# spear_edge/core/sdr/bladerf_native.py

from __future__ import annotations

import ctypes
import logging
import time
import numpy as np
from typing import Optional, Dict, Any

from .base import SDRBase, GainMode

logger = logging.getLogger(__name__)

# Try to load libbladerf
_libbladerf = None
try:
    _libbladerf = ctypes.CDLL("/usr/local/lib/libbladeRF.so.2")
    logger.debug("Loaded libbladeRF.so.2 from /usr/local/lib")
except OSError:
    try:
        _libbladerf = ctypes.CDLL("libbladeRF.so.2")
        logger.debug("Loaded libbladeRF.so.2 from system")
    except OSError:
        try:
            _libbladerf = ctypes.CDLL("libbladeRF.so")
            logger.debug("Loaded libbladeRF.so from system")
        except OSError:
            logger.warning("Could not load libbladeRF - native backend will not be available")

# libbladeRF constants (from Tripwire v2.0 pattern)
# Channel encoding: Use header macro (matches installed library)
# BLADERF_CHANNEL_RX(ch) = ((ch) << 1) | 0x0 → gives 0, 2
def BLADERF_CHANNEL_RX(ch: int) -> int:
    """Encode RX channel: ((ch) << 1) | 0x0 (matches header macro)"""
    return ((ch) << 1) | 0x0

def BLADERF_CHANNEL_TX(ch: int) -> int:
    """Encode TX channel: ((ch) << 1) | 0x1 (matches header macro)"""
    return ((ch) << 1) | 0x1

BLADERF_CHANNEL_RX1 = BLADERF_CHANNEL_RX(0)  # = 0
BLADERF_CHANNEL_RX2 = BLADERF_CHANNEL_RX(1)  # = 2

# Format and layout constants
BLADERF_FORMAT_SC16_Q11 = 0x0001
# Channel layouts for bladerf_sync_config() - use header enum values (verified working)
BLADERF_RX_X1 = 0  # Single RX channel
BLADERF_RX_X2 = 2  # Dual RX channels (MIMO) - not used in Edge, but for reference

# Gain modes
BLADERF_GAIN_MGC = 0  # Manual gain control (disable AGC)
BLADERF_GAIN_AGC = 1  # Automatic gain control

# Error codes (common ones)
BLADERF_ERR_INVAL = -6
BLADERF_ERR_TIMEOUT = -1

# Setup function signatures if libbladerf is available
if _libbladerf is not None:
    # Device management
    _libbladerf.bladerf_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    _libbladerf.bladerf_open.restype = ctypes.c_int
    
    _libbladerf.bladerf_close.argtypes = [ctypes.c_void_p]
    _libbladerf.bladerf_close.restype = None
    
    # Configuration
    _libbladerf.bladerf_set_gain_mode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _libbladerf.bladerf_set_gain_mode.restype = ctypes.c_int
    
    _libbladerf.bladerf_set_sample_rate.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
    ]
    _libbladerf.bladerf_set_sample_rate.restype = ctypes.c_int
    
    _libbladerf.bladerf_set_bandwidth.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
    ]
    _libbladerf.bladerf_set_bandwidth.restype = ctypes.c_int
    
    _libbladerf.bladerf_set_frequency.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    _libbladerf.bladerf_set_frequency.restype = ctypes.c_int
    
    _libbladerf.bladerf_set_gain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _libbladerf.bladerf_set_gain.restype = ctypes.c_int
    
    # Stream management
    _libbladerf.bladerf_sync_config.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint
    ]
    _libbladerf.bladerf_sync_config.restype = ctypes.c_int
    
    _libbladerf.bladerf_enable_module.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
    _libbladerf.bladerf_enable_module.restype = ctypes.c_int
    
    # Data reading
    _libbladerf.bladerf_sync_rx.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), 
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint
    ]
    _libbladerf.bladerf_sync_rx.restype = ctypes.c_int
    
    # Error handling
    _libbladerf.bladerf_strerror.argtypes = [ctypes.c_int]
    _libbladerf.bladerf_strerror.restype = ctypes.c_char_p


class BladeRFNativeDevice(SDRBase):
    """
    Native libbladerf implementation for bladeRF 2.0 micro/xA4.
    
    Replaces SoapySDRDevice with direct libbladerf calls.
    Maintains SDRBase interface for compatibility.
    
    HARD RULES:
      - read_samples() NEVER throws (returns empty on timeout/error)
      - Methods match SDRBase contract (apply_config() works)
      - Stream lifecycle: configure RF params FIRST, then setup stream
    """

    def __init__(self, device_args: Optional[Dict[str, Any]] = None):
        if _libbladerf is None:
            raise RuntimeError("libbladerf not available - cannot use native backend")

        super().__init__()

        self.device_args: Dict[str, Any] = dict(device_args) if device_args else {}
        self.dev: Optional[ctypes.c_void_p] = None
        self.rx_channel: int = 0
        self.max_rx_channels: int = 1  # Edge only uses single-channel
        self.supports_agc: bool = True  # bladeRF supports AGC

        # RF state
        self.center_freq_hz: int = 0
        self.sample_rate_sps: int = 0
        self.bandwidth_hz: Optional[int] = None
        self.gain_db: float = 30.0
        self.gain_mode: GainMode = GainMode.MANUAL

        # Stream state
        self._stream_active: bool = False
        self._stream_configured: bool = False

        # Health tracking (matching SoapySDRDevice pattern)
        self._health_stats = {
            "total_reads": 0,
            "successful_reads": 0,
            "timeout_reads": 0,
            "error_reads": 0,
            "overflow_errors": 0,
            "total_samples": 0,
            "total_read_time_ns": 0,
            "start_time": time.time(),
        }

        # Pre-allocated conversion buffers (reuse for performance)
        self._conv_buf_i: Optional[np.ndarray] = None
        self._conv_buf_q: Optional[np.ndarray] = None
        self._conv_buf_iq: Optional[np.ndarray] = None

        # Overflow counter to prevent log storms
        self._overflow_count = 0

        # Open device
        self._open_device()

    def _open_device(self) -> None:
        """Open bladeRF device using libbladerf."""
        if self.dev is not None:
            return

        dev_ptr = ctypes.c_void_p()
        # Use "*" to open first available device (matching Tripwire pattern)
        ret = _libbladerf.bladerf_open(ctypes.byref(dev_ptr), b"*")

        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            if ret == BLADERF_ERR_INVAL:
                # Device might already be open - try to close and reopen
                logger.warning("Device might be in use, attempting recovery...")
                # For now, raise error - can add recovery logic later
            raise RuntimeError(f"Failed to open bladeRF device: {error_str} (code: {ret})")

        self.dev = dev_ptr
        logger.info("BladeRFNativeDevice: Device opened successfully")

    def open(self) -> None:
        """Open device (already opened in __init__)."""
        if self.dev is None:
            self._open_device()

    def close(self) -> None:
        """Close device and cleanup."""
        if self._stream_active:
            self._deactivate_stream()

        if self.dev is not None:
            _libbladerf.bladerf_close(self.dev)
            self.dev = None
            logger.info("BladeRFNativeDevice: Device closed")

    def set_rx_channel(self, channel: int):
        """Set RX channel (Edge only uses channel 0)."""
        ch = int(channel)
        if ch < 0 or ch >= self.max_rx_channels:
            ch = 0
        self.rx_channel = ch
        # Note: Channel change would require stream rebuild, but Edge only uses ch 0

    def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None):
        """
        Configure RF parameters in CRITICAL ORDER (from project rules):
        1. Disable AGC (if manual gain mode)
        2. Set sample rate FIRST
        3. Set bandwidth
        4. Set frequency
        5. Set gain
        6. Configure and activate stream (happens in _setup_stream)
        """
        if self.dev is None:
            return

        self.center_freq_hz = int(center_freq_hz)
        self.sample_rate_sps = int(sample_rate_sps)
        self.bandwidth_hz = int(bandwidth_hz) if bandwidth_hz else int(sample_rate_sps)

        ch = BLADERF_CHANNEL_RX(self.rx_channel)

        # Step 1: Disable AGC first (if manual gain mode)
        if self.gain_mode == GainMode.MANUAL:
            ret = _libbladerf.bladerf_set_gain_mode(self.dev, ch, BLADERF_GAIN_MGC)
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                logger.warning(f"Failed to set gain mode: {error_str}")

        # Step 2: Set sample rate
        actual_rate = ctypes.c_uint32()
        ret = _libbladerf.bladerf_set_sample_rate(
            self.dev, ch, self.sample_rate_sps, ctypes.byref(actual_rate)
        )
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            raise RuntimeError(f"Failed to set sample rate: {error_str}")

        # Step 3: Set bandwidth
        actual_bw = ctypes.c_uint32()
        ret = _libbladerf.bladerf_set_bandwidth(
            self.dev, ch, self.bandwidth_hz, ctypes.byref(actual_bw)
        )
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            raise RuntimeError(f"Failed to set bandwidth: {error_str}")

        # Step 4: Set frequency
        ret = _libbladerf.bladerf_set_frequency(self.dev, ch, self.center_freq_hz)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            raise RuntimeError(f"Failed to set frequency: {error_str}")

        # Step 5: Set gain (if manual mode)
        if self.gain_mode == GainMode.MANUAL:
            ret = _libbladerf.bladerf_set_gain(self.dev, ch, int(self.gain_db))
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                logger.warning(f"Failed to set gain: {error_str}")

        # Step 6: Stream setup MUST happen last (after all RF settings)
        self._setup_stream()

    def set_gain(self, gain_db: float):
        """Set manual gain."""
        if self.dev is None:
            return

        self.gain_db = float(gain_db)
        ch = BLADERF_CHANNEL_RX(self.rx_channel)

        try:
            ret = _libbladerf.bladerf_set_gain(self.dev, ch, int(self.gain_db))
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                logger.warning(f"Failed to set gain: {error_str}")
        except Exception as e:
            logger.warning(f"set_gain exception: {e}")

    def set_gain_mode(self, mode: GainMode):
        """Set gain mode (manual or AGC)."""
        self.gain_mode = mode
        if self.dev is None:
            return

        ch = BLADERF_CHANNEL_RX(self.rx_channel)
        gain_mode_val = BLADERF_GAIN_AGC if mode == GainMode.AGC else BLADERF_GAIN_MGC

        try:
            ret = _libbladerf.bladerf_set_gain_mode(self.dev, ch, gain_mode_val)
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                logger.warning(f"Failed to set gain mode: {error_str}")
        except Exception as e:
            logger.warning(f"set_gain_mode exception: {e}")

    def _setup_stream(self) -> None:
        """
        Setup and activate RX stream.
        MUST be called AFTER tune() sets all RF parameters.
        """
        if self.dev is None:
            return

        # Tear down existing stream first
        if self._stream_active:
            self._deactivate_stream()

        ch = BLADERF_CHANNEL_RX(self.rx_channel)

        # Configure sync streaming (single-channel)
        # Buffer configuration (matching Edge's power-of-two requirement)
        num_buffers = 64  # From SoapySDRDevice pattern
        buffer_size = 131072  # Power-of-two, matches Edge's high-rate chunk size
        num_transfers = 16
        stream_timeout_ms = 5000  # 5 second timeout

        ret = _libbladerf.bladerf_sync_config(
            self.dev,
            BLADERF_RX_X1,              # Single-channel layout
            BLADERF_FORMAT_SC16_Q11,   # CS16 format
            num_buffers,
            buffer_size,
            num_transfers,
            stream_timeout_ms
        )

        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            raise RuntimeError(f"Failed to configure sync RX: {error_str}")

        # Enable RX module
        ret = _libbladerf.bladerf_enable_module(self.dev, ch, True)
        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            raise RuntimeError(f"Failed to enable RX module: {error_str}")

        # Brief wait for stream to stabilize (from Tripwire pattern)
        time.sleep(0.2)

        self._stream_active = True
        self._stream_configured = True
        logger.info("BladeRFNativeDevice: Stream activated successfully")

    def _deactivate_stream(self) -> None:
        """Deactivate and cleanup stream."""
        if not self._stream_active:
            return

        if self.dev is not None:
            ch = BLADERF_CHANNEL_RX(self.rx_channel)
            ret = _libbladerf.bladerf_enable_module(self.dev, ch, False)
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                logger.warning(f"Error disabling RX module: {error_str}")

        self._stream_active = False
        self._stream_configured = False
        logger.info("BladeRFNativeDevice: Stream deactivated")

    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        Read samples from bladeRF.
        Returns empty array on timeout/error (never throws).
        
        Uses CS16 format (int16 I/Q pairs) and converts to complex64.
        """
        if self.dev is None or not self._stream_active:
            return np.empty(0, dtype=np.complex64)

        n = int(num_samples)
        if n <= 0:
            return np.empty(0, dtype=np.complex64)

        # Ensure power-of-two (Edge requirement)
        # Round up to next power-of-two if needed
        if n & (n - 1) != 0:
            # Not a power-of-two, round up
            n = 1 << (n - 1).bit_length()
            logger.debug(f"Rounded read size to power-of-two: {n}")

        # Pre-allocate conversion buffers if needed (reuse for performance)
        if (self._conv_buf_iq is None or len(self._conv_buf_iq) != n):
            self._conv_buf_i = np.empty(n, dtype=np.float32)
            self._conv_buf_q = np.empty(n, dtype=np.float32)
            self._conv_buf_iq = np.empty(n, dtype=np.complex64)

        # Allocate CS16 buffer (2 int16 per complex sample)
        buf_size = n * 2
        buf = (ctypes.c_int16 * buf_size)()

        # Track read timing
        read_start_ns = time.perf_counter_ns()
        self._health_stats["total_reads"] += 1

        # Read from bladeRF (blocking call)
        timeout_ms = 250  # 250ms timeout (increased from 100ms for production workloads)
        ret = _libbladerf.bladerf_sync_rx(
            self.dev,
            buf,
            n,  # Number of samples (per-channel)
            None,  # No metadata
            timeout_ms
        )

        read_time_ns = time.perf_counter_ns() - read_start_ns
        self._health_stats["total_read_time_ns"] += read_time_ns

        if ret != 0:
            # Error or timeout
            if ret == BLADERF_ERR_TIMEOUT:
                # Timeout: track separately but don't count as error
                # Timeouts are environmental (system load, USB hub contention) not device errors
                self._health_stats["timeout_reads"] += 1
            else:
                # Actual error (not timeout): count as error
                self._health_stats["error_reads"] += 1
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                logger.debug(f"bladerf_sync_rx error: {error_str} (code: {ret})")
            return np.empty(0, dtype=np.complex64)

        # Successful read
        self._health_stats["successful_reads"] += 1
        self._health_stats["total_samples"] += n

        # Convert CS16 buffer to numpy array
        arr = np.frombuffer(buf, dtype=np.int16)

        # Extract I/Q using zero-copy slicing (from Tripwire pattern)
        # Layout: [I, Q, I, Q, ...]
        # I: indices 0, 2, 4, ... (even)
        # Q: indices 1, 3, 5, ... (odd)
        scale = 1.0 / 2048.0  # SC16_Q11 scaling factor

        # Use pre-allocated buffers for conversion
        np.multiply(
            arr[0::2].astype(np.float32, copy=False), 
            scale, 
            out=self._conv_buf_i
        )
        np.multiply(
            arr[1::2].astype(np.float32, copy=False), 
            scale, 
            out=self._conv_buf_q
        )

        # Combine I/Q into complex array
        self._conv_buf_iq.real = self._conv_buf_i
        self._conv_buf_iq.imag = self._conv_buf_q

        return self._conv_buf_iq.copy()  # Return copy to avoid buffer reuse issues

    def get_health(self) -> dict:
        """Return SDR health metrics for monitoring."""
        stats = self._health_stats.copy()
        elapsed_s = time.time() - stats["start_time"]

        if elapsed_s < 0.1:
            return {
                "status": "unknown",
                "success_rate_pct": 0.0,
                "throughput_mbps": 0.0,
                "samples_per_sec": 0.0,
                "avg_read_time_ms": 0.0,
                "errors": 0,
                "timeouts": 0,
                "reads": {"total": 0, "successful": 0},
                "stream": "inactive" if not self._stream_active else "active",
                "usb_speed": "USB 3.0",  # bladeRF 2.0 micro is USB 3.0
            }

        total_reads = stats["total_reads"]
        successful_reads = stats["successful_reads"]
        success_rate = (successful_reads / total_reads * 100.0) if total_reads > 0 else 0.0

        # Throughput: samples/sec * 8 bytes (complex64) = bytes/sec
        samples_per_sec = stats["total_samples"] / elapsed_s
        throughput_mbps = (samples_per_sec * 8) / (1024 * 1024)  # MB/s

        # Average read time
        avg_read_time_ms = (stats["total_read_time_ns"] / total_reads / 1_000_000) if total_reads > 0 else 0.0

        # Determine status
        # Note: success_rate excludes timeouts (timeouts don't count as errors)
        # Timeouts are environmental issues, not device failures
        if success_rate >= 95.0 and stats["overflow_errors"] == 0:
            status = "good"
        elif success_rate >= 80.0:
            status = "fair"
        else:
            status = "poor"

        stream_status = "active" if self._stream_active else "inactive"

        return {
            "status": status,
            "success_rate_pct": round(success_rate, 1),
            "throughput_mbps": round(throughput_mbps, 2),
            "samples_per_sec": round(samples_per_sec / 1_000_000, 2),  # MS/s
            "avg_read_time_ms": round(avg_read_time_ms, 2),
            "errors": stats["error_reads"],
            "timeouts": stats["timeout_reads"],
            "reads": {
                "total": total_reads,
                "successful": successful_reads,
            },
            "stream": stream_status,
            "usb_speed": "USB 3.0",
        }

    def get_info(self) -> dict:
        """Return device info for UI."""
        return {
            "driver": "bladerf_native",
            "label": "bladeRF 2.0 (Native)",
            "serial": "unknown",  # Can query from device if needed
            "rx_channels": self.max_rx_channels,
            "supports_agc": self.supports_agc,
            "active_rx_channel": self.rx_channel,
        }
