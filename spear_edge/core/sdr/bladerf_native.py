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
_LNA_FUNCTIONS_AVAILABLE = False  # Will be set when binding functions
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
    
    _libbladerf.bladerf_get_gain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    _libbladerf.bladerf_get_gain.restype = ctypes.c_int
    
    # LNA control functions (confirmed available in libbladerf)
    _libbladerf.bladerf_set_lna_gain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _libbladerf.bladerf_set_lna_gain.restype = ctypes.c_int
    
    _libbladerf.bladerf_get_lna_gain.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _libbladerf.bladerf_get_lna_gain.restype = ctypes.c_int
    
    # Gain stage functions (for more granular control if needed)
    _libbladerf.bladerf_set_gain_stage.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    _libbladerf.bladerf_set_gain_stage.restype = ctypes.c_int
    
    _libbladerf.bladerf_get_gain_stage.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
    _libbladerf.bladerf_get_gain_stage.restype = ctypes.c_int
    
    # Bias-tee control (for BT200 external LNA)
    _libbladerf.bladerf_set_bias_tee.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
    _libbladerf.bladerf_set_bias_tee.restype = ctypes.c_int
    
    _libbladerf.bladerf_get_bias_tee.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_bool)]
    _libbladerf.bladerf_get_bias_tee.restype = ctypes.c_int
    
    _LNA_FUNCTIONS_AVAILABLE = True
    logger.debug("LNA and bias-tee control functions available")
    
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
    
    Native libbladerf driver for bladeRF 2.0 micro/xA4.
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
        self.max_rx_channels: int = 2  # bladeRF 2.0 supports 2 RX channels
        self.supports_agc: bool = True  # bladeRF supports AGC
        self.dual_channel_mode: bool = False  # Track if dual channel is active

        # RF state
        self.center_freq_hz: int = 0
        self.sample_rate_sps: int = 0
        self.bandwidth_hz: Optional[int] = None
        self.gain_db: float = 0.0  # Default 0 dB - user can adjust via UI slider
        self.gain_mode: GainMode = GainMode.MANUAL
        
        # LNA state
        self.lna_gain_db: Dict[int, int] = {0: 0, 1: 0}  # Internal LNA gain per channel (0-30 dB)
        self._lna_supported: Dict[int, bool] = {0: True, 1: True}  # Track if LNA is supported per channel (tested on first attempt)
        self._lna_warning_logged: Dict[int, bool] = {0: False, 1: False}  # Only log warning once per channel
        
        # BT200 external LNA state (bias-tee controlled)
        # CRITICAL: BT200 is NOT connected to the SDR - always defaults to False
        # Only enabled if user explicitly sets it via apply_config()
        self.bt200_enabled: Dict[int, bool] = {0: False, 1: False}  # BT200 enabled per channel (default: OFF)

        # Stream state
        self._stream_active: bool = False
        self._stream_configured: bool = False

        # Health tracking for monitoring SDR performance
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
        
        # CRITICAL: Ensure BT200 is OFF on device open (hardware not connected)
        # BT200 should only be enabled if user explicitly sets it via apply_config()
        for ch in range(self.max_rx_channels):
            # Explicitly disable BT200 to ensure hardware state matches (hardware not connected)
            self.set_bt200_enabled(ch, False)

    def open(self) -> None:
        """Open device (already opened in __init__)."""
        if self.dev is None:
            self._open_device()
            # CRITICAL: Ensure BT200 is OFF (hardware not connected)
            # BT200 should only be enabled if user explicitly sets it
            for ch in range(self.max_rx_channels):
                self.set_bt200_enabled(ch, False)

    def close(self) -> None:
        """Close device and cleanup."""
        if self._stream_active:
            self._deactivate_stream()

        if self.dev is not None:
            _libbladerf.bladerf_close(self.dev)
            self.dev = None
            logger.info("BladeRFNativeDevice: Device closed")

    def set_rx_channel(self, channel: int):
        """Set RX channel (0 or 1)."""
        ch = int(channel)
        if ch < 0 or ch >= self.max_rx_channels:
            ch = 0
        self.rx_channel = ch
        # Note: Channel change would require stream rebuild if stream is active

    def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None, dual_channel: bool = False):
        """
        Configure RF parameters in CRITICAL ORDER (from project rules):
        1. Disable AGC (if manual gain mode)
        2. Set sample rate FIRST
        3. Set bandwidth
        4. Set frequency
        5. Set gain
        6. Set LNA gain (if configured)
        7. Configure and activate stream (happens in _setup_stream)
        
        Args:
            center_freq_hz: Center frequency in Hz
            sample_rate_sps: Sample rate in samples per second
            bandwidth_hz: Bandwidth in Hz (defaults to sample_rate_sps)
            dual_channel: If True, configure both channels for dual RX mode
        """
        if self.dev is None:
            return

        self.center_freq_hz = int(center_freq_hz)
        self.sample_rate_sps = int(sample_rate_sps)
        self.bandwidth_hz = int(bandwidth_hz) if bandwidth_hz else int(sample_rate_sps)
        self.dual_channel_mode = dual_channel

        # Configure channels (single or dual)
        channels_to_config = [0, 1] if dual_channel else [self.rx_channel]
        
        for ch_num in channels_to_config:
            ch = BLADERF_CHANNEL_RX(ch_num)

            # Step 1: Disable AGC first (if manual gain mode)
            if self.gain_mode == GainMode.MANUAL:
                ret = _libbladerf.bladerf_set_gain_mode(self.dev, ch, BLADERF_GAIN_MGC)
                if ret != 0:
                    error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                    logger.warning(f"Failed to set gain mode for ch{ch_num}: {error_str}")

            # Step 2: Set sample rate
            actual_rate = ctypes.c_uint32()
            ret = _libbladerf.bladerf_set_sample_rate(
                self.dev, ch, self.sample_rate_sps, ctypes.byref(actual_rate)
            )
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                raise RuntimeError(f"Failed to set sample rate for ch{ch_num}: {error_str}")

            # Step 3: Set bandwidth
            actual_bw = ctypes.c_uint32()
            ret = _libbladerf.bladerf_set_bandwidth(
                self.dev, ch, self.bandwidth_hz, ctypes.byref(actual_bw)
            )
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                raise RuntimeError(f"Failed to set bandwidth for ch{ch_num}: {error_str}")

            # Step 4: Set frequency
            ret = _libbladerf.bladerf_set_frequency(self.dev, ch, self.center_freq_hz)
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                raise RuntimeError(f"Failed to set frequency for ch{ch_num}: {error_str}")

            # Step 5: Set BT200 bias-tee (if explicitly enabled for this channel)
            # CRITICAL: BT200 is only enabled if user explicitly sets it to True
            # Note: LNA gain is now automatically optimized by bladerf_set_gain() - no manual control
            # Only set BT200 if it's explicitly True (not just in the dict)
            if ch_num in self.bt200_enabled and self.bt200_enabled[ch_num] is True:
                self.set_bt200_enabled(ch_num, True)
            else:
                # Explicitly disable to ensure hardware state matches (BT200 not connected by default)
                self.set_bt200_enabled(ch_num, False)

        # Step 6: Stream setup MUST happen before gain can be set
        # CRITICAL: Gain must be set AFTER stream is configured, not before!
        # Setting gain before stream setup causes it to be ignored/clamped to 60 dB
        self._setup_stream(dual_channel)
        
        # Step 7: Set gain AFTER stream is configured (if manual mode)
        # CRITICAL: This must happen after _setup_stream() or gain will be clamped to 60 dB
        if self.gain_mode == GainMode.MANUAL:
            import time
            time.sleep(0.05)  # Brief delay after stream setup
            
            for ch_num in channels_to_config:
                ch = BLADERF_CHANNEL_RX(ch_num)
                
                # Ensure gain mode is MANUAL
                ret_mode = _libbladerf.bladerf_set_gain_mode(self.dev, ch, BLADERF_GAIN_MGC)
                if ret_mode != 0:
                    error_str = _libbladerf.bladerf_strerror(ret_mode).decode('utf-8', errors='ignore')
                    logger.warning(f"[TUNE] Failed to set gain mode to MANUAL for ch{ch_num}: {error_str}")
                else:
                    gain_int = int(self.gain_db)
                    print(f"[TUNE] Setting gain for ch{ch_num} to {gain_int} dB (requested: {self.gain_db} dB)")
                    ret = _libbladerf.bladerf_set_gain(self.dev, ch, gain_int)
                    if ret != 0:
                        error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                        logger.warning(f"[TUNE] Failed to set gain for ch{ch_num} to {gain_int} dB: {error_str}")
                        print(f"[TUNE] ERROR: Failed to set gain: {error_str}")
                    else:
                        # Small delay before reading back
                        time.sleep(0.05)
                        # Verify gain was actually applied by reading back from hardware
                        try:
                            gain_ptr = ctypes.c_int()
                            ret_get = _libbladerf.bladerf_get_gain(self.dev, ch, ctypes.byref(gain_ptr))
                            if ret_get == 0:
                                applied = gain_ptr.value
                                print(f"[TUNE] Gain verification: requested {gain_int} dB, hardware reports {applied} dB")
                                if applied != gain_int:
                                    logger.warning(f"[TUNE] Gain mismatch: requested {gain_int} dB, but hardware applied {applied} dB")
                                    print(f"[TUNE] WARNING: Gain mismatch! Requested {gain_int} dB but got {applied} dB")
                                    # Update internal state to match hardware
                                    self.gain_db = float(applied)
                                else:
                                    logger.info(f"[TUNE] Set gain for ch{ch_num} to {gain_int} dB (verified after stream setup)")
                                    print(f"[TUNE] OK: Gain set to {gain_int} dB (verified)")
                            else:
                                error_str = _libbladerf.bladerf_strerror(ret_get).decode('utf-8', errors='ignore')
                                logger.warning(f"[TUNE] Could not verify gain from hardware: {error_str}")
                                print(f"[TUNE] WARNING: Could not verify gain: {error_str}")
                        except Exception as e:
                            logger.warning(f"[TUNE] Exception verifying gain: {e}")
                            logger.info(f"[TUNE] Set gain for ch{ch_num} to {gain_int} dB (verification failed)")
                            print(f"[TUNE] WARNING: Exception verifying gain: {e}")

    def set_gain(self, gain_db: float, channel: Optional[int] = None):
        """Set manual gain.
        
        CRITICAL: Gain can only be set when gain_mode is MANUAL.
        If AGC is enabled, this will disable AGC and set manual gain.
        
        Args:
            gain_db: Gain in dB
            channel: Channel number (0 or 1). If None, uses current rx_channel.
        """
        if self.dev is None:
            logger.warning("set_gain: Device not open")
            return
        
        self.gain_db = float(gain_db)
        ch_num = channel if channel is not None else self.rx_channel
        ch = BLADERF_CHANNEL_RX(ch_num)
        
        # CRITICAL FIX: Ensure gain_mode is MANUAL before setting gain
        # If AGC is enabled, gain changes will be ignored by hardware
        # Always set gain mode to MANUAL to ensure it's in the correct state
        ret_mode = _libbladerf.bladerf_set_gain_mode(self.dev, ch, BLADERF_GAIN_MGC)
        if ret_mode != 0:
            error_str = _libbladerf.bladerf_strerror(ret_mode).decode('utf-8', errors='ignore')
            logger.error(f"Failed to set gain mode to MANUAL for ch{ch_num}: {error_str}")
            return
        
        # Update internal state
        self.gain_mode = GainMode.MANUAL
        
        # Small delay to ensure gain mode is applied before setting gain value
        import time
        time.sleep(0.01)  # 10ms delay
        
        try:
            gain_int = int(self.gain_db)
            logger.info(f"[GAIN] Setting gain for ch{ch_num}: {gain_int} dB")
            ret = _libbladerf.bladerf_set_gain(self.dev, ch, gain_int)
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                logger.error(f"[GAIN] Failed to set gain for ch{ch_num} to {gain_int} dB: {error_str} (code: {ret})")
            else:
                # Small delay before reading back to ensure gain is applied
                time.sleep(0.01)
                
                # Read back from hardware to verify what actually applied
                try:
                    gain_ptr = ctypes.c_int()
                    ret_get = _libbladerf.bladerf_get_gain(self.dev, ch, ctypes.byref(gain_ptr))
                    if ret_get == 0:
                        applied = gain_ptr.value
                        if applied != gain_int:
                            logger.warning(f"[GAIN] Gain mismatch: requested {gain_int} dB, but hardware applied {applied} dB")
                            # Update internal state to match hardware
                            self.gain_db = float(applied)
                        else:
                            logger.info(f"[GAIN] Gain set to {gain_int} dB for ch{ch_num} (verified)")
                    else:
                        error_str = _libbladerf.bladerf_strerror(ret_get).decode('utf-8', errors='ignore')
                        logger.warning(f"[GAIN] Could not read back gain from hardware: {error_str}")
                except Exception as e:
                    logger.warning(f"[GAIN] Exception reading back gain: {e}")
        except Exception as e:
            logger.error(f"[GAIN] set_gain exception for ch{ch_num}: {e}", exc_info=True)
    
    def set_lna_gain(self, channel: int, gain_db: int) -> bool:
        """Set LNA gain for specified channel.
        
        Args:
            channel: RX channel (0 or 1)
            gain_db: LNA gain in dB (typically 0, 6, 12, 18, 24, 30)
        
        Returns:
            True if successful, False otherwise
        """
        if self.dev is None:
            return False
        
        if channel < 0 or channel >= self.max_rx_channels:
            logger.warning(f"Invalid channel for LNA: {channel}")
            return False
        
        # Clamp to valid range (0-30 dB, typically in 6 dB steps)
        gain_db = max(0, min(30, int(gain_db)))
        
        return self._set_lna_gain_internal(channel, gain_db)
    
    def _set_lna_gain_internal(self, channel: int, gain_db: int) -> bool:
        """Internal LNA gain setter using bladerf_set_lna_gain.
        
        Validates and snaps to nearest valid step, then verifies what was actually applied.
        """
        # If LNA is not supported on this channel, skip silently
        if not self._lna_supported.get(channel, True):
            return False
        
        ch = BLADERF_CHANNEL_RX(channel)
        
        # Valid LNA gain steps for bladeRF 2.0 (typically 0, 6, 12, 18, 24, 30 dB)
        VALID_LNA_STEPS = [0, 6, 12, 18, 24, 30]
        
        # Snap to nearest valid step
        gain_db_snapped = min(VALID_LNA_STEPS, key=lambda g: abs(g - gain_db))
        if gain_db_snapped != gain_db:
            logger.debug(f"LNA gain {gain_db} dB snapped to nearest valid step: {gain_db_snapped} dB")
        
        # Always store the snapped value for tracking
        self.lna_gain_db[channel] = gain_db_snapped
        
        try:
            # Use bladerf_set_lna_gain (confirmed available in libbladerf)
            ret = _libbladerf.bladerf_set_lna_gain(self.dev, ch, gain_db_snapped)
            if ret == 0:
                # Read back from hardware to verify what actually applied
                try:
                    applied = _libbladerf.bladerf_get_lna_gain(self.dev, ch)
                    if applied >= 0:
                        logger.info(f"[TUNE] LNA requested={gain_db_snapped} dB applied={applied} dB for ch{channel}")
                        # Update stored value to match what hardware actually has
                        if applied != gain_db_snapped:
                            logger.warning(f"[TUNE] LNA gain mismatch: requested {gain_db_snapped} dB but hardware reports {applied} dB")
                            self.lna_gain_db[channel] = applied
                    else:
                        logger.info(f"Set LNA gain for ch{channel}: {gain_db_snapped} dB (read-back unavailable)")
                except Exception as e:
                    logger.debug(f"Could not read back LNA gain: {e}")
                    logger.info(f"Set LNA gain for ch{channel}: {gain_db_snapped} dB")
                return True
            else:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                
                # Check if error is "Operation not supported" - mark as unsupported
                if "not supported" in error_str.lower() or "unsupported" in error_str.lower():
                    self._lna_supported[channel] = False
                    if not self._lna_warning_logged.get(channel, False):
                        logger.info(f"LNA gain control not supported on ch{channel} - disabling LNA control for this channel")
                        self._lna_warning_logged[channel] = True
                    return False
                
                # Other errors - log warning only once
                if not self._lna_warning_logged.get(channel, False):
                    logger.warning(f"Failed to set LNA gain for ch{channel}: {error_str}")
                    self._lna_warning_logged[channel] = True
                return False
        except Exception as e:
            # Mark as unsupported on exception
            self._lna_supported[channel] = False
            if not self._lna_warning_logged.get(channel, False):
                logger.info(f"LNA gain control not available on ch{channel}: {e}")
                self._lna_warning_logged[channel] = True
            return False
    
    def get_lna_gain(self, channel: int) -> Optional[int]:
        """Get current LNA gain for specified channel.
        
        Returns stored value. If LNA functions are available, attempts to query hardware.
        """
        if self.dev is None:
            return self.lna_gain_db.get(channel, 0)
        
        if channel < 0 or channel >= self.max_rx_channels:
            return None
        
        ch = BLADERF_CHANNEL_RX(channel)
        
        try:
            # Use bladerf_get_lna_gain (confirmed available)
            ret = _libbladerf.bladerf_get_lna_gain(self.dev, ch)
            if ret >= 0:
                self.lna_gain_db[channel] = ret
                return ret
        except Exception as e:
            logger.warning(f"get_lna_gain exception: {e}")
        
        # Return stored value if hardware query fails
        return self.lna_gain_db.get(channel, 0)
    
    def set_bt200_enabled(self, channel: int, enabled: bool) -> bool:
        """Enable/disable BT200 external LNA via bias-tee.
        
        The BT200 is a bias-tee powered external LNA that connects inline
        between the antenna and bladeRF RX SMA port. When bias-tee is enabled,
        the BT200 is powered and active (~16-20 dB gain). When disabled,
        it operates in passive bypass mode (~4 dB insertion loss).
        
        Args:
            channel: RX channel (0 or 1)
            enabled: True to enable BT200 (bias-tee on), False to disable (bias-tee off)
        
        Returns:
            True if successful, False otherwise
        """
        if self.dev is None:
            return False
        
        if channel < 0 or channel >= self.max_rx_channels:
            logger.warning(f"Invalid channel for BT200: {channel}")
            return False
        
        ch = BLADERF_CHANNEL_RX(channel)
        
        try:
            ret = _libbladerf.bladerf_set_bias_tee(self.dev, ch, enabled)
            if ret == 0:
                self.bt200_enabled[channel] = enabled
                status = "enabled" if enabled else "disabled"
                logger.info(f"BT200 bias-tee for ch{channel}: {status}")
                return True
            else:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                logger.warning(f"Failed to set BT200 bias-tee for ch{channel}: {error_str}")
                return False
        except Exception as e:
            logger.warning(f"set_bt200_enabled exception for ch{channel}: {e}")
            return False
    
    def get_bt200_enabled(self, channel: int) -> Optional[bool]:
        """Get current BT200 bias-tee state for specified channel."""
        if self.dev is None:
            return self.bt200_enabled.get(channel, False)
        
        if channel < 0 or channel >= self.max_rx_channels:
            return None
        
        ch = BLADERF_CHANNEL_RX(channel)
        
        try:
            enabled = ctypes.c_bool()
            ret = _libbladerf.bladerf_get_bias_tee(self.dev, ch, ctypes.byref(enabled))
            if ret == 0:
                self.bt200_enabled[channel] = enabled.value
                return enabled.value
        except Exception as e:
            logger.warning(f"get_bt200_enabled exception: {e}")
        
        # Return stored value if hardware query fails
        return self.bt200_enabled.get(channel, False)

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

    def _setup_stream(self, dual_channel: bool = False) -> None:
        """
        Setup and activate RX stream.
        MUST be called AFTER tune() sets all RF parameters.
        
        Args:
            dual_channel: If True, configure for dual RX (both channels)
        """
        if self.dev is None:
            return

        # Tear down existing stream first
        if self._stream_active:
            self._deactivate_stream()

        # Choose channel layout
        channel_layout = BLADERF_RX_X2 if dual_channel else BLADERF_RX_X1

        # Configure sync streaming
        # Buffer configuration (matching Edge's power-of-two requirement)
        num_buffers = 64  # Standard bladeRF buffer count
        buffer_size = 131072  # Power-of-two, matches Edge's high-rate chunk size
        num_transfers = 16
        stream_timeout_ms = 5000  # 5 second timeout

        ret = _libbladerf.bladerf_sync_config(
            self.dev,
            channel_layout,            # Single or dual channel
            BLADERF_FORMAT_SC16_Q11,   # CS16 format
            num_buffers,
            buffer_size,
            num_transfers,
            stream_timeout_ms
        )

        if ret != 0:
            error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
            raise RuntimeError(f"Failed to configure sync RX: {error_str}")

        # Enable RX module(s)
        if dual_channel:
            # Enable both channels
            ch0 = BLADERF_CHANNEL_RX(0)
            ch1 = BLADERF_CHANNEL_RX(1)
            ret0 = _libbladerf.bladerf_enable_module(self.dev, ch0, True)
            ret1 = _libbladerf.bladerf_enable_module(self.dev, ch1, True)
            if ret0 != 0 or ret1 != 0:
                error_str = _libbladerf.bladerf_strerror(ret0 if ret0 != 0 else ret1).decode('utf-8', errors='ignore')
                raise RuntimeError(f"Failed to enable RX modules: {error_str}")
        else:
            # Enable single channel
            ch = BLADERF_CHANNEL_RX(self.rx_channel)
            ret = _libbladerf.bladerf_enable_module(self.dev, ch, True)
            if ret != 0:
                error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                raise RuntimeError(f"Failed to enable RX module: {error_str}")

        # Brief wait for stream to stabilize (from Tripwire pattern)
        time.sleep(0.2)

        self._stream_active = True
        self._stream_configured = True
        self.dual_channel_mode = dual_channel
        logger.info(f"BladeRFNativeDevice: Stream activated ({'dual' if dual_channel else 'single'} channel)")

    def _deactivate_stream(self) -> None:
        """Deactivate and cleanup stream."""
        if not self._stream_active:
            return

        if self.dev is not None:
            if self.dual_channel_mode:
                # Disable both channels
                ch0 = BLADERF_CHANNEL_RX(0)
                ch1 = BLADERF_CHANNEL_RX(1)
                _libbladerf.bladerf_enable_module(self.dev, ch0, False)
                _libbladerf.bladerf_enable_module(self.dev, ch1, False)
            else:
                # Disable single channel
                ch = BLADERF_CHANNEL_RX(self.rx_channel)
                ret = _libbladerf.bladerf_enable_module(self.dev, ch, False)
                if ret != 0:
                    error_str = _libbladerf.bladerf_strerror(ret).decode('utf-8', errors='ignore')
                    logger.warning(f"Error disabling RX module: {error_str}")

        self._stream_active = False
        self._stream_configured = False
        self.dual_channel_mode = False
        logger.info("BladeRFNativeDevice: Stream deactivated")

    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        Read samples from bladeRF.
        Returns empty array on timeout/error (never throws).
        
        For dual channel mode, returns channel 0 samples only.
        (Dual channel interleaved data: [ch0_i, ch0_q, ch1_i, ch1_q, ...])
        
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

        # Allocate CS16 buffer
        # For dual channel: 2 channels * 2 (I/Q) * n samples
        # For single channel: 2 (I/Q) * n samples
        buf_size = n * 2 * (2 if self.dual_channel_mode else 1)
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
        self._health_stats["total_samples"] += n * (2 if self.dual_channel_mode else 1)

        # Convert CS16 buffer to numpy array
        arr = np.frombuffer(buf, dtype=np.int16)
        
        # CRITICAL: Check for ADC clipping/overload BEFORE scaling
        # SC16_Q11 format: 11 fractional bits, full-scale is ±2048 (not ±32768)
        # According to libbladeRF.h: "Values in the range [-2048, 2048) represent [-1.0, 1.0)"
        # Valid range is [-2048, 2047] inclusive, so max should be 2047, not 2048
        raw = np.abs(arr)
        raw_max = int(raw.max())
        raw_min = int(raw.min())
        raw_mean = float(np.mean(raw))
        raw_std = float(np.std(raw))
        
        # Check actual min/max of raw array (before abs) to see if we're getting values outside Q11 range
        arr_min = int(arr.min())
        arr_max = int(arr.max())
        
        # Normalize IQ correctly for SC16_Q11
        # CRITICAL: There's a contradiction in bladeRF documentation:
        # - libbladeRF.h says: Q11 uses [-2048, 2047] range, divide by 2048
        # - API_QUICK_REFERENCE.md says: divide by 32768.0 (full int16 range)
        # 
        # We're seeing values up to 2048 (outside Q11 range), which suggests:
        # 1. Hardware may be outputting values slightly outside Q11 range when saturated
        # 2. OR the format isn't strictly Q11 in practice
        #
        # Allow configurable scaling mode via env var for testing:
        # - SPEAR_IQ_SCALING_MODE=q11 (default, per libbladeRF.h)
        # - SPEAR_IQ_SCALING_MODE=int16 (per API_QUICK_REFERENCE.md)
        from spear_edge.settings import settings
        
        # Check format violations based on scaling mode
        # Q11 valid range: [-2048, 2047], so abs should be <= 2047
        # int16 valid range: [-32768, 32767], so abs should be <= 32767
        if settings.IQ_SCALING_MODE == "int16":
            format_max = 32767
            format_name = "int16"
        else:
            format_max = 2047
            format_name = "Q11"
        
        format_violations = np.sum(np.abs(arr) > format_max)
        format_violation_frac = float(format_violations) / len(arr) if len(arr) > 0 else 0.0
        q11_violation_frac = format_violation_frac  # Keep for backward compatibility in logs
        if settings.IQ_SCALING_MODE == "int16":
            scale = 1.0 / 32768.0  # Full int16 range normalization
            self._scaling_mode = "int16 (1/32768)"
            rail_threshold = 32767  # int16 full-scale threshold
        else:
            scale = 1.0 / 2048.0  # Q11 normalization: 1.0 = Q11 full-scale (±2048)
            self._scaling_mode = "Q11 (1/2048)"
            rail_threshold = 2047  # Q11 full-scale threshold
        
        # Rail detection: use appropriate threshold based on scaling mode
        rail_frac = float(np.mean(raw >= rail_threshold))
        
        # Diagnostic: Log raw sample levels, clipping, and gain settings
        # Format sanity check: raw_max should be in range ~0..2047 for Q11
        if self._health_stats["successful_reads"] % 200 == 0:  # Every 200 reads
            raw_mean = int(np.mean(raw))
            raw_std = int(np.std(raw))
            bt200_status = self.bt200_enabled.get(self.rx_channel, False)
            lna_gain = self.lna_gain_db.get(self.rx_channel, 0)
            
            # Format sanity: log expected range for Q11
            # If raw_max is consistently > 2047, either:
            # 1. You're saturating (reduce gain/LNA/BT200)
            # 2. Format is not actually Q11 (unlikely with bladeRF)
            # 3. DC offset or other issue
            # 4. Hardware issue (values outside Q11 range)
            if format_violation_frac > 0.001:
                logger.warning(f"[SDR] {format_name.upper()} FORMAT VIOLATION: {format_violation_frac*100:.3f}% of samples > {format_max}! "
                             f"{format_name.upper()} valid range is [{-format_max-1}, {format_max}], but seeing values up to {raw_max}. "
                             f"This suggests hardware issue or format mismatch.")
            
            # Check if we're getting values outside expected range - this would indicate format issue
            if arr_max > format_max or arr_min < -format_max-1:
                logger.warning(f"[SDR] {format_name.upper()} FORMAT VIOLATION: arr range [{arr_min}, {arr_max}] exceeds {format_name.upper()} valid range [{-format_max-1}, {format_max}]! "
                             f"violations={format_violation_frac*100:.3f}%. "
                             f"This suggests hardware may be outputting different format than expected. "
                             f"Try switching SPEAR_IQ_SCALING_MODE (current: {settings.IQ_SCALING_MODE}).")
            
            from spear_edge.settings import settings
            scaling_info = getattr(self, '_scaling_mode', f"{settings.IQ_SCALING_MODE} (unknown)")
            logger.info(f"[SDR] Format sanity: raw_max={raw_max} (expected ~0..2047 for Q11), "
                       f"arr_range=[{arr_min}, {arr_max}], raw_mean={int(raw_mean)}, raw_std={int(raw_std)}, "
                       f"rail_frac={rail_frac*100:.3f}%, q11_violations={q11_violation_frac*100:.3f}%, "
                       f"scaling={scaling_info}, "
                       f"gain={self.gain_db:.1f} dB, lna={lna_gain} dB, "
                       f"bt200={'ON' if bt200_status else 'OFF'}, "
                       f"TOTAL_GAIN={self.gain_db + lna_gain + (20 if bt200_status else 0):.1f} dB")
            
            # Warn if raw_max suggests format issue (not Q11)
            if raw_max > 3000 and rail_frac < 0.001:
                logger.warning(f"[SDR] FORMAT WARNING: raw_max={raw_max} is > 3000 but rail_frac={rail_frac*100:.3f}% is low. "
                             f"This suggests format might not be Q11, or there's a scaling issue.")

        if self.dual_channel_mode:
            # Dual channel: interleaved [ch0_i, ch0_q, ch1_i, ch1_q, ch0_i, ch0_q, ...]
            # For now, return channel 0 only (can be extended to return both channels)
            # Extract ch0 I/Q: indices 0, 1, 4, 5, 8, 9, ...
            i = arr[0::4].astype(np.float32, copy=False) * scale  # ch0 I
            q = arr[1::4].astype(np.float32, copy=False) * scale  # ch0 Q
        else:
            # Single channel: [I, Q, I, Q, ...]
            # I: indices 0, 2, 4, ... (even)
            # Q: indices 1, 3, 5, ... (odd)
            i = arr[0::2].astype(np.float32, copy=False) * scale
            q = arr[1::2].astype(np.float32, copy=False) * scale

        # Use pre-allocated buffers for conversion
        np.multiply(i, 1.0, out=self._conv_buf_i[:len(i)])
        np.multiply(q, 1.0, out=self._conv_buf_q[:len(i)])

        # Combine I/Q into complex array
        self._conv_buf_iq[:len(i)].real = self._conv_buf_i[:len(i)]
        self._conv_buf_iq[:len(i)].imag = self._conv_buf_q[:len(i)]
        
        # Calculate DC offset (before removal in scan_task) - this can cause clipping and raise noise floor
        # Calculate early so we can include it in clipping warnings
        dc_offset_i = float(np.mean(i))
        dc_offset_q = float(np.mean(q))
        dc_offset_mag = float(np.sqrt(dc_offset_i**2 + dc_offset_q**2))
        
        # Check for digital full-scale clipping after normalization (Q11: |iq|>=0.98 means near Q11 full-scale)
        iq = self._conv_buf_iq[:len(i)]
        fs_frac = float(np.mean(np.abs(iq) >= 0.98))  # Close to Q11 full-scale in float domain
        
        # Warn if clipping detected (now that we have i/q for DC offset calculation)
        if rail_frac > 0.001:  # More than 0.1% of samples hitting rails
            total_gain = self.gain_db + self.lna_gain_db.get(self.rx_channel, 0) + (20 if self.bt200_enabled.get(self.rx_channel, False) else 0)
            bt200_status = self.bt200_enabled.get(self.rx_channel, False)
            lna_gain = self.lna_gain_db.get(self.rx_channel, 0)
            
            # Recommend safe gain range based on clipping severity
            if rail_frac > 0.05:  # Severe clipping (>5%)
                recommended_gain = max(0, self.gain_db - 20)  # Reduce by 20 dB
            elif rail_frac > 0.01:  # Moderate clipping (>1%)
                recommended_gain = max(0, self.gain_db - 10)  # Reduce by 10 dB
            else:  # Light clipping (>0.1%)
                recommended_gain = max(0, self.gain_db - 5)  # Reduce by 5 dB
            
            # Calculate signal statistics
            normalized_mean = float(np.mean(np.abs(iq)))
            normalized_std = float(np.std(np.abs(iq)))
            signal_ratio = normalized_std / max(normalized_mean, 0.001) if normalized_mean > 0.001 else 0.0
            
            # Determine root cause and provide specific guidance
            cause_msg = ""
            if dc_offset_mag > 0.1:
                cause_msg = f"LARGE DC OFFSET ({dc_offset_mag:.4f}) - Check LO leakage or hardware issues."
            elif normalized_mean > 0.3:
                if signal_ratio < 1.5:
                    cause_msg = f"VERY STRONG WIDEBAND SIGNAL (mean={normalized_mean:.4f}, ratio={signal_ratio:.2f}) - "
                    cause_msg += "Even at 0 dB gain, input is too strong. Solutions: 1) Move/disable nearby transmitter, "
                    cause_msg += "2) Use different frequency, 3) Add external attenuator (10-20 dB)."
                else:
                    cause_msg = f"STRONG NARROWBAND SIGNAL (mean={normalized_mean:.4f}, ratio={signal_ratio:.2f}) - "
                    cause_msg += "Strong signal present. Move transmitter away or use different frequency."
            else:
                cause_msg = f"Signal level high (mean={normalized_mean:.4f}) but gain already at minimum."
            
            rail_type = "int16" if settings.IQ_SCALING_MODE == "int16" else "Q11"
            logger.warning(f"[SDR] ADC CLIPPING: {rail_frac*100:.3f}% at {rail_type} rails, raw_max={raw_max}, "
                         f"TOTAL_GAIN={total_gain:.1f} dB (gain={self.gain_db:.1f}, lna={lna_gain}, bt200={'ON' if bt200_status else 'OFF'}). "
                         f"Signal: mean={normalized_mean:.4f}, std={normalized_std:.4f}, ratio={signal_ratio:.2f}. "
                         f"DC offset: {dc_offset_mag:.4f}. CAUSE: {cause_msg}")
        
        # Log comprehensive diagnostics if clipping or high levels detected
        # Log more frequently when clipping (every 50 reads instead of 200) to catch issues faster
        log_interval = 50 if (rail_frac > 0.001 or raw_max > 1500) else 200
        if (rail_frac > 0.001 or fs_frac > 0.001 or raw_max > 1500) and self._health_stats["successful_reads"] % log_interval == 0:
            normalized_max = float(np.max(np.abs(iq)))
            normalized_mean = float(np.mean(np.abs(iq)))
            normalized_std = float(np.std(np.abs(iq)))
            
            # Calculate signal statistics to identify if there's a strong signal present
            # If std is high relative to mean, there's likely a strong signal
            signal_ratio = normalized_std / max(normalized_mean, 0.001)
            
            logger.info(f"[SDR] Signal levels: raw_max={raw_max}, normalized_max={normalized_max:.4f}, "
                       f"normalized_mean={normalized_mean:.4f}, normalized_std={normalized_std:.4f}, "
                       f"signal_ratio={signal_ratio:.2f}, "
                       f"dc_offset_mag={dc_offset_mag:.4f} (I={dc_offset_i:.4f}, Q={dc_offset_q:.4f}), "
                       f"rail_frac={rail_frac*100:.3f}%, fs_frac={fs_frac*100:.3f}%")
            
            # If DC offset is significant, it could be causing clipping
            if dc_offset_mag > 0.1:
                logger.warning(f"[SDR] LARGE DC OFFSET DETECTED: {dc_offset_mag:.4f} (I={dc_offset_i:.4f}, Q={dc_offset_q:.4f}). "
                             f"This can cause clipping and raise noise floor. Check LO leakage or hardware issues.")
            
            # If signal_ratio is high (>2.0) and mean is high, there's likely a strong signal present
            if signal_ratio > 2.0 and normalized_mean > 0.2:
                logger.warning(f"[SDR] STRONG SIGNAL DETECTED: signal_ratio={signal_ratio:.2f}, mean={normalized_mean:.4f}. "
                             f"Even at 0 dB gain, a very strong nearby transmitter (e.g., ELRS at 915 MHz) "
                             f"can cause clipping. Try: 1) Move transmitter away, 2) Use different frequency, "
                             f"3) Add external attenuator if needed.")

        return self._conv_buf_iq[:len(i)].copy()  # Return copy to avoid buffer reuse issues

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
            "dual_channel_mode": self.dual_channel_mode,
            "lna_gain": {
                "ch0": self.lna_gain_db.get(0, 0),
                "ch1": self.lna_gain_db.get(1, 0),
            },
            "bt200_enabled": {
                "ch0": self.bt200_enabled.get(0, False),
                "ch1": self.bt200_enabled.get(1, False),
            },
        }
