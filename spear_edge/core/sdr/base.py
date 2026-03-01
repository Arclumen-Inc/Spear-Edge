from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class GainMode(str, Enum):
    MANUAL = "manual"
    AGC = "agc"


@dataclass
class SdrConfig:
    center_freq_hz: int
    sample_rate_sps: int
    gain_mode: GainMode = GainMode.MANUAL
    gain_db: float = 0.0  # Default 0 dB - user can adjust via UI slider
    rx_channel: int = 0
    bandwidth_hz: Optional[int] = None
    # LNA gain is now automatically optimized by bladerf_set_gain() - no manual control
    bt200_enabled: Optional[bool] = None  # BT200 external LNA enabled (bias-tee on/off). 
                                           # CRITICAL: Default is False (hardware NOT connected). 
                                           # Only set to True if user explicitly enables it.
    dual_channel: bool = False  # Enable dual RX mode (both channels)


class SDRBase:
    """
    Abstract SDR interface.
    All SDR drivers must follow this contract.
    """

    supports_agc: bool = False
    max_rx_channels: int = 1

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def apply_config(self, cfg: SdrConfig):
        """
        Apply a full SDR configuration atomically.
        """
        self.set_rx_channel(cfg.rx_channel)
        
        # Get dual_channel setting (default False for backward compatibility)
        dual_channel = getattr(cfg, 'dual_channel', False)
        
        # CRITICAL: Update internal state BEFORE tune() so tune() uses correct values
        # This ensures gain_mode and gain_db are set correctly when tune() runs
        self.set_gain_mode(cfg.gain_mode)
        if hasattr(self, 'gain_db'):
            self.gain_db = float(cfg.gain_db)
        
        self.tune(
            center_freq_hz=cfg.center_freq_hz,
            sample_rate_sps=cfg.sample_rate_sps,
            bandwidth_hz=cfg.bandwidth_hz,
            dual_channel=dual_channel,
        )
        
        # Ensure gain is set (tune() sets it, but call again to be absolutely sure)
        # LNA gain is now automatically optimized by bladerf_set_gain() - no manual control needed
        if cfg.gain_mode == GainMode.MANUAL:
            self.set_gain(cfg.gain_db)
        
        # Apply BT200 external LNA (bias-tee) if configured
        # CRITICAL: BT200 is NOT connected to the SDR - it should ALWAYS be OFF unless explicitly enabled by user
        # Default is False (BT200 not connected, hardware not present)
        # SAFETY: If gain is very low (0-5 dB), automatically disable BT200 to prevent clipping
        # BT200 adds ~16-20 dB gain which can cause clipping even at low system gain
        # CRITICAL: Only enable if user explicitly sets bt200_enabled=True in config
        bt200_to_apply = False  # Default to OFF (hardware not connected)
        if hasattr(cfg, 'bt200_enabled') and cfg.bt200_enabled is True:
            # Only enable if explicitly set to True (not just truthy, must be exactly True)
            bt200_to_apply = True
        # If not explicitly True, always disable (even if None or False)
        
        if cfg.gain_mode == GainMode.MANUAL and cfg.gain_db <= 5.0 and bt200_to_apply:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"[SDR] SAFETY: Gain is {cfg.gain_db:.1f} dB (very low), but BT200 is enabled. "
                          f"Auto-disabling BT200 to prevent clipping. Set gain > 5 dB to use BT200.")
            bt200_to_apply = False
        
        # Always apply BT200 setting (even if False) to ensure hardware state matches
        # This ensures BT200 is explicitly disabled if not enabled by user
        if hasattr(self, 'set_bt200_enabled'):
            # For dual channel, set BT200 on both channels
            if dual_channel:
                self.set_bt200_enabled(0, bt200_to_apply)
                self.set_bt200_enabled(1, bt200_to_apply)
            else:
                self.set_bt200_enabled(cfg.rx_channel, bt200_to_apply)

    def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None):
        raise NotImplementedError

    def set_gain(self, gain_db: float):
        raise NotImplementedError

    def set_gain_mode(self, mode: GainMode):
        """
        Drivers that do not support AGC must ignore AGC safely.
        """
        return

    def set_rx_channel(self, channel: int):
        """
        Select RX channel (0, 1, ...)
        Drivers with a single channel should clamp to 0.
        """
        return

    def read_samples(self, num_samples):
        raise NotImplementedError
    
    def get_info(self) -> dict:
        """
        Return static + dynamic SDR capabilities for UI.
        Must NEVER throw.
        Concrete drivers should override.
        """
        return {
            "driver": "unknown",
            "label": "Unknown SDR",
            "rx_channels": self.max_rx_channels,
            "supports_agc": self.supports_agc,
            "note": "base SDR (no hardware info)",
        }
    
    def get_health(self) -> dict:
        """
        Return SDR health metrics for monitoring.
        Must NEVER throw.
        Concrete drivers should override.
        """
        return {
            "status": "unknown",
            "success_rate_pct": 0.0,
            "throughput_mbps": 0.0,
            "samples_per_sec": 0.0,
            "avg_read_time_ms": 0.0,
            "errors": 0,
            "timeouts": 0,
            "reads": {"total": 0, "successful": 0},
            "stream": "inactive",
            "usb_speed": "Unknown",
        }