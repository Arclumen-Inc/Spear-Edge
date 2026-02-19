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
    gain_db: float = 30.0
    rx_channel: int = 0
    bandwidth_hz: Optional[int] = None


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
        self.tune(
            center_freq_hz=cfg.center_freq_hz,
            sample_rate_sps=cfg.sample_rate_sps,
            bandwidth_hz=cfg.bandwidth_hz,
        )
        self.set_gain_mode(cfg.gain_mode)
        if cfg.gain_mode == GainMode.MANUAL:
            self.set_gain(cfg.gain_db)

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