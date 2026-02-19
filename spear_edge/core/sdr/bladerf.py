from __future__ import annotations

import time
import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

from spear_edge.core.sdr.base import SDRBase, SdrConfig, GainMode


class BladeRFSdr(SDRBase):
    """
    bladeRF RX implementation using SoapySDR.

    Supports:
    - bladeRF 2.0 micro
    - bladeRF 2.0 xA4
    """

    def __init__(self, device_args: dict | None = None):
        super().__init__()
        self.device_args = device_args or {"driver": "bladerf"}
        self.dev: SoapySDR.Device | None = None
        self.rx_stream = None
        self.rx_channel = 0

    # -------------------------------------------------
    # Lifecycle
    # -------------------------------------------------

    def open(self) -> None:
        if self.dev is not None:
            return

        self.dev = SoapySDR.Device(self.device_args)

        # Default RX channel
        self.rx_channel = 0

    def close(self) -> None:
        if self.rx_stream is not None:
            self.dev.deactivateStream(self.rx_stream)
            self.dev.closeStream(self.rx_stream)
            self.rx_stream = None

        self.dev = None

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    def apply_config(self, cfg: SdrConfig) -> None:
        if self.dev is None:
            raise RuntimeError("SDR not open")

        ch = cfg.rx_channel

        # Frequency
        self.dev.setFrequency(SOAPY_SDR_RX, ch, cfg.center_freq_hz)

        # Sample rate
        self.dev.setSampleRate(SOAPY_SDR_RX, ch, cfg.sample_rate_sps)

        # Bandwidth (optional)
        if cfg.bandwidth_hz:
            self.dev.setBandwidth(SOAPY_SDR_RX, ch, cfg.bandwidth_hz)

        # Gain
        if cfg.gain_mode == GainMode.AGC:
            self.dev.setGainMode(SOAPY_SDR_RX, ch, True)
        else:
            self.dev.setGainMode(SOAPY_SDR_RX, ch, False)
            self.dev.setGain(SOAPY_SDR_RX, ch, cfg.gain_db)

        self.rx_channel = ch

    # -------------------------------------------------
    # Streaming
    # -------------------------------------------------

    def start_rx(self) -> None:
        if self.dev is None:
            raise RuntimeError("SDR not open")

        if self.rx_stream is not None:
            return

        self.rx_stream = self.dev.setupStream(
            SOAPY_SDR_RX,
            SOAPY_SDR_CF32,
            [self.rx_channel],
        )

        self.dev.activateStream(self.rx_stream)

    def stop_rx(self) -> None:
        if self.rx_stream is None:
            return

        self.dev.deactivateStream(self.rx_stream)
        self.dev.closeStream(self.rx_stream)
        self.rx_stream = None

    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        Read complex64 IQ samples.
        """
        if self.rx_stream is None:
            self.start_rx()

        buff = np.empty(num_samples, dtype=np.complex64)
        sr = self.dev.readStream(self.rx_stream, [buff], num_samples)

        if sr.ret > 0:
            return buff[: sr.ret]

        # Timeout or error
        return np.empty(0, dtype=np.complex64)
