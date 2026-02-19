import numpy as np
from .base import SDRBase, GainMode


class MockSDR(SDRBase):
    supports_agc = True
    max_rx_channels = 2

    def __init__(self):
        self.is_open = False
        self.center_freq_hz = 0
        self.sample_rate_sps = 0
        self.gain_db = 30.0
        self.gain_mode = GainMode.MANUAL
        self.rx_channel = 0

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def set_rx_channel(self, channel: int):
        self.rx_channel = int(channel)

    def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None):
        self.center_freq_hz = int(center_freq_hz)
        self.sample_rate_sps = int(sample_rate_sps)

    def set_gain(self, gain_db: float):
        self.gain_db = float(gain_db)

    def set_gain_mode(self, mode: GainMode):
        self.gain_mode = mode

    def read_samples(self, num_samples):
        # Fake IQ with noise + weak tone
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        tone = np.exp(2j * np.pi * np.linspace(0, 1, num_samples))
        return (noise + 0.01 * tone).astype(np.complex64)
    
    def get_info(self) -> dict:
        return {
            "driver": "mock",
            "label": "Mock SDR (no hardware)",
            "rx_channels": 1,
            "supports_agc": False,
            "note": "SoapySDR not available or failed to initialize",
        }