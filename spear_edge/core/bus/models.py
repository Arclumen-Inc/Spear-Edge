# spear_edge/core/bus/models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass(frozen=True)
class LiveSpectrumFrame:
    ts: float
    center_freq_hz: int
    sample_rate_sps: int
    fft_size: int
    power_dbfs: List[float]  # max-hold (for FFT line)
    power_inst_dbfs: Optional[List[float]] = None  # instant (for waterfall)
    noise_floor_dbfs: Optional[float] = None
    freqs_hz: Optional[List[float]] = None  # Optional - client can compute from center_freq, sample_rate, fft_size
    meta: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class CueEvent:
    ts: float
    node_id: str
    freq_hz: int
    scan_plan: Optional[str] = None
    confidence: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class CaptureRequest:
    ts: float
    reason: str                 # "tripwire" | "manual" | "tasked"
    freq_hz: float
    sample_rate_sps: int
    duration_s: float
    rx_channel: int = 0
    scan_plan: Optional[str] = None
    priority: int = 0
    source_node: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class CaptureResult:
    ts: float
    request_ts: float
    freq_hz: float
    sample_rate_sps: int
    duration_s: float
    iq_path: str
    meta_path: str
    spec_path: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    source_node: Optional[str] = None
    scan_plan: Optional[str] = None
    stage: Optional[str] = None  # Tripwire detection stage: energy, cue, or confirmed