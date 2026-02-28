"""
Tripwire v2.0 Event Schemas

Pydantic models for Tripwire v2.0 event types as defined in the
EDGE Integration Guide for Tripwire v2.0.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RfCueEvent(BaseModel):
    """RF Cue Event - Single RF detection event."""
    type: str = Field(default="rf_cue", alias="event_type")
    event_id: Optional[str] = None
    event_group_id: Optional[str] = None
    freq_hz: float
    delta_db: Optional[float] = None
    confidence: Optional[float] = None
    classification: Optional[str] = None
    scan_plan: Optional[str] = None
    bandwidth_hz: Optional[float] = None
    timestamp: float
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
    heading_deg: Optional[float] = None
    
    class Config:
        allow_population_by_field_name = True


class AoaConeEvent(BaseModel):
    """AoA Cone Event - Angle of Arrival cone event."""
    type: str = Field(default="aoa_cone", alias="event_type")
    bearing_deg: float
    cone_width_deg: float
    confidence: Optional[float] = None
    trend: Optional[str] = None  # "closing", "opening", "stable"
    timestamp: float
    multipath_flag: Optional[bool] = None
    strength_db: Optional[float] = None
    delta_db: Optional[float] = None
    noise_db: Optional[float] = None
    center_freq_mhz: Optional[float] = None
    status: Optional[str] = None  # "active", "inactive"
    squelch_passed: Optional[bool] = None
    bearing_history: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        allow_population_by_field_name = True


class FhssClusterEvent(BaseModel):
    """FHSS Cluster Event - Frequency Hopping cluster."""
    type: str = Field(default="fhss_cluster", alias="event_type")
    event_id: Optional[str] = None
    freq_hz: float
    hop_count: Optional[int] = None
    span_mhz: Optional[float] = None
    unique_buckets: Optional[int] = None
    confidence: Optional[float] = None
    classification: Optional[str] = None
    scan_plan: Optional[str] = None
    timestamp: float
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    
    class Config:
        allow_population_by_field_name = True


class RfEnergyEvent(BaseModel):
    """RF Energy Start/End Event."""
    type: str = Field(alias="event_type")  # "rf_energy_start" or "rf_energy_end"
    freq_hz: float
    timestamp: float
    classification: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True


class RfSpikeEvent(BaseModel):
    """RF Spike Event - Transient spike detection."""
    type: str = Field(default="rf_spike", alias="event_type")
    event_id: Optional[str] = None
    freq_hz: float
    delta_db: Optional[float] = None
    confidence: Optional[float] = None
    timestamp: float
    
    class Config:
        allow_population_by_field_name = True


class IbwCalibrationEvent(BaseModel):
    """IBW Calibration Event."""
    type: str = Field(alias="event_type")  # "ibw_calibration_start", "progress", "complete"
    timestamp: float
    progress: Optional[float] = None  # 0.0-1.0 for progress events
    
    class Config:
        allow_population_by_field_name = True


class DfMetricsEvent(BaseModel):
    """Manual DF Metrics Update."""
    type: str = Field(default="df_metrics", alias="event_type")
    timestamp: float
    metrics: Optional[Dict[str, Any]] = None
    
    class Config:
        allow_population_by_field_name = True


class DfBearingEvent(BaseModel):
    """Manual DF Bearing Update."""
    type: str = Field(default="df_bearing", alias="event_type")
    bearing_deg: float
    timestamp: float
    
    class Config:
        allow_population_by_field_name = True


def parse_tripwire_event(data: Dict[str, Any]) -> BaseModel:
    """
    Parse a Tripwire event dict into the appropriate Pydantic model.
    
    Supports both 'type' and 'event_type' fields for backward compatibility.
    """
    event_type = data.get("type") or data.get("event_type", "")
    
    # Map event types to models
    if event_type == "rf_cue":
        return RfCueEvent(**data)
    elif event_type == "aoa_cone":
        return AoaConeEvent(**data)
    elif event_type == "fhss_cluster":
        return FhssClusterEvent(**data)
    elif event_type in ("rf_energy_start", "rf_energy_end"):
        return RfEnergyEvent(**data)
    elif event_type == "rf_spike":
        return RfSpikeEvent(**data)
    elif event_type.startswith("ibw_calibration_"):
        return IbwCalibrationEvent(**data)
    elif event_type == "df_metrics":
        return DfMetricsEvent(**data)
    elif event_type == "df_bearing":
        return DfBearingEvent(**data)
    else:
        # Unknown event type - return as generic dict
        # This maintains backward compatibility with v1.1 events
        from pydantic import create_model
        GenericEvent = create_model("GenericEvent", **{k: (type(v), ...) for k, v in data.items()})
        return GenericEvent(**data)
