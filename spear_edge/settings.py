from pydantic import BaseModel
import os

class Settings(BaseModel):
    APP_NAME: str = "Spear Edge v1.0"
    HOST: str = os.getenv("SPEAR_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SPEAR_PORT", "8080"))

    # Scan defaults
    DEFAULT_CENTER_FREQ_HZ: int = int(os.getenv("SPEAR_CENTER_FREQ_HZ", "915000000"))
    DEFAULT_SAMPLE_RATE_SPS: int = int(os.getenv("SPEAR_SAMPLE_RATE_SPS", "2400000"))
    DEFAULT_FFT_SIZE: int = int(os.getenv("SPEAR_FFT_SIZE", "2048"))
    DEFAULT_FPS: float = float(os.getenv("SPEAR_FPS", "15.0"))
    
    # RF Calibration (Option A: True SC16_Q11)
    # Option 1: True Q11 dBFS (0.0) - 0 dBFS = Q11 full-scale (±2048)
    # Option 2: SDR++-style 16-bit dBFS (-24.08) - matches SDR++ expectations
    # Set via env var: SPEAR_CALIBRATION_OFFSET_DB=0.0 (Q11) or -24.08 (SDR++-style)
    CALIBRATION_OFFSET_DB: float = float(os.getenv("SPEAR_CALIBRATION_OFFSET_DB", "0.0"))
    
    # IQ Scaling mode (for debugging Q11 vs int16 format issues)
    # "q11" = use 1/2048.0 scaling (Q11 format, per libbladeRF.h)
    # "int16" = use 1/32768.0 scaling (full int16 range, matches SDR++ behavior)
    # Default: "int16" (matches SDR++ and produces correct noise floor levels)
    IQ_SCALING_MODE: str = os.getenv("SPEAR_IQ_SCALING_MODE", "int16").lower()
    
    # DC Removal (for wideband signals like FPV VTX)
    # False = disabled (recommended for wideband signals to avoid distortion)
    # True = enabled (subtract block mean, can distort wideband signals near DC)
    # Default: False (disabled) to preserve wideband signal structure
    # Set via env var: SPEAR_DC_REMOVAL=true
    DC_REMOVAL: bool = os.getenv("SPEAR_DC_REMOVAL", "false").lower() in ("true", "1", "yes")

    # CoT (Cursor on Target) identity — unique per SPEAR instance (uid + callsign). Env overrides file on startup.
    SPEAR_EDGE_ID: str = os.getenv("SPEAR_EDGE_ID", "").strip()

    # GPSD configuration (works with USB GPS or GPIO/UART-backed gpsd sources)
    GPSD_HOST: str = os.getenv("SPEAR_GPSD_HOST", "127.0.0.1")
    GPSD_PORT: int = int(os.getenv("SPEAR_GPSD_PORT", "2947"))
    GPS_POLL_INTERVAL_S: float = float(os.getenv("SPEAR_GPS_POLL_INTERVAL_S", "1.0"))

    # Wi-Fi monitor service (separate from bladeRF SDR path)
    WIFI_MONITOR_AUTOSTART: bool = os.getenv("SPEAR_WIFI_MONITOR_AUTOSTART", "false").lower() in ("true", "1", "yes")
    WIFI_MONITOR_BACKEND: str = os.getenv("SPEAR_WIFI_MONITOR_BACKEND", "kismet").strip().lower()
    # Jetson Orin Nano predictable naming (override with SPEAR_WIFI_MONITOR_IFACE)
    WIFI_MONITOR_IFACE: str = os.getenv("SPEAR_WIFI_MONITOR_IFACE", "wlP1p1s0")
    WIFI_MONITOR_CHANNEL_MODE: str = os.getenv("SPEAR_WIFI_MONITOR_CHANNEL_MODE", "hop").strip().lower()
    WIFI_MONITOR_POLL_INTERVAL_S: float = float(os.getenv("SPEAR_WIFI_MONITOR_POLL_INTERVAL_S", "2.0"))
    WIFI_MONITOR_HOP_CHANNELS: str = os.getenv("SPEAR_WIFI_MONITOR_HOP_CHANNELS", "1,6,11,36,44,149")
    WIFI_MONITOR_KISMET_CMD: str = os.getenv("SPEAR_WIFI_MONITOR_KISMET_CMD", "").strip()
    WIFI_MONITOR_KISMET_URL: str = os.getenv("SPEAR_WIFI_MONITOR_KISMET_URL", "http://127.0.0.1:2501").strip()
    WIFI_MONITOR_KISMET_USERNAME: str = os.getenv("SPEAR_WIFI_MONITOR_KISMET_USERNAME", "").strip()
    WIFI_MONITOR_KISMET_PASSWORD: str = os.getenv("SPEAR_WIFI_MONITOR_KISMET_PASSWORD", "").strip()
    WIFI_MONITOR_KISMET_TIMEOUT_S: float = float(os.getenv("SPEAR_WIFI_MONITOR_KISMET_TIMEOUT_S", "3.0"))

    # Optional spear_manager integration for remote service control
    WIFI_MANAGER_URL: str = os.getenv("SPEAR_WIFI_MANAGER_URL", "http://127.0.0.1:8081").strip()
    WIFI_MANAGER_TOKEN: str = os.getenv("SPEAR_WIFI_MANAGER_TOKEN", "").strip()

settings = Settings()
