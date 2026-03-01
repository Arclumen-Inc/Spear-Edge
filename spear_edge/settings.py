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

settings = Settings()
