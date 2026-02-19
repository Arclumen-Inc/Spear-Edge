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

settings = Settings()
