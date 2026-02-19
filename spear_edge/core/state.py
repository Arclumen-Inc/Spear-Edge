# spear_edge/core/state.py
"""
Global runtime state (singletons).

This module exists to avoid circular imports between FastAPI app,
routers, and background services.
"""

from typing import Optional

from spear_edge.core.gps.gpsd import GpsdClient

# These will be initialized by app.py
gps: Optional[GpsdClient] = None
engine = None
sdr = None
