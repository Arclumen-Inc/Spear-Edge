# spear_edge/api/ws/live_fft_ws.py
import json
import struct
import time
import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from spear_edge.settings import settings

MAGIC = b"SPRF"
VERSION = 1
FLAG_HAS_INST = 0x01
HEADER_LEN = 32

_HDR = struct.Struct("<4sBBHIqIff")  
# 4s magic
# B  version
# B  flags
# H  header_len
# I  fft_size
# q  center_freq_hz
# I  sample_rate_sps
# f  ts (float32)
# f  noise_floor_dbfs

async def live_fft_ws(websocket: WebSocket, orchestrator):
    await websocket.accept()
    q = await orchestrator.bus.subscribe("live_spectrum", maxsize=2)  # already small

    # Tell client we'll send binary frames next (still readable/debuggable)
    # Calibration offset: 0.0 = true Q11 dBFS, -24.08 = SDR++-style 16-bit dBFS
    # This is a display-only offset - backend uses true Q11 scaling internally
    await websocket.send_text(json.dumps({
        "type": "hello", 
        "proto": 1, 
        "binary": True,
        "calibration_offset_db": settings.CALIBRATION_OFFSET_DB,
        "power_units": "dBFS",
    }))

    try:
        while True:
            # "Latest wins": if UI is slow, drop older frames
            evt = await q.get()
            try:
                while True:
                    evt = q.get_nowait()
            except Exception:
                pass

            # Pull fields
            fft_size = int(evt.fft_size)
            cf = int(evt.center_freq_hz)
            sr = int(evt.sample_rate_sps)
            ts = float(evt.ts if evt.ts is not None else time.monotonic())
            noise = float(evt.noise_floor_dbfs if evt.noise_floor_dbfs is not None else 0.0)

            # Arrays (convert lists -> float32 bytes)
            p0 = np.asarray(evt.power_dbfs, dtype=np.float32)
            inst = getattr(evt, "power_inst_dbfs", None)
            if inst is not None:
                p1 = np.asarray(inst, dtype=np.float32)
                flags = FLAG_HAS_INST
            else:
                p1 = None
                flags = 0

            # Safety: enforce expected length
            if p0.size != fft_size:
                fft_size = int(p0.size)  # last resort; better than crashing
            if p1 is not None and p1.size != fft_size:
                p1 = None
                flags = 0

            header = _HDR.pack(
                MAGIC,
                VERSION,
                flags,
                HEADER_LEN,
                fft_size,
                cf,
                sr,
                np.float32(ts),     # float32 timestamp
                np.float32(noise),
            )

            if p1 is None:
                payload = header + p0.tobytes(order="C")
            else:
                payload = header + p0.tobytes(order="C") + p1.tobytes(order="C")

            await websocket.send_bytes(payload)

    except WebSocketDisconnect:
        pass
    finally:
        await orchestrator.bus.unsubscribe("live_spectrum", q)
