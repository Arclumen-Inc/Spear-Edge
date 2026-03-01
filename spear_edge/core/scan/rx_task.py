# spear_edge/core/scan/rx_task.py

import asyncio
import threading
import numpy as np


class RxTask:
    """
    Continuously drains SDR into ring buffer using ONE dedicated thread.
    No intermediate queue, no asyncio drain loop.
    Ring buffer is thread-safe, so this is the lowest-jitter path.
    """

    def __init__(self, sdr, ring, chunk_size: int = 65536):
        self.sdr = sdr
        self.ring = ring
        self.chunk_size = int(chunk_size)

        self._running = False
        self._task = None
        self._thread = None
        self._stop_event = threading.Event()

        # stats (optional but very useful)
        self.overflows = 0
        self.empty_reads = 0
        self.read_calls = 0

        # reusable buffer for future optimization (if driver supports it)
        self._reuse = None

    def is_running(self):
        return self._running and self._task and not self._task.done()

    async def start(self):
        if self.is_running():
            return
        self._running = True
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._read_thread, daemon=True, name="rx_read_thread"
        )
        self._thread.start()

        # lightweight async task just to keep lifecycle consistent
        self._task = asyncio.create_task(self._monitor(), name="rx_task_monitor")

    async def stop(self):
        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def _read_thread(self):
        """
        Hard real-time-ish drain loop:
        - read a BIG chunk each call to reduce overhead
        - push immediately into ring
        - never block on other subsystems
        """
        # Choose a sane chunk:
        # - big enough to reduce call rate
        # - not so big that latency becomes awful
        chunk = max(16384, self.chunk_size)

        # If user sets very high SR, increase chunk automatically
        # Example: 30 MS/s => chunk should be 131072 or more
        # You can tune this later; it's safe.
        if getattr(self.sdr, "sample_rate_sps", 0) >= 10_000_000:
            chunk = max(chunk, 131072)

        while not self._stop_event.is_set():
            self.read_calls += 1
            iq = self.sdr.read_samples(chunk)

            if iq is None or iq.size == 0:
                # read_samples() returns empty on timeout/errors
                self.empty_reads += 1
                # tiny wait prevents pure busy-spin when stream is unhappy
                self._stop_event.wait(0.0001)
                continue

            # Push straight into ring (thread-safe)
            self.ring.push(iq)

    async def _monitor(self):
        """
        Optional: print stats every few seconds without touching hot path.
        """
        last = 0
        while self._running:
            await asyncio.sleep(2.0)
            if self.read_calls != last:
                last = self.read_calls
                # Keep prints light
                # print(f"[RxTask] reads={self.read_calls} empty={self.empty_reads} overflows={self.overflows}")
                pass
