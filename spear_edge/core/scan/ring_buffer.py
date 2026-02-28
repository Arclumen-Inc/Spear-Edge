import numpy as np
import threading
import time


class IQRingBuffer:
    """
    High-performance circular buffer for complex64 IQ samples.
    Thread-safe, zero Python object churn.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = np.zeros(self.capacity, dtype=np.complex64)
        self.write_idx = 0
        self.read_idx = 0
        self.size = 0
        self.lock = threading.Lock()
        
        # Metrics for lock contention tracking
        self._push_wait_times = []  # Time spent waiting for lock in push()
        self._pop_wait_times = []   # Time spent waiting for lock in pop()
        self._push_lock_hold_times = []  # Time holding lock in push()
        self._pop_lock_hold_times = []   # Time holding lock in pop()

    def push(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.complex64)
        n = samples.size

        if n == 0:
            return

        # Track lock wait time
        wait_start = time.perf_counter_ns()
        with self.lock:
            wait_time = (time.perf_counter_ns() - wait_start) / 1_000_000  # ms
            if wait_time > 0.001:  # Only track if > 1us
                self._push_wait_times.append(wait_time)
            
            lock_hold_start = time.perf_counter_ns()
            if n >= self.capacity:
                self.buffer[:] = samples[-self.capacity :]
                self.write_idx = 0
                self.read_idx = 0
                self.size = self.capacity
                return

            end = (self.write_idx + n) % self.capacity

            if end < self.write_idx:
                split = self.capacity - self.write_idx
                self.buffer[self.write_idx :] = samples[:split]
                self.buffer[:end] = samples[split:]
            else:
                self.buffer[self.write_idx : end] = samples

            self.write_idx = end
            self.size = min(self.capacity, self.size + n)

            if self.size == self.capacity:
                self.read_idx = self.write_idx
            
            # Track lock hold time
            lock_hold_time = (time.perf_counter_ns() - lock_hold_start) / 1_000_000  # ms
            if lock_hold_time > 0.1:  # Only track if > 0.1ms (significant)
                self._push_lock_hold_times.append(lock_hold_time)

    def pop(self, n: int) -> np.ndarray:
        # Track lock wait time
        wait_start = time.perf_counter_ns()
        with self.lock:
            wait_time = (time.perf_counter_ns() - wait_start) / 1_000_000  # ms
            if wait_time > 0.001:  # Only track if > 1us
                self._pop_wait_times.append(wait_time)
            
            lock_hold_start = time.perf_counter_ns()
            
            if self.size < n:
                return np.empty(0, dtype=np.complex64)

            end = (self.read_idx + n) % self.capacity

            if end < self.read_idx:
                out = np.concatenate(
                    (self.buffer[self.read_idx :], self.buffer[:end])
                )
            else:
                out = self.buffer[self.read_idx : end].copy()

            self.read_idx = end
            self.size -= n
            
            # Track lock hold time
            lock_hold_time = (time.perf_counter_ns() - lock_hold_start) / 1_000_000  # ms
            if lock_hold_time > 0.1:  # Only track if > 0.1ms (significant)
                self._pop_lock_hold_times.append(lock_hold_time)

            return out
    
    def get_lock_metrics(self) -> dict:
        """Get lock contention metrics for debugging."""
        import numpy as np
        
        def stats(times):
            if not times:
                return {"count": 0, "avg_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}
            arr = np.array(times)
            return {
                "count": len(arr),
                "avg_ms": float(np.mean(arr)),
                "max_ms": float(np.max(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
            }
        
        return {
            "push_wait": stats(self._push_wait_times),
            "pop_wait": stats(self._pop_wait_times),
            "push_hold": stats(self._push_lock_hold_times),
            "pop_hold": stats(self._pop_lock_hold_times),
        }