import numpy as np
import threading


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

    def push(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.complex64)
        n = samples.size

        if n == 0:
            return

        with self.lock:
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

    def pop(self, n: int) -> np.ndarray:
        with self.lock:
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

            return out
