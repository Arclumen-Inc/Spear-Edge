# spear_edge/core/sdr/soapy.py

from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np
import time

from .base import SDRBase, GainMode

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_CS16
except Exception:
    SoapySDR = None


class SoapySDRDevice(SDRBase):
    """
    Generic SoapySDR-backed SDR driver.

    HARD RULES:
      - Never auto-pick the "audio" Soapy driver
      - read_samples() NEVER throws (returns empty on timeout/error)
      - Methods match SDRBase contract (apply_config() works)
    """

    def __init__(self, device_args: Optional[Dict[str, Any]] = None):
        if SoapySDR is None:
            raise RuntimeError("SoapySDR not available")

        super().__init__()

        self.device_args: Dict[str, Any] = dict(device_args) if device_args else {}
        self.dev: Optional["SoapySDR.Device"] = None
        self.rx_stream = None

        self.driver: Optional[str] = None
        self.rx_channel: int = 0
        self.max_rx_channels: int = 1
        self.supports_agc: bool = False

        self.center_freq_hz: int = 0
        self.sample_rate_sps: int = 0
        self.bandwidth_hz: Optional[int] = None
        self.gain_db: float = 30.0
        self.gain_mode: GainMode = GainMode.MANUAL

        # Overflow counter to prevent log storms
        self._overflow_count = 0

        # Health tracking
        self._health_stats = {
            "total_reads": 0,
            "successful_reads": 0,
            "timeout_reads": 0,
            "error_reads": 0,
            "overflow_errors": 0,
            "total_samples": 0,
            "total_read_time_ns": 0,
            "start_time": time.time(),
        }
        import time as _time
        self._time_module = _time

        self._open_device()

    # ----------------------------
    # Device lifecycle
    # ----------------------------

    def _open_device(self) -> None:
        enum_args = dict(self.device_args)

        raw = SoapySDR.Device.enumerate(enum_args)
        if not raw:
            raise RuntimeError("No SoapySDR devices detected")

        devs: List[Dict[str, Any]] = []
        for d in raw:
            dd = dict(d)
            if dd.get("driver") == "audio":
                continue
            devs.append(dd)

        if not devs:
            raise RuntimeError("Only Soapy 'audio' device found")

        # Prefer bladeRF first if multiple exist
        prefer = ["bladerf", "plutosdr", "hackrf", "rtlsdr", "uhd"]
        devs.sort(
            key=lambda d: prefer.index(d.get("driver")) if d.get("driver") in prefer else 999
        )

        chosen = devs[0]
        print("[SDR] Using Soapy device:", chosen)

        self.dev = SoapySDR.Device(chosen)
        self.driver = chosen.get("driver")

        try:
            self.max_rx_channels = int(self.dev.getNumChannels(SOAPY_SDR_RX))
        except Exception:
            self.max_rx_channels = 1

        try:
            self.supports_agc = bool(self.dev.hasGainMode(SOAPY_SDR_RX, 0))
        except Exception:
            self.supports_agc = False

        # NOTE: Stream setup is deferred until tune() is called
        # bladeRF requires: setSampleRate -> setBandwidth -> setFrequency -> setupStream -> activateStream
        # Do NOT activate stream here - it must happen AFTER tuning

    def _setup_stream(self) -> None:
        if self.dev is None:
            return

        # Tear down existing stream first
        if self.rx_stream is not None:
            try:
                self.dev.deactivateStream(self.rx_stream)
                self.dev.closeStream(self.rx_stream)
            except Exception:
                pass
            self.rx_stream = None

        try:
            # Use CS16 format for better performance at high sample rates
            # CS16 = 4 bytes/sample vs CF32 = 8 bytes/sample (50% bandwidth reduction)
            # This is critical for reliable capture at 10-30 MS/s on Jetson + USB
            self.rx_stream = self.dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [self.rx_channel])
            
            # CRITICAL: Set stream buffer size to prevent overflow (-4 errors)
            # Larger buffers help when reading at high sample rates
            # Note: Not all SoapySDR devices support setStreamBuffers
            # If it fails, we rely on the dedicated reading thread to drain fast enough
            try:
                # Try to set number of buffers (more buffers = more tolerance for read delays)
                if hasattr(self.dev, 'setStreamBuffers'):
                    self.dev.setStreamBuffers(self.rx_stream, 64)  # Increased from 32 to 64
                    print(f"[SDR] Stream buffers set to 64")
                else:
                    # Device doesn't support setStreamBuffers - that's okay
                    pass
            except Exception as e:
                # Silently ignore - not all devices support this
                pass
            
            # Set buffer length if supported (bladeRF-specific)
            try:
                stream_args = {
                    "buffers": "64",
                    "bufferLength": "131072",
                }
                # Note: setupStream args are set at creation, but we can try to configure after
                # Some drivers allow runtime configuration via writeSetting
                if hasattr(self.dev, 'writeSetting'):
                    # bladeRF may support these settings
                    pass
            except Exception:
                pass
            
            self.dev.activateStream(self.rx_stream)
            print(f"[SDR] Stream activated successfully (CS16 format)")
        except Exception as e:
            self.rx_stream = None
            raise RuntimeError(f"SoapySDR RX stream setup failed: {e}")

    def open(self) -> None:
        # already opened in __init__
        return

    def close(self) -> None:
        if self.dev and self.rx_stream:
            try:
                self.dev.deactivateStream(self.rx_stream)
                self.dev.closeStream(self.rx_stream)
            except Exception:
                pass
        self.rx_stream = None
        self.dev = None

    # ----------------------------
    # Configuration
    # ----------------------------

    def set_rx_channel(self, channel: int):
        ch = int(channel)
        if ch < 0 or ch >= self.max_rx_channels:
            ch = 0
        if ch != self.rx_channel:
            self.rx_channel = ch
            # Channel change requires stream rebuild
            # But only if we have valid RF settings (sample_rate > 0)
            # Otherwise, stream will be created when tune() is called
            if self.dev is not None and self.sample_rate_sps > 0:
                self._setup_stream()

    def tune(self, center_freq_hz, sample_rate_sps, bandwidth_hz=None):
        if not self.dev:
            return
    
        ch = int(self.rx_channel)
    
        self.center_freq_hz = int(center_freq_hz)
        self.sample_rate_sps = int(sample_rate_sps)
        self.bandwidth_hz = int(bandwidth_hz) if bandwidth_hz else int(sample_rate_sps)
    
        # bladeRF REQUIRED ORDER
        self.dev.setSampleRate(SOAPY_SDR_RX, ch, float(self.sample_rate_sps))
        self.dev.setBandwidth(SOAPY_SDR_RX, ch, float(self.bandwidth_hz))
        self.dev.setFrequency(SOAPY_SDR_RX, ch, float(self.center_freq_hz))
    
        # bladeRF RX enable kick
        try:
            self.dev.writeSetting("ENABLE_CHANNEL", "RX", "true")
        except Exception:
            pass

        # ✅ STREAM SETUP MUST HAPPEN LAST (after all RF settings)
        # bladeRF requires stream to be created/activated AFTER sample rate/frequency are set
        self._setup_stream()

    def set_gain(self, gain_db: float):
        if self.dev is None:
            return
        self.gain_db = float(gain_db)
        try:
            self.dev.setGain(SOAPY_SDR_RX, self.rx_channel, self.gain_db)
        except Exception:
            pass

    def set_gain_mode(self, mode: GainMode):
        self.gain_mode = mode
        if self.dev is None:
            return
        if not self.supports_agc:
            return
        try:
            self.dev.setGainMode(SOAPY_SDR_RX, self.rx_channel, mode == GainMode.AGC)
        except Exception:
            pass

    # ----------------------------
    # Sampling (never throws)
    # ----------------------------

    def read_samples(self, num_samples: int) -> np.ndarray:
        if self.dev is None or self.rx_stream is None:
            return np.empty(0, dtype=np.complex64)

        n = int(num_samples)
        if n <= 0:
            return np.empty(0, dtype=np.complex64)

        # CS16 format: interleaved int16 (I, Q, I, Q...)
        # Need to read as int16 buffer, then convert to complex64
        cs16_buff = np.empty(n * 2, dtype=np.int16)  # 2 int16 per complex sample

        # Track read time for health metrics
        read_start_ns = time.perf_counter_ns()
        self._health_stats["total_reads"] += 1

        try:
            # Use 100ms timeout for live mode (fast retries prevent overflow)
            # For captures, the capture manager can handle longer waits if needed
            sr = self.dev.readStream(self.rx_stream, [cs16_buff], n, timeoutUs=100_000)
            read_time_ns = time.perf_counter_ns() - read_start_ns
            self._health_stats["total_read_time_ns"] += read_time_ns
            
            # sr.ret: >0 samples, 0 timeout, <0 error
            if sr.ret > 0:
                # Successful read
                self._health_stats["successful_reads"] += 1
                self._health_stats["total_samples"] += sr.ret
                
                # Convert CS16 (interleaved int16) to complex64
                # Scale from int16 range [-32768, 32767] to [-1.0, 1.0]
                iq_int16 = cs16_buff[: sr.ret * 2].reshape(sr.ret, 2)
                buff = np.empty(sr.ret, dtype=np.complex64)
                buff.real = iq_int16[:, 0].astype(np.float32) / 32768.0
                buff.imag = iq_int16[:, 1].astype(np.float32) / 32768.0
                return buff
            elif sr.ret == 0:
                # Timeout
                self._health_stats["timeout_reads"] += 1
                # Timeout - log for debugging but return empty
                # This is normal if stream is not producing data fast enough
                return np.empty(0, dtype=np.complex64)
            else:
                # Error (negative return code)
                self._health_stats["error_reads"] += 1
                # -1 = SOAPY_SDR_TIMEOUT
                # -4 = SOAPY_SDR_OVERFLOW (buffer overflow - reading too slow)
                # -7 = SOAPY_SDR_UNDERFLOW (buffer underflow - reading too fast)
                if sr.ret == -4:
                    self._overflow_count += 1
                    self._health_stats["overflow_errors"] += 1
                    if self._overflow_count % 200 == 0:
                        print(f"[SDR] OVERFLOW x{self._overflow_count} (reading too slowly)")
                elif sr.ret == -7:
                    print(f"[SDR] readStream UNDERFLOW (ret={sr.ret}): Reading too fast, buffer empty")
                else:
                    print(f"[SDR] readStream error: ret={sr.ret}")
                return np.empty(0, dtype=np.complex64)
        except Exception as e:
            self._health_stats["error_reads"] += 1
            print(f"[SDR] readStream exception: {e}")
            return np.empty(0, dtype=np.complex64)

    # ----------------------------
    # Info for UI
    # ----------------------------

    def get_info(self) -> dict:
        if self.dev is None:
            return {
                "driver": self.driver,
                "label": None,
                "serial": None,
                "rx_channels": self.max_rx_channels,
                "supports_agc": self.supports_agc,
            }

        try:
            hw = dict(self.dev.getHardwareInfo() or {})
        except Exception:
            hw = {}

        return {
            "driver": self.driver,
            "label": hw.get("label"),
            "serial": hw.get("serial"),
            "rx_channels": self.max_rx_channels,
            "supports_agc": self.supports_agc,
            "active_rx_channel": self.rx_channel,
        }

    def get_health(self) -> dict:
        """
        Return SDR health metrics for monitoring.
        """
        stats = self._health_stats.copy()
        elapsed_s = time.time() - stats["start_time"]
        
        if elapsed_s < 0.1:
            # Too early for meaningful stats
            return {
                "status": "unknown",
                "success_rate_pct": 0.0,
                "throughput_mbps": 0.0,
                "samples_per_sec": 0.0,
                "avg_read_time_ms": 0.0,
                "errors": 0,
                "timeouts": 0,
                "reads": {"total": 0, "successful": 0},
                "stream": "inactive" if self.rx_stream is None else "active",
                "usb_speed": self._get_usb_speed(),
            }
        
        total_reads = stats["total_reads"]
        successful_reads = stats["successful_reads"]
        success_rate = (successful_reads / total_reads * 100.0) if total_reads > 0 else 0.0
        
        # Throughput: samples/sec * 8 bytes (complex64) = bytes/sec
        samples_per_sec = stats["total_samples"] / elapsed_s
        throughput_mbps = (samples_per_sec * 8) / (1024 * 1024)  # MB/s
        
        # Average read time
        avg_read_time_ms = (stats["total_read_time_ns"] / total_reads / 1_000_000) if total_reads > 0 else 0.0
        
        # Determine status
        if success_rate >= 95.0 and stats["overflow_errors"] == 0:
            status = "good"
        elif success_rate >= 80.0:
            status = "fair"
        else:
            status = "poor"
        
        # Stream status
        stream_status = "active" if self.rx_stream is not None else "inactive"
        if self.rx_stream is not None and self.max_rx_channels > 1:
            stream_status = f"{self.max_rx_channels}-Ch | Active"
        elif self.rx_stream is not None:
            stream_status = "Single-Ch | Active"
        
        return {
            "status": status,
            "success_rate_pct": round(success_rate, 1),
            "throughput_mbps": round(throughput_mbps, 2),
            "samples_per_sec": round(samples_per_sec / 1_000_000, 2),  # MS/s
            "avg_read_time_ms": round(avg_read_time_ms, 2),
            "errors": stats["error_reads"],
            "timeouts": stats["timeout_reads"],
            "reads": {
                "total": total_reads,
                "successful": successful_reads,
            },
            "stream": stream_status,
            "usb_speed": self._get_usb_speed(),
        }
    
    def _get_usb_speed(self) -> str:
        """Try to detect USB speed from device info."""
        if self.dev is None:
            return "Unknown"
        
        try:
            # Try to get USB speed from device settings or hardware info
            hw_info = dict(self.dev.getHardwareInfo() or {})
            # Some SoapySDR drivers expose USB speed
            if "usb_speed" in hw_info:
                speed = hw_info["usb_speed"]
                if "3.0" in str(speed) or "super" in str(speed).lower():
                    return "USB 3.0"
                elif "2.0" in str(speed):
                    return "USB 2.0"
            
            # For bladeRF, check if it's a 2.0 micro (typically USB 3.0)
            if self.driver == "bladerf":
                return "USB 3.0"  # bladeRF 2.0 micro is USB 3.0
        except Exception:
            pass
        
        return "Unknown"
