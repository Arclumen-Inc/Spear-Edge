# SPEAR-Edge Hardware Guide

## Supported Hardware

### Primary Platform
- **NVIDIA Jetson Orin Nano**
  - CUDA-capable GPU for ML acceleration
  - USB 3.0 ports for bladeRF connection
  - ARM64 architecture

### SDR Hardware
- **bladeRF 2.0 micro** (xA4 or xA9)
  - Frequency range: 47 MHz - 6 GHz
  - Sample rate: Up to 61.44 MS/s (USB 3.0)
  - Dual RX channels
  - 12-bit ADC

## Hardware Setup

### USB Connection

**Requirements:**
- USB 3.0 connection (required for high sample rates)
- USB 2.0 may work for lower rates but not recommended
- Direct connection (no hubs if possible)

**Verification:**
```bash
lsusb | grep -i blade
# Should show: Nuand bladeRF 2.0 micro
```

**USB Speed Check:**
```bash
bladeRF-cli -i
# Check "USB Speed" field (should be "SuperSpeed" for USB 3.0)
```

### Antenna Connection

- Connect antenna to RX port
- Use appropriate antenna for frequency range
- Verify antenna impedance (typically 50Ω)

### External LNA (BT200)

**Important:** BT200 external LNA is **NOT connected by default**.

- BT200 adds ~16-20 dB gain
- Can cause clipping even at low system gain
- Only enable if hardware is connected
- Default: **Disabled** (bt200_enabled=False)

**Enabling BT200:**
- Set `bt200_enabled=True` in SDR config
- Ensure system gain > 5 dB (safety check)
- Monitor for clipping

## RF Configuration

### Frequency Range

- **Supported:** 47 MHz - 6 GHz
- **Default:** 915 MHz (ISM band)
- **Tuning:** Set via center_freq_hz parameter

### Sample Rate

- **Maximum:** 30-40 MS/s (with performance optimizations)
- **Recommended:** 2.4 - 20 MS/s for general use
- **High Performance:** 20-40 MS/s (requires USB 3.0, optimized configuration)
- **Default:** 2.4 MS/s
- **Note:** System automatically optimizes buffers and chunk sizes for high rates

### Bandwidth

- **Default:** Matches sample rate
- **Range:** Limited by sample rate
- **Setting:** bandwidth_hz parameter

### Gain Control

**Gain Modes:**
- **Manual:** Fixed gain (user-controlled)
- **AGC:** Automatic gain control (if supported)

**Gain Components:**
1. **System Gain:** Overall gain (0-60 dB typical)
2. **LNA Gain:** Automatically optimized by driver
3. **BT200 Gain:** External LNA (+16-20 dB, if enabled)

**Gain Selection:**
- Start low (0-10 dB)
- Increase until signal visible
- Avoid clipping (samples max out)
- Target noise floor: -70 to -95 dBFS

**Safety Checks:**
- BT200 auto-disabled if gain ≤ 5 dB
- Prevents clipping from excessive gain

## Hardware Constraints

### bladeRF Stream Lifecycle

**CRITICAL ORDER:**
1. Set sample rate FIRST
2. Set bandwidth
3. Set frequency
4. Enable RX channel
5. Create/activate stream LAST

**Never activate stream before RF parameters are set.**

**Example:**
```python
# ✅ Correct order
sdr.setSampleRate(SOAPY_SDR_RX, ch, sample_rate)
sdr.setBandwidth(SOAPY_SDR_RX, ch, bandwidth)
sdr.setFrequency(SOAPY_SDR_RX, ch, frequency)
sdr.writeSetting("ENABLE_CHANNEL", "RX", "true")
# Stream setup happens last
sdr._setup_stream()
```

### Read Sizes

**Requirement:** Power-of-two read sizes only

**Valid sizes:**
- 8192 (8K)
- 16384 (16K)
- 32768 (32K)
- 65536 (64K)

**Invalid sizes:**
- 10000 (arbitrary)
- 15000 (arbitrary)

**Default:** `BLADE_RF_READ_SAMPLES = 8192`

### Ring Buffer

**Size:** `int(sample_rate_sps * 0.5)` (0.5 seconds)

**Purpose:**
- Buffers IQ samples between RX task and scan task
- Prevents data loss during FFT processing
- Thread-safe (uses threading.Lock)

## Calibration

### IQ Scaling

**Modes:**
- **Q11:** `1/2048.0` scaling (libbladeRF native)
- **int16:** `1/32768.0` scaling (SDR++-style)

**Default:** `int16` (matches SDR++ behavior)

**Configuration:**
```bash
export SPEAR_IQ_SCALING_MODE=int16
```

### RF Calibration Offset

**Purpose:** Calibrate power readings to dBm

**Options:**
- **0.0:** True Q11 dBFS (raw bladeRF values)
- **-24.08:** SDR++-style 16-bit dBFS

**Configuration:**
```bash
export SPEAR_CALIBRATION_OFFSET_DB=0.0
```

**Note:** This is a display-only offset. Backend uses true Q11 scaling internally.

## Performance Tuning

### Sample Rate Limits

**Jetson Orin Nano:**
- Maximum: 30-40 MS/s (with optimizations)
- Recommended: 2.4 - 20 MS/s for general use
- High Performance: 20-40 MS/s (optimized automatically)
- Lower rates = less CPU usage

**Optimization:**
- System automatically optimizes USB buffers and chunk sizes
- Use minimum sample rate for signal bandwidth
- Use decimation for higher rates
- Monitor CPU usage and USB health metrics

### FFT Processing

**FFT Size:**
- Default: 2048
- Larger = better frequency resolution, more CPU
- Smaller = faster processing, less resolution

**FPS:**
- Default: 15
- Higher = smoother display, more CPU
- Lower = less CPU, choppier display

### Memory Usage

**Ring Buffer:**
- Size: `sample_rate * 0.5 * sizeof(complex64)`
- Example: 2.4 MS/s = ~9.6 MB

**Capture Storage:**
- IQ files: `sample_rate * duration * sizeof(complex64)`
- Example: 10 MS/s × 5s = ~400 MB

## Troubleshooting

### Hardware Not Detected

1. **Check USB connection:**
   ```bash
   lsusb | grep -i blade
   ```

2. **Verify libbladeRF:**
   ```bash
   bladeRF-cli --probe
   ```

3. **Check permissions:**
   ```bash
   sudo usermod -a -G plugdev $USER
   # Logout and login
   ```

4. **Verify USB 3.0:**
   - Check USB port (blue = USB 3.0)
   - Check cable (USB 3.0 compatible)
   - Check `bladeRF-cli -i` output

### Stream Errors

**"Stream returns 0 samples":**
- Stream not activated
- Wrong read size (not power-of-two)
- Stream deactivated

**Fix:**
- Verify stream lifecycle order
- Check read size is power-of-two
- Verify stream is active

**"Sample rate reverts to 4 MHz":**
- Stream activated before RF parameters set

**Fix:**
- Set RF parameters before stream activation
- Follow correct lifecycle order

### Gain Issues

**Clipping (samples max out):**
- Gain too high
- BT200 enabled with low system gain

**Fix:**
- Reduce system gain
- Disable BT200 if not needed
- Check gain settings

**Low signal levels:**
- Gain too low
- Antenna not connected
- Wrong frequency

**Fix:**
- Increase gain gradually
- Check antenna connection
- Verify frequency range

### Performance Issues

**High CPU usage:**
- Sample rate too high
- FPS too high
- FFT size too large

**Fix:**
- Reduce sample rate
- Reduce FPS
- Reduce FFT size

**Buffer underruns:**
- CPU too slow for sample rate
- Ring buffer too small

**Fix:**
- Reduce sample rate
- Increase ring buffer size (if possible)

## Hardware Specifications

### bladeRF 2.0 micro

- **Frequency Range:** 47 MHz - 6 GHz
- **Sample Rate:** Up to 61.44 MS/s
- **ADC Resolution:** 12-bit
- **RX Channels:** 2
- **TX Channels:** 2 (not used in Edge)
- **Interface:** USB 3.0
- **Power:** USB powered

### Jetson Orin Nano

- **CPU:** 6-core ARM Cortex-A78AE
- **GPU:** 1024-core NVIDIA Ampere
- **Memory:** 8 GB LPDDR5
- **USB:** USB 3.0 ports
- **Power:** 10-25W

## Best Practices

### Gain Management

1. Start with low gain (0-10 dB)
2. Increase gradually
3. Monitor for clipping
4. Target noise floor: -70 to -95 dBFS

### Sample Rate Selection

1. Use minimum for signal bandwidth
2. Consider CPU limits
3. Test with real hardware
4. Monitor performance

### Frequency Planning

1. Set center frequency to signal of interest
2. Sample rate determines range
3. Range = center ± (sample_rate / 2)
4. Plan for signal bandwidth

### Hardware Safety

1. Never enable BT200 unless hardware connected
2. Monitor for clipping
3. Use appropriate gain levels
4. Follow stream lifecycle order

## References

- [bladeRF Documentation](https://github.com/Nuand/bladeRF/wiki)
- [libbladeRF API Reference](https://github.com/Nuand/bladeRF/blob/master/host/libraries/libbladeRF/include/libbladeRF.h)
- [Jetson Orin Nano Documentation](https://developer.nvidia.com/embedded/jetson-orin-nano)
