# bladeRF 2.0 micro xA4 - Quick API Reference

## Critical Functions Used in SPEAR-Edge

### Device Management
- `bladerf_open()` - Open bladeRF device
- `bladerf_close()` - Close device
- `bladerf_get_device_list()` - Enumerate devices

### RF Configuration (CRITICAL ORDER)
1. `bladerf_set_gain_mode()` - Set AGC/MGC mode (MUST be first if using manual gain)
2. `bladerf_set_sample_rate()` - Set sample rate (MUST be before frequency)
3. `bladerf_set_bandwidth()` - Set bandwidth
4. `bladerf_set_frequency()` - Set center frequency
5. `bladerf_set_gain()` - Set manual gain (only if MGC mode)
6. `bladerf_set_lna_gain()` - Set internal LNA gain (0, 6, 12, 18, 24, 30 dB)
7. `bladerf_set_bias_tee()` - Enable/disable BT200 external LNA (bias-tee)

### Stream Management
- `bladerf_sync_config()` - Configure stream (format, channels, buffers)
- `bladerf_enable_module()` - Enable RX module
- `bladerf_sync_rx()` - Read samples (blocking)
- `bladerf_sync_tx()` - Write samples (blocking)

### Sample Format
- **Format**: `BLADERF_FORMAT_SC16_Q11` (CS16 - interleaved int16 I/Q)
- **Scaling**: int16 values → divide by 32768.0 to get [-1, 1] float range
- **Read Size**: MUST be power-of-two (8192, 16384, 32768, etc.)

### Gain Control
- **Range**: 0-60 dB (typical: 15-35 dB for bladeRF 2.0)
- **Mode**: `BLADERF_GAIN_MGC` (manual) or `BLADERF_GAIN_AGC` (automatic)
- **LNA Gain**: 0, 6, 12, 18, 24, 30 dB (internal LNA)
- **BT200**: External LNA via bias-tee (requires hardware connection)

### Error Codes
- `BLADERF_ERR_TIMEOUT` (-1) - Read timeout (not critical)
- `BLADERF_ERR_INVAL` (-6) - Invalid parameter
- `0` - Success

## Important Notes

1. **Stream Lifecycle**: Configure ALL RF parameters BEFORE setting up stream
2. **Gain Mode**: Must set to MGC before applying manual gain
3. **Read Sizes**: Power-of-two only (hardware requirement)
4. **Sample Scaling**: Use 1/32768.0 for proper normalization
5. **Dual Channel**: Interleaved format [ch0_i, ch0_q, ch1_i, ch1_q, ...]

## See Also

- `libbladeRF.h` - Complete API reference with all functions
- `bladeRF2.h` - bladeRF 2.0 specific definitions
- Official docs: https://github.com/Nuand/bladeRF/wiki
