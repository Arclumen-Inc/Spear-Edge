# bladeRF 2.0 micro xA4 Reference Files

## Downloaded Files

### Core API Headers
- **libbladeRF.h** (159K) - Complete libbladeRF C API with all functions, constants, types, and documentation
- **bladeRF2.h** (17K) - bladeRF 2.0 specific definitions and constants
- **device_calibration.h** - Device calibration structures (if available)

### Documentation
- **bladeRF_README.md** (3.3K) - Main repository README with overview and build instructions
- **bladeRF_CHANGELOG.txt** (66K) - Complete version history and changes
- **API_QUICK_REFERENCE.md** (2.2K) - Quick reference for functions used in SPEAR-Edge

### Reference Guides
- **README.md** (1.5K) - This references folder overview
- **DOWNLOAD_SOURCES.md** (1.5K) - Sources and links for additional documentation

## Key Information from Headers

### Sample Format
- Format: `BLADERF_FORMAT_SC16_Q11` (CS16 - interleaved int16)
- Scaling: Divide by 32768.0 to normalize to [-1, 1] range
- Read sizes: MUST be power-of-two (8192, 16384, etc.)

### Gain Control
- Main gain: 0-60 dB (via `bladerf_set_gain()`)
- LNA gain: 0, 6, 12, 18, 24, 30 dB (via `bladerf_set_lna_gain()`)
- Gain modes: `BLADERF_GAIN_MGC` (manual) or `BLADERF_GAIN_AGC` (automatic)

### Critical Configuration Order
1. Set gain mode (MGC/AGC)
2. Set sample rate
3. Set bandwidth
4. Set frequency
5. Set gain (if MGC)
6. Set LNA gain
7. Configure and activate stream

## Additional Resources

- Official GitHub: https://github.com/Nuand/bladeRF
- Wiki: https://github.com/Nuand/bladeRF/wiki
- Nuand Website: https://www.nuand.com/

