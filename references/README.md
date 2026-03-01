# bladeRF 2.0 micro xA4 Reference Documentation

This folder contains reference documentation for the bladeRF 2.0 micro xA4 hardware and libbladerf API.

## Contents

- `bladeRF_README.md` - Main bladeRF repository README
- `bladeRF_CHANGELOG.md` - bladeRF changelog and version history
- `libbladeRF.h` - Complete libbladeRF C API header file
- `libbladerf_index.md` - libbladeRF documentation index (if available)

## Official Sources

- **GitHub Repository**: https://github.com/Nuand/bladeRF
- **Official Website**: https://www.nuand.com/
- **Documentation**: https://github.com/Nuand/bladeRF/wiki

## Key Resources

### Hardware Documentation
- bladeRF 2.0 micro xA4 hardware specifications
- RF frontend characteristics
- Gain stages and LNA specifications
- Bias-tee (BT200) external LNA support

### libbladeRF API
- See `libbladeRF.h` for complete API reference
- Function signatures and constants
- Error codes and return values
- Stream configuration and data formats

### Important Constants (from libbladeRF.h)
- `BLADERF_FORMAT_SC16_Q11` - Sample format (CS16)
- `BLADERF_CHANNEL_RX(ch)` - Channel encoding
- `BLADERF_GAIN_MGC` / `BLADERF_GAIN_AGC` - Gain modes
- Error codes: `BLADERF_ERR_TIMEOUT`, `BLADERF_ERR_INVAL`, etc.

## Notes

- All documentation is downloaded from official Nuand/bladeRF GitHub repository
- For latest updates, check: https://github.com/Nuand/bladeRF
- Hardware-specific details for bladeRF 2.0 micro xA4 variant
