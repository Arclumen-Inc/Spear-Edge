# File Organization

This document describes the organization of test files and documentation in the SPEAR-Edge project.

## Directory Structure

### `/tests/` - Test Files

All test scripts and test utilities are located in the `tests/` directory:

**Test Scripts:**
- `test_capture_modes.py` - Tests for manual and armed capture modes
- `test_fft_setup.py` - FFT setup and configuration tests
- `test_bladerf_setup.py` - bladeRF hardware setup tests
- `test_fft_normalization.py` - FFT normalization tests
- `test_bladerf_noise.py` - bladeRF noise floor analysis tests
- `test_bladerf_channels.py` - Dual channel tests
- `test_bladerf_native.py` - Native bladeRF driver tests
- `test_bladerf_stress.py` - Stress tests for bladeRF
- `test_edge_integration.py` - Full integration tests
- `test_edge_integration_simple.py` - Simplified integration tests
- `test_aoa_fusion.py` - AoA fusion tests
- `test_max_sample_rate.py` - Maximum sample rate tests
- `test_full_pipeline.py` - Full pipeline tests

**Test Utilities:**
- `torchsig_converter.py` - Utility to convert captures to torchsig format

**Test Output:**
- `test_bladerf_noise_results.log` - Test log files

### `/docs/` - Documentation

All documentation files are located in the `docs/` directory:

**Capture Documentation:**
- `CAPTURE_ARTIFACTS_ANALYSIS.md` - Analysis of capture artifacts for ML inference
- `CAPTURE_TEST_GUIDE.md` - Guide for testing capture modes
- `CAPTURE_MODES_REVIEW.md` - In-depth review of capture modes

**Technical Documentation:**
- `FFT_WATERFALL_CODE_REFERENCE.md` - Reference for FFT and waterfall code
- `BLADERF_ASSESSMENT.md` - bladeRF hardware assessment
- `LIBBLADERF_MIGRATION_SUMMARY.md` - Summary of libbladeRF migration
- `OPTION_A_IMPLEMENTATION.md` - Implementation details for Option A
- `ASSESSMENT.md` - Overall system assessment
- `CHANGELOG.txt` - Project changelog

**Calibration Documentation:**
- `CALIBRATION_GUIDE.md` - RF calibration guide
- `CALIBRATION_SETUP.md` - Calibration setup instructions
- `bladerf_noise_floor_analysis.txt` - Noise floor analysis

**ML Documentation:**
- `ML_INFERENCE_PLAN.txt` - ML inference implementation plan

**Requirements:**
- `Spear Edge Software Requirements.txt` - Software requirements specification

**Integration Guides:**
- `EDGE Integration Guide for Tripwi.txt` - Tripwire integration guide
- `test_aoa_fusion_README.md` - AoA fusion test documentation

### `/references/` - Reference Materials

Reference documentation and external resources:
- `README.md` - References overview
- `API_QUICK_REFERENCE.md` - Quick API reference
- `DOWNLOAD_SOURCES.md` - Download sources
- `FILE_LIST.md` - File listing
- `bladeRF_README.md` - bladeRF reference

## Running Tests

All tests can be run from the project root:

```bash
# Run a specific test
python3 tests/test_capture_modes.py --all

# Run all tests in tests directory
python3 -m pytest tests/

# Run a test utility
python3 tests/torchsig_converter.py <capture_dir> --inspect-only
```

## Documentation Access

All documentation can be accessed from the `docs/` directory:

```bash
# View documentation
cat docs/CAPTURE_ARTIFACTS_ANALYSIS.md
cat docs/CAPTURE_TEST_GUIDE.md

# Or open in your editor
code docs/CAPTURE_ARTIFACTS_ANALYSIS.md
```

## File Organization Rules

1. **Test files** (`test_*.py`) → `/tests/`
2. **Documentation files** (`*.md`, `*.txt`) → `/docs/`
3. **Reference materials** → `/references/`
4. **Root directory** should only contain:
   - Project configuration files (`requirements.txt`, `setup.py`, etc.)
   - Main application code (`spear_edge/`)
   - Data directories (`data/`)
   - Build artifacts (if any)

## Last Updated

2024-12-19 - Organized all test and documentation files into their respective directories.
