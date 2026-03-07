# Quick Calibration Setup

## Current Situation
- **System shows**: Noise floor around -55 to -70 dBFS (from your screenshot)
- **TinySA shows**: Noise floor -90 to -100 dBm, ELRS signals -30 to -10 dBm
- **System gain**: 15 dB (from screenshot)

## Calculate Calibration Offset

Based on your measurements:
- System noise floor: ~-65 dBFS (average of -55 to -70)
- TinySA noise floor: ~-95 dBm (average of -90 to -100)

**Formula**: `offset = measured_dBm - displayed_dBFS`
**Calculation**: `offset = -95 - (-65) = -30 dB`

## Set Calibration Offset

### Option 1: Environment Variable (Recommended)
```bash
export SPEAR_CALIBRATION_OFFSET_DB=-30.0
```

### Option 2: Edit settings.py
Change line 21 in `spear_edge/settings.py`:
```python
CALIBRATION_OFFSET_DB: float = float(os.getenv("SPEAR_CALIBRATION_OFFSET_DB", "-30.0"))
```

## After Setting Offset

1. **Restart the system**
2. **Check the UI**:
   - Should show "Units: dBm" in the header
   - Power axis should show "dBm" labels
   - Noise floor should show around -90 to -100 dBm
   - ELRS signals should show -30 to -10 dBm when visible

## Fine-Tuning

If the values don't match exactly:
1. Note the current displayed value (e.g., -92 dBm)
2. Note what TinySA shows (e.g., -95 dBm)
3. Adjust offset: `new_offset = current_offset + (tinySA_value - displayed_value)`
4. Example: If showing -92 but should be -95, add -3: `-30 + (-3) = -33`

## Important Notes

- The offset is **additive**: `displayed_dBm = displayed_dBFS + offset`
- If gain changes significantly, you may need to re-calibrate
- The offset should be relatively stable across gain settings (within ±5 dB)
