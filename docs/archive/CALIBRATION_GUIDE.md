# RF Calibration Guide

## Overview

The system displays power in **dBFS** (dB relative to full scale), but you need **dBm** (absolute power) for accurate measurements. The calibration offset converts dBFS to dBm.

## Your Calibration Data (TinySA Ultra)

- **Noise floor**: -90 to -100 dBm
- **ELRS signals**: -30 to -10 dBm

## How to Calculate Calibration Offset

### Step 1: Check Current Display Values

1. Start a scan at 915 MHz with your current settings
2. Note the **dBFS** value shown for:
   - Noise floor (should be around -90 to -110 dBFS)
   - ELRS signal peaks (when they appear)

### Step 2: Calculate Offset

**Formula:** `offset = measured_dBm - displayed_dBFS`

**Examples:**
- If system shows **-95 dBFS** and TinySA shows **-95 dBm** → offset = **0 dB**
- If system shows **-100 dBFS** and TinySA shows **-95 dBm** → offset = **+5 dB**
- If system shows **-90 dBFS** and TinySA shows **-95 dBm** → offset = **-5 dB**

### Step 3: Set Calibration Offset

**Option 1: Environment Variable (Recommended)**
```bash
export SPEAR_CALIBRATION_OFFSET_DB=0.0  # Adjust based on your calculation
```

**Option 2: Edit settings.py**
```python
CALIBRATION_OFFSET_DB: float = float(os.getenv("SPEAR_CALIBRATION_OFFSET_DB", "0.0"))
# Change "0.0" to your calculated offset
```

### Step 4: Verify Calibration

After setting the offset:
1. Restart the system
2. Check that noise floor now shows **-90 to -100 dBm** (matches TinySA)
3. Check that ELRS signals show **-30 to -10 dBm** (matches TinySA)

## Quick Calibration Method

If you want a quick estimate without checking exact values:

1. **Average method**: Use the middle of your noise floor range
   - TinySA noise floor: -90 to -100 dBm → average = **-95 dBm**
   - If system shows around -95 dBFS → offset ≈ **0 dB**
   - If system shows around -100 dBFS → offset ≈ **+5 dB**
   - If system shows around -90 dBFS → offset ≈ **-5 dB**

2. **Signal method**: Use ELRS signal peaks
   - TinySA ELRS: -30 to -10 dBm → average = **-20 dBm**
   - When ELRS signal appears, note the dBFS value
   - Calculate: offset = -20 - displayed_dBFS

## Notes

- Calibration offset is **additive**: `displayed_dBm = displayed_dBFS + offset`
- Offset should be relatively stable across gain settings (within ±5 dB)
- If offset varies significantly with gain, there may be a scaling issue
- Re-calibrate if you change hardware (antenna, cables, etc.)

## Current Configuration

Based on your measurements, a reasonable starting point is:
- **Offset: 0 to +10 dB** (depending on what dBFS values you're seeing)

To find the exact value, compare one measurement point:
1. Note the dBFS value when noise floor is at -95 dBm (middle of your range)
2. Calculate: offset = -95 - displayed_dBFS
