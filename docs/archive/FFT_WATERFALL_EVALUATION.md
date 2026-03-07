# FFT and Waterfall Display Evaluation & Improvement Suggestions

## Executive Summary

This document provides a comprehensive evaluation of the current FFT and waterfall implementation in SPEAR-Edge, comparing it to professional SDR software (SDR++, GQRX, CubicSDR) and providing detailed suggestions for improvements while maintaining the SPEAR green color scheme.

**Current Status**: The implementation is functional and stable, but lacks the visual polish and advanced features found in professional SDR software.

---

## 1. FFT Display Evaluation

### Current Implementation

**Strengths:**
- ✅ Stable autoscaling with noise floor tracking
- ✅ Fast attack/slow decay smoothing (good for FHSS signals)
- ✅ Proper device-pixel ratio handling
- ✅ Shadow effect on trace line
- ✅ Max-hold preservation during downsampling

**Weaknesses:**
- ❌ Single trace line (no peak hold, average, or multiple traces)
- ❌ Basic grid (no minor grid lines, no frequency markers)
- ❌ Limited visual feedback (no peak markers, no cursor readout)
- ❌ Simple line rendering (no fill under curve, no gradient)
- ❌ Fixed 55 dB range (no user control)

### Professional SDR Software Features (SDR++, GQRX)

**SDR++ Features:**
- Multiple trace modes: Instant, Average, Peak Hold, Min Hold
- Adjustable averaging (1-1000 frames)
- Peak markers with frequency/power readout
- Cursor readout (click to see frequency/power)
- Adjustable vertical range (user-controlled dB span)
- Grid with major/minor divisions
- Fill under curve option
- Gradient fill option
- Reference level adjustment
- Multiple FFT windows (Blackman, Hamming, Kaiser, etc.)

**GQRX Features:**
- Peak hold with decay
- Average trace
- Waterfall integration (click waterfall to set frequency)
- Adjustable FFT size on-the-fly
- Multiple display modes

### Improvement Suggestions

#### 1.1 Multiple Trace Modes
**Priority: HIGH**

Add support for multiple trace display modes:
- **Instant** (current): Real-time FFT line
- **Peak Hold**: Maximum values with configurable decay time
- **Average**: Exponential moving average (configurable alpha)
- **Min Hold**: Minimum values (useful for noise floor visualization)

**Implementation Notes:**
- Add UI toggle for trace mode selection
- Store separate arrays for each mode
- Peak hold should decay slowly (e.g., 0.99 per frame)
- Average should be configurable (0.1-0.9 alpha)

#### 1.2 Enhanced Grid System
**Priority: MEDIUM**

Improve grid rendering:
- **Major grid lines**: Every 10 dB (current)
- **Minor grid lines**: Every 2-5 dB (new)
- **Frequency markers**: Vertical lines at major frequency divisions
- **Grid opacity**: Configurable (currently fixed)
- **Grid style**: Dotted vs solid option

**Visual Example:**
```
-20 dB ──────────────── (major, solid)
-22 dB ─ ─ ─ ─ ─ ─ ─ ─ (minor, dotted)
-24 dB ─ ─ ─ ─ ─ ─ ─ ─ (minor, dotted)
-30 dB ──────────────── (major, solid)
```

#### 1.3 Peak Markers and Cursor Readout
**Priority: HIGH**

Add interactive features:
- **Peak markers**: Automatic detection and labeling of top N peaks
- **Cursor readout**: Mouse hover shows frequency/power at cursor
- **Click to center**: Click waterfall/FFT to set center frequency
- **Peak table**: Side panel showing detected peaks with frequency/power

**Implementation:**
- Detect peaks using local maxima (configurable threshold above noise floor)
- Display peak markers as vertical lines with labels
- Add mouse move event handler for cursor readout
- Store peak data in array, update on each frame

#### 1.4 Visual Enhancements
**Priority: MEDIUM**

Improve visual appeal:
- **Fill under curve**: Option to fill area under FFT trace
- **Gradient fill**: Gradient from trace to noise floor
- **Line thickness**: Configurable (currently fixed at 2px)
- **Trace color**: Configurable (currently fixed green)
- **Background grid**: Subtle checkerboard or gradient

#### 1.5 User Controls
**Priority: HIGH**

Add user-adjustable parameters:
- **Vertical range**: User-controlled dB span (currently fixed 55 dB)
- **Reference level**: Adjustable top of display (currently auto)
- **Averaging**: Configurable averaging factor (currently fixed)
- **Peak hold decay**: Configurable decay rate
- **FFT smoothing**: Configurable smoothing factor

---

## 2. Waterfall Display Evaluation

### Current Implementation

**Strengths:**
- ✅ Smooth scrolling (device-pixel correct)
- ✅ Noise-floor anchored scaling
- ✅ SPEAR green color scheme
- ✅ Brightness/contrast controls
- ✅ Gamma correction

**Weaknesses:**
- ❌ Single color scheme (no palette options)
- ❌ Basic color mapping (linear RGB)
- ❌ No time markers (can't see how old data is)
- ❌ No frequency markers overlay
- ❌ Limited color range (only uses green channel effectively)
- ❌ No persistence control (fade rate)

### Professional SDR Software Features

**SDR++ Waterfall:**
- Multiple color palettes (Rainbow, Grayscale, Inverted, etc.)
- Adjustable persistence (fade rate)
- Time markers (vertical lines showing time)
- Frequency markers overlay
- Click to set frequency
- Zoom/pan support
- Histogram equalization
- Color range adjustment (min/max mapping)

**GQRX Waterfall:**
- Multiple color schemes
- Adjustable time span
- Peak hold overlay
- Integration with FFT (click waterfall to tune)

### Improvement Suggestions

#### 2.1 Color Palette Options
**Priority: HIGH**

Add multiple color palettes while keeping SPEAR green as default:
- **SPEAR Green** (current): Green-only scheme
- **Rainbow**: Full spectrum (blue→green→yellow→red)
- **Grayscale**: Black→white
- **Inverted**: White→black
- **Hot**: Black→red→yellow→white
- **Cool**: Black→blue→cyan→white
- **Custom**: User-defined color stops

**Implementation:**
- Create palette functions that map normalized value (0-1) to RGB
- Add UI dropdown for palette selection
- Store palette selection in localStorage

#### 2.2 Enhanced Color Mapping
**Priority: MEDIUM**

Improve color mapping quality:
- **Histogram equalization**: Distribute colors more evenly
- **Logarithmic mapping**: Better visualization of weak signals
- **Color range adjustment**: User-controlled min/max mapping
- **Saturation control**: Adjust color intensity

**Current Issue:**
- Color mapping is linear: `t = (db - dbMin) / (dbMax - dbMin)`
- This compresses weak signals into narrow color range
- Logarithmic mapping would spread colors more evenly

#### 2.3 Time Markers
**Priority: MEDIUM**

Add temporal reference:
- **Vertical time markers**: Lines showing elapsed time
- **Time labels**: Text showing "5s ago", "10s ago", etc.
- **Persistence indicator**: Visual feedback of fade rate
- **Time span control**: User-adjustable waterfall history length

**Implementation:**
- Track frame timestamps
- Draw vertical lines at regular intervals
- Add labels showing time elapsed
- Optionally fade older markers

#### 2.4 Frequency Markers Overlay
**Priority: LOW**

Add frequency reference on waterfall:
- **Vertical frequency lines**: Match FFT frequency markers
- **Frequency labels**: Show frequency at edges
- **Center frequency marker**: Highlight center frequency
- **Bandwidth markers**: Show bandwidth edges

#### 2.5 Persistence Control
**Priority: MEDIUM**

Add configurable persistence:
- **Fade rate**: Adjustable (currently fixed at 0.001 alpha)
- **Persistence time**: User-controlled (e.g., "5 seconds")
- **Fade style**: Linear vs exponential
- **Clear button**: Clear waterfall history

**Current Implementation:**
- Waterfall scrolls down, old data fades via `WF_FADE_ALPHA = 0.0010`
- This is very slow fade (1000 frames to fade completely)
- User should be able to adjust this

#### 2.6 Waterfall Quality Improvements
**Priority: LOW**

Minor visual improvements:
- **Anti-aliasing**: Smooth color transitions
- **Interpolation**: Better color mapping for intermediate values
- **Dithering**: Reduce color banding (optional)
- **Background**: Subtle grid or pattern

---

## 3. Grid and Axes Improvements

### Current Implementation

**Strengths:**
- ✅ Basic power axis (left side)
- ✅ Basic frequency axis (bottom)
- ✅ Center frequency marker

**Weaknesses:**
- ❌ No minor grid lines
- ❌ Fixed tick spacing
- ❌ No axis labels rotation
- ❌ Limited axis information
- ❌ No cursor crosshairs

### Improvement Suggestions

#### 3.1 Enhanced Grid System
**Priority: MEDIUM**

- **Major grid lines**: Every 10 dB, every 1 MHz (configurable)
- **Minor grid lines**: Every 2-5 dB, every 100 kHz (configurable)
- **Grid opacity**: Configurable (currently fixed)
- **Grid style**: Solid, dashed, dotted options
- **Grid color**: Configurable (currently fixed)

#### 3.2 Improved Axis Labels
**Priority: MEDIUM**

- **Frequency axis**: Show more precision (currently 3 decimal places)
- **Power axis**: Show units clearly (dBFS vs dBm)
- **Axis title**: "Frequency (MHz)" and "Power (dBFS)"
- **Label rotation**: Rotate frequency labels if crowded
- **Smart spacing**: Adjust tick spacing based on zoom level

#### 3.3 Cursor Crosshairs
**Priority: HIGH**

Add interactive crosshairs:
- **Vertical line**: Follows mouse X position
- **Horizontal line**: Follows mouse Y position
- **Readout box**: Shows frequency/power at cursor
- **Snap to peaks**: Optional snap to detected peaks
- **Click to center**: Click to set center frequency

---

## 4. Performance Optimizations

### Current Performance

**Good:**
- ✅ Device-pixel ratio handling
- ✅ Efficient downsampling for large FFTs
- ✅ Canvas 2D rendering (works on Jetson)

**Could Improve:**
- ⚠️ Waterfall scrolling (drawImage copy)
- ⚠️ FFT trace rendering (path drawing)
- ⚠️ Color mapping (per-pixel calculation)

### Optimization Suggestions

#### 4.1 Waterfall Rendering
**Priority: LOW**

- **Offscreen canvas**: Pre-render waterfall row, then copy
- **ImageData reuse**: Reuse ImageData objects
- **Row caching**: Cache color-mapped rows for identical power values
- **Reduced precision**: Use Uint8Array for color data

#### 4.2 FFT Trace Rendering
**Priority: LOW**

- **Path optimization**: Use `Path2D` objects for reuse
- **Reduced points**: Further downsampling for very large FFTs
- **GPU acceleration**: Consider WebGL for future (currently not viable on Jetson)

#### 4.3 Color Mapping
**Priority: LOW**

- **Lookup table**: Pre-compute color mapping table
- **Integer math**: Use integer calculations where possible
- **SIMD**: Consider WebAssembly SIMD for bulk operations (future)

---

## 5. User Interface Controls

### Current Controls

**Available:**
- ✅ Brightness slider (waterfall)
- ✅ Contrast slider (waterfall)
- ✅ FFT size selector
- ✅ FPS selector

**Missing:**
- ❌ Trace mode selector (instant/peak/average)
- ❌ Vertical range control
- ❌ Reference level control
- ❌ Averaging control
- ❌ Peak hold decay control
- ❌ Color palette selector
- ❌ Persistence control
- ❌ Grid opacity/style controls

### Suggested UI Layout

```
┌─────────────────────────────────────┐
│ FFT Display Controls                │
├─────────────────────────────────────┤
│ Trace Mode: [Instant ▼]             │
│ Vertical Range: [55 dB ▼]           │
│ Reference Level: [Auto ▼]           │
│ Averaging: [0.18 ▼]                 │
│ Peak Hold Decay: [0.99 ▼]           │
│ Fill Under Curve: [☐]               │
│ Grid Opacity: [50% ━━━━━━━━━━]     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Waterfall Controls                  │
├─────────────────────────────────────┤
│ Color Palette: [SPEAR Green ▼]     │
│ Brightness: [-50 ━━━━━━━━━━ +50]   │
│ Contrast: [0.1 ━━━━━━━━━━ 3.0]     │
│ Persistence: [5s ━━━━━━━━━━]       │
│ Time Span: [30s ━━━━━━━━━━]        │
│ Histogram Equalization: [☐]         │
└─────────────────────────────────────┘
```

---

## 6. Signal Processing Quality

### Current Processing

**Backend (scan_task.py):**
- ✅ Nuttall window (matches SDR++)
- ✅ Coherent gain normalization
- ✅ DC offset removal
- ✅ Noise floor estimation (10th percentile)
- ✅ FFT smoothing (alpha=0.01)
- ✅ SNR calculation

**Frontend (app.js):**
- ✅ Fast attack/slow decay (0.55/0.96)
- ✅ Noise floor tracking
- ✅ Autoscaling

### Potential Improvements

#### 6.1 Window Function Options
**Priority: LOW**

Add multiple window options:
- **Nuttall** (current): Good frequency resolution
- **Blackman**: Better sidelobe suppression
- **Kaiser**: Adjustable parameter
- **Hamming**: Faster computation
- **Rectangular**: No windowing (for comparison)

#### 6.2 Advanced Noise Floor Estimation
**Priority: LOW**

Improve noise floor calculation:
- **Multiple methods**: Percentile, median, mode
- **Adaptive threshold**: Adjust based on signal presence
- **Spatial filtering**: Remove signal bins from floor calculation

#### 6.3 Peak Detection
**Priority: MEDIUM**

Add robust peak detection:
- **Local maxima**: Find peaks above noise floor + threshold
- **Peak width**: Calculate 3 dB bandwidth
- **Peak tracking**: Track peaks across frames
- **False alarm reduction**: Filter spurious peaks

---

## 7. Visual Polish

### Current Visual Style

**Good:**
- ✅ SPEAR green color scheme
- ✅ Dark theme
- ✅ Clean layout

**Could Improve:**
- ⚠️ Font consistency
- ⚠️ Spacing and alignment
- ⚠️ Visual hierarchy
- ⚠️ Animation smoothness

### Polish Suggestions

#### 7.1 Typography
**Priority: LOW**

- **Font consistency**: Use monospace for all numeric displays
- **Font sizes**: Consistent sizing hierarchy
- **Font weights**: Use bold for important values

#### 7.2 Spacing and Alignment
**Priority: LOW**

- **Consistent padding**: Standardize spacing
- **Grid alignment**: Align elements to grid
- **Visual balance**: Better distribution of elements

#### 7.3 Animations
**Priority: LOW**

- **Smooth transitions**: Add CSS transitions for controls
- **Loading states**: Show loading during FFT processing
- **Error states**: Visual feedback for errors

---

## 8. Implementation Priority

### Phase 1: High Priority (Core Features)
1. **Multiple trace modes** (Peak Hold, Average)
2. **Peak markers and cursor readout**
3. **User controls** (vertical range, reference level)
4. **Color palette options** (keep SPEAR green default)
5. **Cursor crosshairs**

### Phase 2: Medium Priority (Enhanced Features)
1. **Enhanced grid system** (minor lines, frequency markers)
2. **Time markers on waterfall**
3. **Persistence control**
4. **Improved color mapping** (histogram equalization)
5. **Fill under curve option**

### Phase 3: Low Priority (Polish)
1. **Advanced window functions**
2. **Peak detection and tracking**
3. **Visual polish** (typography, spacing)
4. **Performance optimizations**

---

## 9. Code Organization Suggestions

### Current Structure

**Frontend (app.js):**
- `drawSpectrum()`: Main rendering function (~300 lines)
- `drawPowerAxis()`: Power axis rendering
- `drawFreqAxis()`: Frequency axis rendering
- Constants scattered throughout

### Suggested Refactoring

**New Structure:**
```
app.js
├── FFTDisplay class
│   ├── constructor(canvas, config)
│   ├── render(frame)
│   ├── setTraceMode(mode)
│   ├── setVerticalRange(range)
│   └── ...
├── WaterfallDisplay class
│   ├── constructor(canvas, config)
│   ├── addRow(powerData)
│   ├── setPalette(palette)
│   └── ...
├── GridRenderer class
│   ├── render(ctx, bounds, config)
│   └── ...
└── Constants
    ├── TRACE_MODES
    ├── COLOR_PALETTES
    └── ...
```

**Benefits:**
- Better code organization
- Easier testing
- Reusable components
- Clearer separation of concerns

---

## 10. Comparison Matrix

| Feature | SPEAR-Edge | SDR++ | GQRX | Priority |
|---------|------------|-------|------|----------|
| Multiple trace modes | ❌ | ✅ | ✅ | HIGH |
| Peak markers | ❌ | ✅ | ✅ | HIGH |
| Cursor readout | ❌ | ✅ | ✅ | HIGH |
| Color palettes | ❌ | ✅ | ✅ | HIGH |
| User-controlled range | ❌ | ✅ | ✅ | HIGH |
| Minor grid lines | ❌ | ✅ | ✅ | MEDIUM |
| Time markers | ❌ | ✅ | ✅ | MEDIUM |
| Persistence control | ❌ | ✅ | ✅ | MEDIUM |
| Fill under curve | ❌ | ✅ | ❌ | MEDIUM |
| Histogram equalization | ❌ | ✅ | ❌ | LOW |
| Multiple windows | ❌ | ✅ | ✅ | LOW |

---

## 11. Conclusion

The current FFT and waterfall implementation is **functional and stable**, but lacks the **visual polish and advanced features** found in professional SDR software. The most impactful improvements would be:

1. **Multiple trace modes** (especially Peak Hold)
2. **Peak markers and cursor readout** (essential for signal analysis)
3. **User controls** (vertical range, reference level)
4. **Color palette options** (while keeping SPEAR green as default)

These improvements would bring SPEAR-Edge's display quality closer to professional SDR software while maintaining the distinctive SPEAR green color scheme.

---

## Appendix: Color Palette Examples

### SPEAR Green (Current)
```javascript
R = Math.floor(30 * t)   // 0-30
G = Math.floor(255 * t)   // 0-255
B = Math.floor(10 * t)    // 0-10
```

### Rainbow Palette
```javascript
// Map t (0-1) to hue (240-0 degrees)
hue = 240 * (1 - t)
R, G, B = hslToRgb(hue, 1.0, 0.5)
```

### Grayscale
```javascript
R = G = B = Math.floor(255 * t)
```

### Hot Palette
```javascript
if (t < 0.33) {
  R = Math.floor(255 * t * 3)
  G = 0
  B = 0
} else if (t < 0.66) {
  R = 255
  G = Math.floor(255 * (t - 0.33) * 3)
  B = 0
} else {
  R = 255
  G = 255
  B = Math.floor(255 * (t - 0.66) * 3)
}
```

---

**Document Version**: 1.0  
**Date**: 2024-12-XX  
**Author**: AI Assistant  
**Status**: Evaluation Complete - Ready for Implementation Planning
