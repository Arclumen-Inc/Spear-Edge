# FFT/Waterfall Controls UI Design

## Overview

Add interactive controls (checkboxes, dropdowns, sliders) directly in the "RF Spectrum" panel header, similar to professional SDR software like SDR++.

## Current Structure

```html
<div class="panel-header">RF Spectrum</div>
```

## Proposed Structure

```html
<div class="panel-header">
  <span>RF Spectrum</span>
  <div class="spectrum-controls">
    <!-- FFT Controls -->
    <div class="control-group">
      <label class="control-label">
        <input type="checkbox" id="peakHoldCheckbox">
        <span>Peak</span>
      </label>
      <label class="control-label">
        <input type="checkbox" id="averageCheckbox">
        <span>Avg</span>
      </label>
      <select id="traceModeSelect" class="control-select">
        <option value="instant">Instant</option>
        <option value="peak">Peak Hold</option>
        <option value="average">Average</option>
        <option value="min">Min Hold</option>
      </select>
    </div>
    
    <!-- Waterfall Controls -->
    <div class="control-group">
      <select id="waterfallPaletteSelect" class="control-select">
        <option value="spear">SPEAR Green</option>
        <option value="rainbow">Rainbow</option>
        <option value="grayscale">Grayscale</option>
        <option value="hot">Hot</option>
      </select>
      <label class="control-label">
        <input type="checkbox" id="timeMarkersCheckbox">
        <span>Time</span>
      </label>
    </div>
  </div>
</div>
```

## CSS Styling

```css
/* Panel header already has flexbox - just need to style controls */
.panel-header {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--accent-green);
  border-bottom: 1px solid var(--bg-panel-border);
  padding-bottom: 6px;
  margin-bottom: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px; /* Add gap between title and controls */
}

.spectrum-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.control-group {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 8px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  border: 1px solid rgba(60, 255, 158, 0.2);
}

.control-label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-muted);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}

.control-label input[type="checkbox"] {
  width: 14px;
  height: 14px;
  margin: 0;
  cursor: pointer;
  accent-color: var(--accent-green);
}

.control-label input[type="checkbox"]:checked {
  accent-color: var(--accent-green);
}

.control-label:hover {
  color: var(--accent-green);
}

.control-select {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(60, 255, 158, 0.3);
  border-radius: 4px;
  padding: 3px 6px;
  color: var(--text-main);
  font-size: 11px;
  font-family: monospace;
  cursor: pointer;
  min-width: 80px;
}

.control-select:hover {
  border-color: var(--accent-green);
  background: rgba(60, 255, 158, 0.1);
}

.control-select:focus {
  outline: none;
  border-color: var(--accent-green);
  box-shadow: 0 0 6px rgba(60, 255, 158, 0.3);
}
```

## Visual Layout

```
┌─────────────────────────────────────────────────────────────┐
│ RF Spectrum    [Peak☑] [Avg☐] [Instant▼] [SPEAR Green▼] [Time☐] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                    FFT + Waterfall Canvas                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Notes

1. **Responsive Design**: Controls wrap on smaller screens
2. **Compact Labels**: Use short labels ("Peak", "Avg", "Time") to save space
3. **Visual Grouping**: Group related controls with subtle background/border
4. **Hover States**: Highlight controls on hover for better UX
5. **Consistent Styling**: Match existing SPEAR green theme

## Alternative: Collapsible Controls

If space is limited, use a collapsible button:

```html
<div class="panel-header">
  <span>RF Spectrum</span>
  <button id="spectrumControlsToggle" class="controls-toggle-btn">
    ⚙ Controls
  </button>
</div>
<div id="spectrumControlsPanel" class="spectrum-controls-panel hidden">
  <!-- Controls here -->
</div>
```

## Recommended Controls for Phase 1

**FFT Controls:**
- ☑ Peak Hold checkbox
- ☑ Average checkbox  
- [Trace Mode▼] dropdown (Instant/Peak/Average/Min)

**Waterfall Controls:**
- [Palette▼] dropdown (SPEAR Green/Rainbow/Grayscale)
- ☑ Time Markers checkbox

**Future Additions (Phase 2):**
- Vertical Range slider
- Reference Level input
- Grid Opacity slider
- Persistence control
