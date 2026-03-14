// ======================================================
// SPEAR-EDGE UI (Clean Baseline - Stable Canvas + FFT/WF)
// - Robust DPR-correct canvas sizing
// - Noise-floor anchored FFT scaling
// - Percentile-based waterfall for good contrast
// - Human-friendly tripwire node cards
// - Real-time operator cue queue with capture/ignore
// - Dedicated notification WebSocket for events
// - Endpoint fallbacks for resilience
// ======================================================

// ------------------------------
// DOM REFERENCES
// ------------------------------
const statusEl          = document.getElementById("status");
const canvas            = document.getElementById("spec");
const ctx               = canvas ? canvas.getContext("2d") : null;
const startBtn          = document.getElementById("start");
const stopBtn           = document.getElementById("stop");
const tripwireNodesEl   = document.getElementById("tripwireNodesTop");
const tripwireHeader    = document.getElementById("tripwireHeaderTop");
const tripwireBody      = document.getElementById("tripwireBodyTop");
const tripwireCaret     = tripwireHeader ? tripwireHeader.querySelector(".caret") : null;
const tripwireLedsEl    = document.getElementById("tripwireLedsTop");
const cueListEl         = document.getElementById("cueList");
const captureLogEl      = document.getElementById("captureLog");
const taskBanner        = document.getElementById("taskBanner");
const taskFromEl        = document.getElementById("taskFrom");
const taskFreqEl        = document.getElementById("taskFreq");
const taskProfileEl     = document.getElementById("taskProfile");
const takHeader         = document.getElementById("takHeaderTop");
const takBody           = document.getElementById("takBodyTop");
const takCaret          = takHeader ? takHeader.querySelector(".caret") : null;
const networkHeader     = document.getElementById("networkHeaderTop");
const networkBody       = document.getElementById("networkBodyTop");
const networkCaret      = networkHeader ? networkHeader.querySelector(".caret") : null;
const l4tbr0Input       = document.getElementById("l4tbr0Input");
const eth0Input         = document.getElementById("eth0Input");
const btnSetL4tbr0      = document.getElementById("btnSetL4tbr0");
const btnSetEth0        = document.getElementById("btnSetEth0");
// modePill removed - mode shown via button highlighting
const armedBanner       = document.getElementById("armedBanner");
const aoaFusionCanvas   = document.getElementById("aoaFusionCanvas");
const aoaFusionCtx      = aoaFusionCanvas ? aoaFusionCanvas.getContext("2d") : null;
const aoaStatusEl       = document.getElementById("aoaStatus");
const aoaConesListEl    = document.getElementById("aoaConesList");
const aoaFusionResultEl = document.getElementById("aoaFusionResult");
let mlClassificationsEl = null; // Will be initialized in init()
const captureBanner     = document.getElementById("captureBanner");
const captureProgressBar = document.getElementById("captureProgressBar");
const wsLed             = document.getElementById("wsLed");
const sdrLed            = document.getElementById("sdrLed");
const gpsLed            = document.getElementById("gpsLed");
const takLed            = document.getElementById("takLed");

const freqInput         = document.getElementById("freqInput");       // MHz
const rateInput         = document.getElementById("rateInput");       // MS/s
const fftSizeSelect     = document.getElementById("fftSizeSelect");
const gainModeSelect    = document.getElementById("gainModeSelect"); // manual|agc
const gainSlider        = document.getElementById("gainSlider");     // Main gain slider (0-60 dB)
const gainValue         = document.getElementById("gainValue");      // Gain value display
const bt200Select       = document.getElementById("bt200Select");     // BT200 external LNA enable
const wfBrightnessSlider = document.getElementById("wfBrightnessSlider");
const wfContrastSlider   = document.getElementById("wfContrastSlider");
const wfBrightnessValue  = document.getElementById("wfBrightnessValue");
const wfContrastValue    = document.getElementById("wfContrastValue");
const btnApplySdr       = document.getElementById("btnApplySdr");
const btnManualCapture  = document.getElementById("btnManualCapture");
const bandwidthInput    = document.getElementById("bandwidthInput");
const fpsSelect         = document.getElementById("fpsSelect");       // FPS

const sdrDriverEl       = document.getElementById("sdr-driver");
const sdrRxPortEl       = document.getElementById("sdr-rx-port");
const sdrCenterEl       = document.getElementById("sdr-center");
const sdrRateEl         = document.getElementById("sdr-rate");
const sdrGainEl         = document.getElementById("sdr-gain");
const sdrGainModeEl     = document.getElementById("sdr-gain-mode");

// edgeModeLabel removed - mode shown in modePill instead
const btnManual         = document.getElementById("btnManual");
const btnArmed          = document.getElementById("btnArmed");

// Spectrum display controls
const traceModeSelect   = document.getElementById("traceModeSelect");
const waterfallPaletteSelect = document.getElementById("waterfallPaletteSelect");
const timeMarkersCheckbox = document.getElementById("timeMarkersCheckbox");


// ------------------------------
// API ENDPOINTS (with fallback)
// ------------------------------
const API = {
  async status() {
    const r = await fetch("/health/status", { cache: "no-store" });
    if (!r.ok) throw new Error("status_unavailable");
    return await r.json();
  },

  async hubNodes() {
    const urls = ["/api/hub/nodes", "/hub/nodes"];
    for (const u of urls) {
      try {
        const r = await fetch(u, { cache: "no-store" });
        if (r.ok) return await r.json();
      } catch (_) {}
    }
    throw new Error("hub_nodes_unavailable");
  },
  async captures() {
    const r = await fetch("/live/captures", { cache: "no-store" });
    if (!r.ok) throw new Error("captures_unavailable");
    return await r.json();
  },

  async sdrInfo() {
    try {
      const r = await fetch("/live/sdr/info", { cache: "no-store" });
      if (r.ok) return await r.json();
    } catch (_) {}
    return null;
  },
  async sdrHealth() {
    try {
      const r = await fetch("/health/sdr", { cache: "no-store" });
      if (r.ok) return await r.json();
    } catch (_) {}
    return null;
  },
  async getNetworkConfig() {
    try {
      const r = await fetch("/api/network/config", { cache: "no-store" });
      if (r.ok) return await r.json();
    } catch (_) {}
    return null;
  },
  async setNetworkInterface(interfaceName, address) {
    const r = await fetch("/api/network/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interface: interfaceName, address }),
    });
    return await r.json().catch(() => ({}));
  },
  async liveStart(payload) {
    const urls = ["/live/start"];
    let lastErr = null;
    for (const u of urls) {
      try {
        const r = await fetch(u, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (r.ok) return await r.json().catch(() => ({}));
        lastErr = new Error(`live_start_failed_${r.status}`);
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error("live_start_failed");
  },
  async sdrConfig(payload) {
    const r = await fetch("/live/sdr/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await r.json().catch(() => ({}));
  },
  /** Apply only gain (for real-time slider). Does not send frequency/rate/bandwidth. */
  async sdrGainOnly(payload) {
    const r = await fetch("/live/sdr/gain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await r.json().catch(() => ({}));
  },
};

// ------------------------------
// CONSTANTS
// ------------------------------
const FFT_HEIGHT_FRAC   = 0.40; // 40% FFT, 60% waterfall
const DB_MIN            = -140;
const DB_MAX            = 20;
const FFT_SMOOTH_ALPHA  = 0.18;
const WF_NOISE_PCT      = 0.20;
const WF_NF_SMOOTH      = 0.06;
const WF_MIN_REL_DB     = 5;
const WF_MAX_REL_DB     = 35;
const WF_GAMMA          = 1.35;
const WF_FADE_ALPHA     = 0.0010;
const CONNECTED_SECS    = 5;

const POLL_STATUS_MS    = 1000;
const POLL_TRIPWIRE_MS  = 1000;
const POLL_CAPTURES_MS  = 2000;
const POLL_SDRINFO_MS   = 2000;
const POLL_SDRHEALTH_MS = 1000; // Update health every second

// ------------------------------
// STATE
// ------------------------------
let fftWs               = null;
let notifyWs            = null;
let edgeMode            = "manual"; // manual | armed | tasked
let activeTask          = null;

let lastSpectrum        = null;
let smoothedNoiseFloor  = null;

// Trace mode state
let traceMode = "instant";  // instant, peak, average, min
let peakHoldSpectrum = null;  // Peak hold trace
let averageSpectrum = null;   // Average trace
let minHoldSpectrum = null;   // Min hold trace
let averageAlpha = 0.3;       // Averaging factor

// Waterfall state
let waterfallPalette = "spear";  // spear, rainbow, grayscale, hot, cool
let timeMarkersEnabled = false;
let waterfallTimestamps = [];     // Track timestamps for time markers

// Stable FFT autoscale state
let fftFloorSmoothed = null;
let fftDbMinSmoothed = null;

// FFT autoscale tuning knobs (stable look)
const FFT_VIEW_RANGE_DB = 70;     // Fixed vertical span (70 dB range for better signal visibility)
const FFT_REFERENCE_LEVEL_DBFS = -20;  // Fixed reference level at top of display (absolute dBFS)
const FFT_FLOOR_MARGIN_DB = 0;    // Noise floor margin from bottom (0 = floor at bottom of range)
const FFT_FLOOR_PCT = 0.02;       // Percentile for floor estimate (2nd percentile = true noise floor, not energy-chasing)
const FFT_MAX_STEP_DB = 0.35;     // Max scale movement per frame (prevents jumping)
const FFT_RISE_ALPHA = 0.18;      // Floor rises faster (attack)
const FFT_FALL_ALPHA = 0.03;      // Floor falls slowly (release)

// Waterfall display controls
let wfBrightness = 0;  // Offset in dB (-50 to +50)
let wfContrast = 1.0;  // Contrast multiplier (0.1 to 3.0)
let lastCanvasW         = 0;
let lastCanvasH         = 0;

// WebSocket reconnection state (FFT)
let fftReconnectTimer   = null;

// ------------------------------
// OPERATOR CUE QUEUE (fixed + robust)
// ------------------------------
const cueBuffer = []; // newest first, max 10

// ------------------------------
// ML CLASSIFICATIONS BUFFER
// ------------------------------
const classificationBuffer = []; // newest first, max 15

function addCue(ev) {
  // Deduplication: ignore near-identical cues within 5s
  const recentDup = cueBuffer.some(c =>
    c.node_id === ev.node_id &&
    Math.abs((c.freq_hz || 0) - (ev.freq_hz || 0)) < 50000 &&  // 50 kHz tolerance
    Date.now() / 1000 - (c.ts || 0) < 5
  );
  if (recentDup) return;

  ev.ts = ev.ts || Date.now() / 1000;
  cueBuffer.unshift(ev);
  if (cueBuffer.length > 10) cueBuffer.pop();
  renderCues();
}

function renderCues() {
  if (!cueListEl) return;

  if (cueBuffer.length === 0) {
    cueListEl.innerHTML = `<div class="muted">No active cues…</div>`;
    return;
  }

  cueListEl.innerHTML = cueBuffer.map((c, idx) => {
    const f = c.freq_hz ? (c.freq_hz / 1e6).toFixed(3) : "-";
    const conf = (c.confidence ?? 0).toFixed(2);
    const node = c.callsign || c.node_id || "Unknown";
    const plan = c.scan_plan ? `${c.scan_plan} ` : "";
    const cls = c.classification || "";

    return `
      <div class="tripwire-card">
        <div class="title">${node}</div>
        <div>Freq: ${f} MHz <span style="opacity:0.8">| conf ${conf}</span></div>
        <div class="muted">${plan}${cls}</div>
        <div style="display:flex;gap:8px;margin-top:12px;">
          <button class="btn-capture" onclick="captureCue(${idx})">CAPTURE</button>
          <button class="btn-ignore" onclick="ignoreCue(${idx})">IGNORE</button>
        </div>
      </div>
    `;
  }).join("");
}

async function captureCue(idx) {
  const cue = cueBuffer[idx];
  if (!cue || !cue.freq_hz) {
    console.warn("[CAPTURE] Invalid cue at index", idx);
    return;
  }

  console.log("[CAPTURE] Operator clicked CAPTURE for cue:", cue);

  try {
    const response = await fetch("/api/capture/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reason: "operator_cue",
        freq_hz: cue.freq_hz,
        duration_s: 5.0,
        source_node: cue.node_id || cue.callsign || "unknown",
        scan_plan: cue.scan_plan || null,
        classification: cue.classification || null
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    console.log("[CAPTURE] Successfully queued on backend");
    cueBuffer.splice(idx, 1);
    renderCues();
  } catch (e) {
    console.error("[CAPTURE] Failed to queue capture:", e);
  }
}

function ignoreCue(idx) {
  cueBuffer.splice(idx, 1);
  renderCues();
}

// ------------------------------
// EDGE MODE
// ------------------------------
async function apiSetEdgeMode(mode) {
  await fetch(`/api/edge/mode/${mode}`, { method: "POST" });
  await refreshEdgeMode();
}

async function refreshEdgeMode() {
  try {
    const r = await fetch(`/api/edge/mode`);
    const j = await r.json();
    const mode = j.mode || "manual";
    // Update global state and UI via setEdgeMode (single source of truth)
    setEdgeMode(mode);
  } catch (_) {}
}

// ------------------------------
// SMALL HELPERS
// ------------------------------
function setLed(idOrEl, on) {
  const el = typeof idOrEl === "string" ? document.getElementById(idOrEl) : idOrEl;
  if (!el) {
    console.warn("[setLed] Element not found:", idOrEl);
    return;
  }
  if (on) {
    el.classList.add("on");
  } else {
    el.classList.remove("on");
  }
}

function clamp(v, lo, hi) {
  v = Number(v);
  return Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : lo;
}

function safeJson(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch (_) {
    return String(obj);
  }
}

// ------------------------------
// CANVAS RESIZE (DPR-correct)
// ------------------------------
function resizeCanvas(force = false) {
  if (!canvas || !ctx) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(1, Math.floor(rect.width));
  const cssH = Math.max(1, Math.floor(rect.height));
  const pxW = Math.max(1, Math.floor(cssW * dpr));
  const pxH = Math.max(1, Math.floor(cssH * dpr));
  if (!force && pxW === lastCanvasW && pxH === lastCanvasH) return;
  canvas.width = pxW;
  canvas.height = pxH;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  lastCanvasW = pxW;
  lastCanvasH = pxH;
  clearWaterfall();
}
window.addEventListener("resize", () => resizeCanvas(false));

// ------------------------------
// MODE + SDR CONTROL LOCK
// ------------------------------
function setEdgeMode(mode) {
  if (!["manual", "armed", "tasked"].includes(mode)) return;
  if (edgeMode === mode) return;
  edgeMode = mode;
  updateModeUI();
}

function updateModeUI() {
  // Update button highlighting - active mode button gets green highlight
  if (btnManual) {
    btnManual.classList.toggle("active", edgeMode === "manual");
  }
  if (btnArmed) {
    btnArmed.classList.toggle("active", edgeMode === "armed");
  }
  
  // Lock SDR controls in tasked mode
  if (edgeMode === "tasked") {
    lockSdrControls();
  } else {
    unlockSdrControls();
  }
  
  if (armedBanner) armedBanner.classList.toggle("active", edgeMode === "armed");
  updateGainUiLock();
}

function lockSdrControls() {
  document.querySelectorAll("#leftColumn input, #leftColumn select, #leftColumn button")
    .forEach(el => { 
      // Don't disable health status button (it's display-only)
      if (el.id !== "sdrHealthStatus") {
        el.disabled = true; 
        el.classList.add("locked"); 
      }
    });
}

function unlockSdrControls() {
  document.querySelectorAll("#leftColumn input, #leftColumn select, #leftColumn button")
    .forEach(el => { 
      // Don't disable health status button (it's display-only)
      if (el.id !== "sdrHealthStatus") {
        el.disabled = false; 
        el.classList.remove("locked"); 
      }
    });
}

function updateGainUiLock() {
  // Gain is controlled via slider - this function kept for compatibility
}

// ------------------------------
// TASK BANNER
// ------------------------------
function setActiveTask(task) {
  activeTask = task || null;
  if (!taskBanner) return;
  if (!activeTask) return clearActiveTask();
  taskBanner.classList.remove("hidden");
  if (taskFromEl) taskFromEl.textContent = `From: ${activeTask.source_node || activeTask.node_id || "-"}`;
  if (taskFreqEl) taskFreqEl.textContent = `Freq: ${(Number(activeTask.freq_hz || 0) / 1e6).toFixed(3)} MHz`;
  if (taskProfileEl) taskProfileEl.textContent = `Profile: ${activeTask.scan_plan || "unknown"}`;
  clearWaterfall();
}

function clearActiveTask() {
  activeTask = null;
  if (taskBanner) taskBanner.classList.add("hidden");
  if (taskFromEl) taskFromEl.textContent = "From: -";
  if (taskFreqEl) taskFreqEl.textContent = "Freq: -";
  if (taskProfileEl) taskProfileEl.textContent = "Profile: -";
}

function clearWaterfall() {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  smoothedNoiseFloor = null;
  lastSpectrum = null;
  // Clear trace buffers
  peakHoldSpectrum = null;
  averageSpectrum = null;
  minHoldSpectrum = null;
  waterfallTimestamps = [];
  window._waterfallStartTime = null;  // Reset waterfall start time
}

// ------------------------------
// SDR CONFIG
// ------------------------------
function readSdrForm() {
  const mhz = Number(freqInput?.value ?? 915.0);
  const msps = Number(rateInput?.value ?? 2.0);
  const fftSize = Number(fftSizeSelect?.value ?? 4096);
  const bandwidthMhz = bandwidthInput?.value ? Number(bandwidthInput.value) : null;
  const fps = Number(fpsSelect?.value ?? 30);
  const bt200Enabled = bt200Select?.value === "true";
  const gainDb = gainSlider?.value ? Number(gainSlider.value) : 0.0;  // Read from slider (default 0)
  return {
    center_freq_hz: Math.round(mhz * 1e6),
    sample_rate_sps: Math.round(msps * 1e6),
    fft_size: Number.isFinite(fftSize) ? fftSize : 4096,
    fps: Number.isFinite(fps) ? fps : 30.0,
    gain_mode: gainModeSelect?.value || "manual",
    gain_db: Number.isFinite(gainDb) ? gainDb : 0.0,  // Read from slider (default 0)
    rx_channel: 0,
    bandwidth_hz: bandwidthMhz ? Math.round(bandwidthMhz * 1e6) : null,
    bt200_enabled: bt200Enabled,
  };
}

// Debounce timer for SDR config changes (frequency, sample rate, bandwidth)
let sdrConfigDebounceTimer = null;

// When user last moved the gain slider (ms). Polling must not overwrite slider for a short period.
let lastGainSliderInputAt = 0;
const GAIN_SLIDER_GRACE_MS = 1500;

async function applySdrConfig() {
  if (edgeMode === "tasked") return;
  const cfg = readSdrForm();
  try {
    const configPayload = {
      center_freq_hz: cfg.center_freq_hz,
      sample_rate_sps: cfg.sample_rate_sps,
      gain_mode: cfg.gain_mode,
      gain_db: cfg.gain_db,
      rx_channel: cfg.rx_channel,
      bandwidth_hz: cfg.bandwidth_hz,
    };
    // Add BT200 if configured
    if (cfg.bt200_enabled !== undefined) {
      configPayload.bt200_enabled = cfg.bt200_enabled;
    }
    await API.sdrConfig(configPayload);
  } catch (e) {
    console.warn("[SDR] config failed:", e);
  }
  refreshStatus();
  pollSdrInfo();
}

// Debounced version of applySdrConfig for input field changes
function applySdrConfigDebounced() {
  // Clear existing debounce timer
  if (sdrConfigDebounceTimer) {
    clearTimeout(sdrConfigDebounceTimer);
  }
  
  // Debounce: Wait 300ms after last input change before applying
  // This prevents rapid API calls when user is typing or changing values
  sdrConfigDebounceTimer = setTimeout(() => {
    if (edgeMode !== "tasked") {
      applySdrConfig();
    }
    sdrConfigDebounceTimer = null;
  }, 300); // 300ms debounce delay
}

async function startManualCapture() {
  if (edgeMode === "tasked") {
    console.warn("[CAPTURE] Cannot start manual capture in TASKED mode");
    return;
  }
  
  const cfg = readSdrForm();
  const durationS = 5.0; // Default 5 second capture
  
  console.log("[CAPTURE] Starting manual capture with SDR settings:", cfg);
  
  try {
    const response = await fetch("/api/capture/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reason: "manual",
        freq_hz: cfg.center_freq_hz,
        sample_rate_sps: cfg.sample_rate_sps,
        bandwidth_hz: cfg.bandwidth_hz,
        gain_mode: cfg.gain_mode,
        gain_db: cfg.gain_db,
        rx_channel: cfg.rx_channel,
        duration_s: durationS,
        source_node: null,
        scan_plan: null,
        classification: null
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const result = await response.json();
    console.log("[CAPTURE] Manual capture queued:", result);
    
    if (result.accepted) {
      // Optionally show a success message or update UI
      console.log("[CAPTURE] Capture started successfully");
    } else {
      console.warn("[CAPTURE] Capture queue is full");
    }
  } catch (e) {
    console.error("[CAPTURE] Failed to start manual capture:", e);
  }
}

// ------------------------------
// AXES + GRID DRAW
// ------------------------------
function drawPowerAxis(ctx, fftH, w, dbMin, dbMax, unit = "dBFS") {
  // Draw Y-axis with absolute dBFS values (not relative to noise floor)
  // dbMin and dbMax are absolute dBFS values from the display range
  ctx.save();
  ctx.fillStyle = "rgba(0,255,136,0.9)";
  ctx.font = "11px monospace";
  ctx.textAlign = "left";
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const t = i / ticks;
    const db = dbMin + t * (dbMax - dbMin);
    const y = fftH - t * fftH;
    // Label shows absolute dBFS value (e.g., -20, -40, -60, -80, -100 dBFS)
    const label = db.toFixed(0) + " " + unit;
    const metrics = ctx.measureText(label);
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(2, y - 12, metrics.width + 8, 14);
    ctx.fillStyle = "rgba(0,255,136,0.9)";
    ctx.fillText(label, 6, y - 2);
  }
  ctx.restore();
}

function drawFreqAxis(ctx, fftH, w, meta) {
  const cf = Number(meta?.center_freq_hz);
  const sr = Number(meta?.sample_rate_sps);
  const n = Number(meta?.fft_size);

  if (!Number.isFinite(cf) || !Number.isFinite(sr) || !Number.isFinite(n) || n <= 0) return;

  ctx.save();
  ctx.fillStyle = "rgba(0,255,136,0.9)";
  ctx.font = "11px monospace";
  ctx.textAlign = "center";

  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const t = i / ticks;
    const x = t * w;

    // Map x ∈ [0,w] to freq ∈ [cf - sr/2, cf + sr/2]
    const fHz = cf + (t - 0.5) * sr;
    const mhz = fHz / 1e6;

    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x - 38, fftH - 18, 76, 14);
    ctx.fillStyle = "rgba(0,255,136,0.9)";
    ctx.fillText(mhz.toFixed(3) + " MHz", x, fftH - 6);
  }
  ctx.restore();
}

// ------------------------------
// COLOR PALETTE FUNCTIONS
// ------------------------------
function getPaletteColor(t, palette) {
  // Clamp t to [0, 1]
  t = Math.max(0, Math.min(1, t));
  
  switch (palette) {
    case "spear":
      // SPEAR Green (current default)
      return {
        r: Math.floor(30 * t),
        g: Math.floor(255 * t),
        b: Math.floor(10 * t)
      };
    
    case "rainbow":
      // Rainbow: blue -> cyan -> green -> yellow -> red
      const hue = 240 * (1 - t);  // 240 (blue) to 0 (red)
      return hslToRgb(hue / 360, 1.0, 0.5);
    
    case "grayscale":
      // Grayscale: black to white
      const gray = Math.floor(255 * t);
      return { r: gray, g: gray, b: gray };
    
    case "hot":
      // Hot: black -> red -> yellow -> white
      if (t < 0.33) {
        return { r: Math.floor(255 * t * 3), g: 0, b: 0 };
      } else if (t < 0.66) {
        return { r: 255, g: Math.floor(255 * (t - 0.33) * 3), b: 0 };
      } else {
        return { r: 255, g: 255, b: Math.floor(255 * (t - 0.66) * 3) };
      }
    
    case "cool":
      // Cool: black -> blue -> cyan -> white
      if (t < 0.5) {
        return { r: 0, g: Math.floor(255 * t * 2), b: Math.floor(255 * t * 2) };
      } else {
        const val = (t - 0.5) * 2;
        return { r: Math.floor(255 * val), g: 255, b: 255 };
      }
    
    default:
      return getPaletteColor(t, "spear");
  }
}

function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return {
    r: Math.round(r * 255),
    g: Math.round(g * 255),
    b: Math.round(b * 255)
  };
}

// ------------------------------
// DRAW SPECTRUM (FFT + WATERFALL)
// ------------------------------
function drawSpectrum(frame) {
  if (!frame) return;

  // ----------------------------
  // COMPATIBILITY SHIM
  // Accept older / alternate backend field names
  // ----------------------------
  if (!frame.power_dbfs && Array.isArray(frame.power_db)) frame.power_dbfs = frame.power_db;
  if (!frame.power_dbfs && Array.isArray(frame.power)) frame.power_dbfs = frame.power;

  if (!frame.power_inst_dbfs && Array.isArray(frame.power_inst_db)) frame.power_inst_dbfs = frame.power_inst_db;
  if (!frame.power_inst_dbfs && Array.isArray(frame.power_inst)) frame.power_inst_dbfs = frame.power_inst;

  if (!frame.freqs_hz && Array.isArray(frame.freqs)) frame.freqs_hz = frame.freqs;

  // Guard: must have at least a few bins
  // Fix: Float32Array is NOT an Array, so use length check instead of Array.isArray
  if (!ctx || !canvas || !frame.power_dbfs || frame.power_dbfs.length < 8) return;
  
  if (canvas.width < 50 || canvas.height < 50) {
    resizeCanvas(true);
    return;
  }
  
  // Compute device-space coordinates once
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  const pxW = canvas.width;
  const pxH = canvas.height;
  let pxFftH = Math.floor(cssH * FFT_HEIGHT_FRAC * dpr);
  // When clientHeight is 0 or tiny, pxFftH becomes 0 and the waterfall scroll copies the whole
  // canvas (including the FFT line), making the line appear to move down each frame. Use canvas
  // buffer height so the FFT region is always the top 40% and never zero-height.
  if (pxFftH <= 0 && pxH > 0) {
    pxFftH = Math.max(1, Math.floor(pxH * FFT_HEIGHT_FRAC));
  }
  const pxWfH  = pxH - pxFftH;

  // CSS-space for FFT drawing
  const w = (canvas.width / dpr) || 1;
  const h = (canvas.height / dpr) || 1;
  const fftH = Math.floor(h * FFT_HEIGHT_FRAC);
  const wfH = h - fftH;

  // For wideband signals, use smoothed power for FFT line (better visibility)
  // For narrowband/bursty signals, use instant power (shows bursts)
  // Detect wideband by checking if noise floor is very low (< -75 dBFS) or signal spans many bins
  const noiseFloor = frame.noise_floor_dbfs !== undefined ? frame.noise_floor_dbfs : null;
  const tempArr = frame.power_inst_dbfs || frame.power_dbfs || [];
  const signalSpan = tempArr.length > 0 ? (Math.max(...tempArr) - (noiseFloor !== null ? noiseFloor : Math.min(...tempArr))) : 0;
  const isWideband = noiseFloor !== null && noiseFloor < -75.0;  // Low noise floor indicates wideband signal
  
  // Use smoothed for wideband (better for analog video), instant for narrowband (better for bursts)
  const fftArr = (isWideband && frame.power_dbfs) ? frame.power_dbfs : (frame.power_inst_dbfs || frame.power_dbfs);
  const wfArr = frame.power_inst_dbfs || frame.power_dbfs;  // instant (for waterfall)
  
  // Store diagnostic data for logging after display range is calculated
  let diagnosticData = null;
  if (!window._lastFftDiag || (Date.now() - window._lastFftDiag) > 2000) {
    const fftMin = Math.min(...fftArr.map(Number).filter(Number.isFinite));
    const fftMax = Math.max(...fftArr.map(Number).filter(Number.isFinite));
    const fftMean = fftArr.map(Number).filter(Number.isFinite).reduce((a, b) => a + b, 0) / fftArr.length;
    const backendNoiseFloor = frame.noise_floor_dbfs !== undefined ? frame.noise_floor_dbfs : null;
    
    // Frontend noise floor calculation (what we actually use)
    const sortedF = fftArr
      .map(Number)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const idx = Math.max(0, Math.floor(sortedF.length * FFT_FLOOR_PCT));
    const frontendFloorRaw = sortedF.length ? sortedF[idx] : -80;
    
    // Edge bin analysis
    const edgeBinCount = Math.max(10, Math.floor(fftArr.length * 0.05));  // 5% or 10 bins
    const edgeBinsStart = fftArr.slice(0, edgeBinCount).map(Number).filter(Number.isFinite);
    const edgeBinsEnd = fftArr.slice(-edgeBinCount).map(Number).filter(Number.isFinite);
    const centerStart = Math.floor(fftArr.length / 2) - Math.floor(edgeBinCount / 2);
    const centerEnd = Math.floor(fftArr.length / 2) + Math.floor(edgeBinCount / 2);
    const centerBins = fftArr.slice(centerStart, centerEnd).map(Number).filter(Number.isFinite);
    
    const edgeStartMean = edgeBinsStart.reduce((a, b) => a + b, 0) / edgeBinsStart.length;
    const edgeStartMax = Math.max(...edgeBinsStart);
    const edgeEndMean = edgeBinsEnd.reduce((a, b) => a + b, 0) / edgeBinsEnd.length;
    const edgeEndMax = Math.max(...edgeBinsEnd);
    const centerMean = centerBins.reduce((a, b) => a + b, 0) / centerBins.length;
    const centerMax = Math.max(...centerBins);
    
    // Noise floor with and without edge bins
    const excludeCount = Math.floor(fftArr.length * 0.05);
    const centerSpectrum = fftArr.slice(excludeCount, -excludeCount).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    const floorWithoutEdgesIdx = Math.max(0, Math.floor(centerSpectrum.length * FFT_FLOOR_PCT));
    const floorWithoutEdges = centerSpectrum.length ? centerSpectrum[floorWithoutEdgesIdx] : frontendFloorRaw;
    
    diagnosticData = {
      fftMin, fftMax, fftMean, backendNoiseFloor, frontendFloorRaw, floorWithoutEdges,
      edgeBinCount, edgeStartMean, edgeStartMax, edgeEndMean, edgeEndMax,
      centerMean, centerMax, centerBinsLength: centerBins.length
    };
  }
  
  // Diagnostic logging moved to after dbMax is calculated (see below, after line 1076)
  
  // Determine power units from frame metadata or global calibration FIRST
  // Priority: frame metadata > global calibration > default (dBFS)
  // This must be done BEFORE auto-scaling so we can adjust TARGET_FLOOR_DB appropriately
  const calibrationOffset = frame.calibration_offset_db !== undefined ? 
                            Number(frame.calibration_offset_db) : 
                            globalCalibrationOffset;
  const powerUnits = frame.power_units || 
                     globalPowerUnits ||
                     (Math.abs(calibrationOffset) > 0.1 ? "dBm" : "dBFS");

  // ================================
  // WATERFALL (DPR-CORRECT, STABLE)
  // Must happen FIRST - before FFT clearing/drawing
  // ================================
  if (frame.power_inst_dbfs && wfArr && wfArr.length > 0 && pxWfH > 0) {
    // Reset transform to identity for device-pixel operations
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Scroll waterfall down by 1 device pixel
    if (pxWfH > 1) {
      ctx.drawImage(
        canvas,
        0, pxFftH,
        pxW, pxWfH - 1,
        0, pxFftH + 1,
        pxW, pxWfH - 1
      );
    }

    // ---- Stable waterfall scaling (pre-WebGL behavior) ----
    // Use 10th percentile for noise floor (more robust than median for waterfall)
    const wfSorted = wfArr
      .map(Number)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const wfFloorIdx = Math.max(0, Math.floor(wfSorted.length * 0.10));
    const wfNoiseFloor = wfSorted.length ? wfSorted[wfFloorIdx] : -75;
    
    if (smoothedNoiseFloor == null) {
      smoothedNoiseFloor = wfNoiseFloor;
    } else {
      // Slow drift toward actual noise floor
      smoothedNoiseFloor = 0.95 * smoothedNoiseFloor + 0.05 * wfNoiseFloor;
    }

    // Waterfall range: noise floor - 15 dB to noise floor + 50 dB
    // This gives good visibility of signals while showing noise floor context
    const wfDbMin = smoothedNoiseFloor - 15;
    const wfDbMax = smoothedNoiseFloor + 50;
    const wfRange = Math.max(1, wfDbMax - wfDbMin);

    // Draw ONE row (device space)
    const row = ctx.createImageData(pxW, 1);
    const data = row.data;
    const nBins = wfArr.length;

    for (let x = 0; x < pxW; x++) {
      const idx = Math.min(Math.floor(x * nBins / pxW), nBins - 1);
      const db = Number(wfArr[idx]);

      if (!Number.isFinite(db)) continue;

      // Normalize to 0-1 range
      let t = (db - wfDbMin) / wfRange;
      
      // Apply brightness: shift the normalized value
      // Brightness is in dB, convert to normalized offset
      const brightnessOffset = wfBrightness / wfRange;
      t = t + brightnessOffset;
      
      // Apply contrast: center around 0.5, multiply, then restore
      t = ((t - 0.5) * wfContrast) + 0.5;
      
      // Clamp and apply gamma
      t = Math.max(0, Math.min(1, t));
      t = Math.pow(t, WF_GAMMA);

      // Get color from selected palette
      const color = getPaletteColor(t, waterfallPalette);
      
      const o = x * 4;
      data[o + 0] = color.r;
      data[o + 1] = color.g;
      data[o + 2] = color.b;
      data[o + 3] = 255;
    }

    ctx.putImageData(row, 0, pxFftH);
    
    // Draw time markers if enabled
    if (timeMarkersEnabled) {
      const now = frame.ts ? frame.ts * 1000 : Date.now();
      const fps = 30; // Approximate FPS
      
      // Track start time if not set
      if (!window._waterfallStartTime) {
        window._waterfallStartTime = now;
      }
      
      // Draw markers every 5 seconds
      const markerInterval = 5000; // 5 seconds
      const elapsed = now - window._waterfallStartTime;
      const rowsScrolled = Math.floor((elapsed / 1000) * fps);
      
      ctx.strokeStyle = "rgba(60, 255, 158, 0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.fillStyle = "rgba(60, 255, 158, 0.8)";
      ctx.font = "10px monospace";
      ctx.textAlign = "left";
      
      // Draw markers for each 5-second interval
      for (let t = 0; t <= elapsed; t += markerInterval) {
        const ageSeconds = t / 1000;
        if (ageSeconds > 0 && ageSeconds <= 30) {
          const markerRows = Math.floor((t / 1000) * fps);
          const markerY = pxFftH + markerRows;
          
          if (markerY >= pxFftH && markerY < pxH) {
            ctx.beginPath();
            ctx.moveTo(0, markerY);
            ctx.lineTo(pxW, markerY);
            ctx.stroke();
            
            // Add label
            ctx.fillText(`${Math.floor(ageSeconds)}s`, 4, markerY - 2);
          }
        }
      }
      
      ctx.setLineDash([]);
    }
    
    ctx.restore();
  }

  // ---------
  // Stable FFT autoscale (fixed span, smooth floor tracking)
  // - Fixed vertical span (70 dB) for stable look
  // - Uses backend noise floor (more stable and accurate)
  // - Smoothly tracks noise floor with asymmetrical attack/release
  // - Prevents jumping (max 0.35 dB per frame movement)
  // - Makes weak signals visible by following true noise floor
  // ---------
  
  // USE BACKEND NOISE FLOOR (more stable and accurate)
  // Backend uses adaptive percentile: 2nd percentile for wideband signals, 10th percentile for narrowband
  // This excludes signal energy from noise floor calculation for wideband signals
  // This is more accurate than frontend recalculation
  const backendFloor = frame.noise_floor_dbfs !== undefined ? frame.noise_floor_dbfs : null;
  
  // Fallback: calculate from frontend if backend value not available
  let floorNow;
  if (backendFloor !== null && Number.isFinite(backendFloor)) {
    // Use backend noise floor directly
    floorNow = backendFloor;
  } else {
    // Fallback: calculate from frontend (excluding edge bins)
    const excludePct = 0.05;  // Exclude 5% from each edge
    const excludeCount = Math.floor(fftArr.length * excludePct);
    const centerSpectrum = fftArr.slice(excludeCount, -excludeCount)
      .map(Number)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const idx = Math.max(0, Math.floor(centerSpectrum.length * FFT_FLOOR_PCT));
    floorNow = centerSpectrum.length ? centerSpectrum[idx] : -80;
  }

  // Smoothed floor with asymmetrical attack/release
  // Rises faster (attack) when floor increases, falls slowly (release) when floor decreases
  // CRITICAL: If noise floor changes significantly (>15 dB), reset immediately (wideband signal detected)
  // This prevents slow adaptation when backend switches from contaminated to true noise floor
  // Lowered threshold from 20 dB to 15 dB to catch the -67.8 to -87.0 dBFS transition
  if (fftFloorSmoothed == null) {
    fftFloorSmoothed = floorNow;
  } else {
    const floorDelta = Math.abs(floorNow - fftFloorSmoothed);
    if (floorDelta > 15.0) {
      // Large change detected: likely wideband signal or noise floor fix applied
      // Reset immediately to new value
      console.log(`[FFT UI] Noise floor reset: ${fftFloorSmoothed.toFixed(1)} → ${floorNow.toFixed(1)} dBFS (delta: ${floorDelta.toFixed(1)} dB)`);
      fftFloorSmoothed = floorNow;
    } else {
      const alpha = (floorNow > fftFloorSmoothed) ? FFT_RISE_ALPHA : FFT_FALL_ALPHA;
      fftFloorSmoothed = (1 - alpha) * fftFloorSmoothed + alpha * floorNow;
    }
  }

  // Fixed reference level display (absolute dBFS values)
  // Reference level is fixed at top, display range autoscales to show noise floor at bottom
  // This ensures Y-axis labels show absolute dBFS values, not relative values
  
  // Target dbMin so noise floor sits at the bottom of the chart
  const dbMinTarget = fftFloorSmoothed - FFT_FLOOR_MARGIN_DB;
  
  // Clamp movement per frame to prevent hopping
  if (fftDbMinSmoothed == null) {
    fftDbMinSmoothed = dbMinTarget;
  }
  const delta = dbMinTarget - fftDbMinSmoothed;
  const step = Math.max(-FFT_MAX_STEP_DB, Math.min(FFT_MAX_STEP_DB, delta));
  fftDbMinSmoothed += step;

  // Calculate display range with fixed reference level at top
  // If noise floor is very low, extend range downward (but keep reference level fixed)
  let dbMin = fftDbMinSmoothed;
  let dbMax = dbMin + FFT_VIEW_RANGE_DB;  // Default: fixed span from noise floor
  
  // If calculated max exceeds reference level, use reference level and extend downward
  if (dbMax > FFT_REFERENCE_LEVEL_DBFS) {
    dbMax = FFT_REFERENCE_LEVEL_DBFS;
    // Extend range downward if needed to show noise floor
    const actualRange = dbMax - dbMin;
    if (actualRange < FFT_VIEW_RANGE_DB) {
      // Range is smaller than desired, but we're constrained by reference level
      // This is OK - we'll show what we can
    }
  } else {
    // Noise floor is high enough that we can use full range below reference level
    // Keep the calculated range
  }
  
  // WIDEBAND SIGNAL OPTIMIZATION: Use fixed 45 dB range for wideband signals
  // Wideband signals have low peak-to-noise ratio, so a 70 dB range compresses them too much
  // When noise floor is very low (< -75 dBFS), it indicates wideband signal is present
  // Use a fixed 45 dB range from noise floor for stable, consistent display
  let widebandOptimizationApplied = false;
  if (fftFloorSmoothed !== null && fftFloorSmoothed < -75.0) {
    // Very low noise floor indicates wideband signal (true noise is low, signal spreads power)
    // Use fixed 45 dB range from noise floor for stability
    const widebandRangeDb = 45.0;  // Fixed 45 dB range for wideband signals
    
    const newDbMax = Math.min(FFT_REFERENCE_LEVEL_DBFS, fftFloorSmoothed + widebandRangeDb);
    const newDbMin = fftFloorSmoothed;
    
    // Only apply if current range is much larger (prevents constant adjustment)
    const currentRange = dbMax - dbMin;
    if (currentRange > 50.0 && newDbMax > newDbMin) {
      // Set fixed range for stable display
      dbMin = newDbMin;
      dbMax = newDbMax;
      widebandOptimizationApplied = true;
    }
  }
  
  // Final display range (absolute dBFS values - labels will show these exact values)
  
  // Log diagnostic data now that we have display range (dbMin and dbMax are now defined)
  if (diagnosticData) {
    console.log(`[FFT UI] Range: ${diagnosticData.fftMin.toFixed(1)} to ${diagnosticData.fftMax.toFixed(1)} dBFS, mean: ${diagnosticData.fftMean.toFixed(1)} dBFS`);
    console.log(`[FFT UI] Noise Floor Comparison:`);
    console.log(`  Backend (from frame): ${diagnosticData.backendNoiseFloor !== null ? diagnosticData.backendNoiseFloor.toFixed(1) : 'N/A'} dBFS`);
    console.log(`  Frontend (2nd pct, raw): ${diagnosticData.frontendFloorRaw.toFixed(1)} dBFS`);
    console.log(`  Frontend (without edges): ${diagnosticData.floorWithoutEdges.toFixed(1)} dBFS`);
    console.log(`  Difference (with vs without edges): ${(diagnosticData.frontendFloorRaw - diagnosticData.floorWithoutEdges).toFixed(1)} dB`);
    console.log(`[FFT UI] Edge Bin Analysis:`);
    console.log(`  Edge bins (first ${diagnosticData.edgeBinCount}): mean=${diagnosticData.edgeStartMean.toFixed(1)} dBFS, max=${diagnosticData.edgeStartMax.toFixed(1)} dBFS`);
    console.log(`  Edge bins (last ${diagnosticData.edgeBinCount}): mean=${diagnosticData.edgeEndMean.toFixed(1)} dBFS, max=${diagnosticData.edgeEndMax.toFixed(1)} dBFS`);
    console.log(`  Center bins (${diagnosticData.centerBinsLength}): mean=${diagnosticData.centerMean.toFixed(1)} dBFS, max=${diagnosticData.centerMax.toFixed(1)} dBFS`);
    console.log(`  Edge elevation: start=${(diagnosticData.edgeStartMean-diagnosticData.centerMean).toFixed(1)} dB, end=${(diagnosticData.edgeEndMean-diagnosticData.centerMean).toFixed(1)} dB`);
    // dbMin and dbMax are now defined above
    const displayDbMin = typeof dbMin !== 'undefined' ? dbMin : diagnosticData.frontendFloorRaw;
    const displayDbMax = typeof dbMax !== 'undefined' ? dbMax : (displayDbMin + FFT_VIEW_RANGE_DB);
    const peakInRange = diagnosticData.fftMax >= displayDbMin && diagnosticData.fftMax <= displayDbMax;
    const peakPositionPct = peakInRange ? ((diagnosticData.fftMax - displayDbMin) / (displayDbMax - displayDbMin) * 100) : null;
    
    console.log(`[FFT UI] Display Range: ${displayDbMin.toFixed(1)} to ${displayDbMax.toFixed(1)} dBFS (span: ${(displayDbMax - displayDbMin).toFixed(1)} dB)`);
    console.log(`[FFT UI] Peak: ${diagnosticData.fftMax.toFixed(1)} dBFS, ` +
               `in range: ${peakInRange}, ` +
               `position: ${peakPositionPct !== null ? peakPositionPct.toFixed(1) + '%' : 'N/A'}, ` +
               `wideband optimization: ${typeof widebandOptimizationApplied !== 'undefined' ? widebandOptimizationApplied : 'N/A'}`);
    window._lastFftDiag = Date.now();
  }
  
  // Apply calibration offset (display-only: Q11 dBFS -> SDR++-style if configured)
  // Backend uses true Q11 scaling internally, offset is only for UI display
  const p = fftArr.map(v => {
    const x = Number(v) + globalCalibrationOffset;
    return Number.isFinite(x) ? x : dbMin;
  });

  // Clear ONLY the FFT area (never the waterfall) - device space
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, pxFftH);
  ctx.restore();

  // Header
  ctx.fillStyle = "#00ff88";
  ctx.font = "12px monospace";
  ctx.textAlign = "left";
  if (Number.isFinite(frame.center_freq_hz)) {
    ctx.fillText("Center: " + (frame.center_freq_hz / 1e6).toFixed(3) + " MHz", 8, 14);
    ctx.fillText("Units: " + powerUnits, 8, 28);
  }

  // Center marker
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.beginPath();
  ctx.moveTo(w / 2, 0);
  ctx.lineTo(w / 2, fftH);
  ctx.stroke();

  // Tasked overlay
  if (edgeMode === "tasked") {
    ctx.fillStyle = "rgba(0,255,136,0.04)";
    ctx.fillRect(0, 0, w, fftH);
    ctx.fillStyle = "#00ff88";
    ctx.font = "bold 12px monospace";
    ctx.fillText("TASKED CAPTURE", 10, 16);
    if (activeTask?.freq_hz) {
      ctx.font = "12px monospace";
      ctx.fillText(`FOI ${(activeTask.freq_hz / 1e6).toFixed(3)} MHz`, 10, 32);
    }
  }

  // Stable FFT trace: fast attack, slow decay (keeps bursts visible)
  // This is the "SDR UI feel" - spikes show up immediately, then fade slowly
  const dbNow = p.map(v => clamp(v, dbMin, dbMax));
  
  // Downsample for display if FFT size is very large (performance optimization)
  // Keep max 4096 points for rendering (canvas width is typically much smaller anyway)
  // For wideband signals, use mean instead of max to show signal energy better
  const MAX_DISPLAY_POINTS = 4096;
  const downsample = dbNow.length > MAX_DISPLAY_POINTS;
  const downsampleStep = downsample ? Math.ceil(dbNow.length / MAX_DISPLAY_POINTS) : 1;
  const displayLength = downsample ? Math.floor(dbNow.length / downsampleStep) : dbNow.length;
  
  // Downsample the current frame for display
  // For wideband signals: use mean to show signal energy (wideband spreads power across bins)
  // For narrowband signals: use max to preserve peaks (narrowband has sharp peaks)
  const dbNowDisplay = [];
  if (downsample) {
    for (let i = 0; i < displayLength; i++) {
      const startIdx = i * downsampleStep;
      const endIdx = Math.min(startIdx + downsampleStep, dbNow.length);
      
      if (isWideband) {
        // Wideband: use mean to show signal energy (power is spread across bins)
        let sum = 0;
        for (let j = startIdx; j < endIdx; j++) {
          sum += dbNow[j];
        }
        dbNowDisplay.push(sum / (endIdx - startIdx));
      } else {
        // Narrowband: use max to preserve peaks
        let maxVal = dbNow[startIdx];
        for (let j = startIdx + 1; j < endIdx; j++) {
          if (dbNow[j] > maxVal) maxVal = dbNow[j];
        }
        dbNowDisplay.push(maxVal);
      }
    }
    
    // Log downsampling info (throttled)
    if (!window._lastDownsampleLog || (Date.now() - window._lastDownsampleLog) > 2000) {
      console.log(`[FFT UI] Downsampling: ${dbNow.length} -> ${displayLength} points, ` +
                 `step=${downsampleStep}, method=${isWideband ? 'mean (wideband)' : 'max (narrowband)'}`);
      window._lastDownsampleLog = Date.now();
    }
  } else {
    dbNowDisplay.push(...dbNow);
  }
  
  // Handle different trace modes
  if (traceMode === "peak") {
    // Peak Hold mode: track maximum values with slow decay
    if (!peakHoldSpectrum || peakHoldSpectrum.length !== displayLength) {
      peakHoldSpectrum = dbNowDisplay.slice();
    } else {
      const PEAK_DECAY = 0.995;  // Very slow decay (0.5% per frame)
      for (let i = 0; i < displayLength; i++) {
        if (dbNowDisplay[i] > peakHoldSpectrum[i]) {
          peakHoldSpectrum[i] = dbNowDisplay[i];  // Fast attack
        } else {
          peakHoldSpectrum[i] = PEAK_DECAY * peakHoldSpectrum[i] + (1 - PEAK_DECAY) * dbNowDisplay[i];  // Slow decay
        }
      }
    }
    lastSpectrum = peakHoldSpectrum.slice();
  } else if (traceMode === "average") {
    // Average mode: exponential moving average
    if (!averageSpectrum || averageSpectrum.length !== displayLength) {
      averageSpectrum = dbNowDisplay.slice();
    } else {
      for (let i = 0; i < displayLength; i++) {
        averageSpectrum[i] = averageAlpha * dbNowDisplay[i] + (1 - averageAlpha) * averageSpectrum[i];
      }
    }
    lastSpectrum = averageSpectrum.slice();
  } else if (traceMode === "min") {
    // Min Hold mode: track minimum values
    if (!minHoldSpectrum || minHoldSpectrum.length !== displayLength) {
      minHoldSpectrum = dbNowDisplay.slice();
    } else {
      const MIN_RISE = 0.99;  // Slow rise when signal increases
      for (let i = 0; i < displayLength; i++) {
        if (dbNowDisplay[i] < minHoldSpectrum[i]) {
          minHoldSpectrum[i] = dbNowDisplay[i];  // Fast fall
        } else {
          minHoldSpectrum[i] = MIN_RISE * minHoldSpectrum[i] + (1 - MIN_RISE) * dbNowDisplay[i];  // Slow rise
        }
      }
    }
    lastSpectrum = minHoldSpectrum.slice();
  } else {
    // Instant mode (default): fast attack, slow decay
    const ATTACK = 0.55;  // Rise speed (0..1) - higher = faster attack (spikes appear quickly)
    const DECAY  = 0.96;  // Fall speed (0..1) - closer to 1 = slower decay (spikes fade slowly)
    
    if (!lastSpectrum || lastSpectrum.length !== displayLength) {
      lastSpectrum = dbNowDisplay.slice();
    } else {
      for (let i = 0; i < displayLength; i++) {
        const cur = lastSpectrum[i];
        const nxt = dbNowDisplay[i];
        if (nxt > cur) {
          // Fast attack: when signal rises, follow it quickly
          lastSpectrum[i] = ATTACK * nxt + (1 - ATTACK) * cur;
        } else {
          // Slow decay: when signal falls, fade slowly (keeps spikes visible)
          lastSpectrum[i] = DECAY * cur + (1 - DECAY) * nxt;
        }
      }
    }
  }

  // FFT trace
  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 2;
  ctx.shadowColor = "#00ff88";
  ctx.shadowBlur = 6;
  ctx.beginPath();
  for (let i = 0; i < lastSpectrum.length; i++) {
    const x = (i / (lastSpectrum.length - 1)) * w;
    const t = (lastSpectrum[i] - dbMin) / (dbMax - dbMin);
    const y = (1 - clamp(t, 0, 1)) * fftH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Axes
  drawPowerAxis(ctx, fftH, w, dbMin, dbMax, powerUnits);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath();
  ctx.moveTo(0, fftH + 0.5);
  ctx.lineTo(w, fftH + 0.5);
  ctx.stroke();
  drawFreqAxis(ctx, fftH, w, frame);
}

// ------------------------------
// WEBSOCKETS
// ------------------------------
function parseBinarySpectrumFrame(arrayBuffer) {
  console.log("[BIN] buffer len:", arrayBuffer.byteLength);

  const dv = new DataView(arrayBuffer);

  // Need at least 8 bytes to read header length
  if (dv.byteLength < 8) return null;

  // magic "SPRF"
  const m0 = dv.getUint8(0), m1 = dv.getUint8(1), m2 = dv.getUint8(2), m3 = dv.getUint8(3);
  const magic = String.fromCharCode(m0, m1, m2, m3);
  console.log("[BIN] magic:", magic);
  
  if (m0 !== 0x53 || m1 !== 0x50 || m2 !== 0x52 || m3 !== 0x46) return null; // "SPRF"

  const version = dv.getUint8(4);
  const flags = dv.getUint8(5);
  const headerLen = dv.getUint16(6, true);
  if (version !== 1 || headerLen < 32) return null;
  
  // Now check we have the full header
  if (dv.byteLength < headerLen) return null;

  const fftSize = dv.getUint32(8, true);
  // Correct center frequency parsing (BigInt-safe)
  const centerFreqHz = Number(dv.getBigInt64(12, true));
  const sampleRateSps = dv.getUint32(20, true);
  const ts = dv.getFloat32(24, true);
  const noiseFloor = dv.getFloat32(28, true);

  const hasInst = (flags & 0x01) !== 0;

  const off0 = headerLen;
  const bytes0 = fftSize * 4;
  const off1 = off0 + bytes0;
  const bytesNeed = hasInst ? (headerLen + bytes0 + bytes0) : (headerLen + bytes0);

  if (dv.byteLength < bytesNeed) return null;

  // Float32Array views into the buffer (zero-copy)
  const power0 = new Float32Array(arrayBuffer, off0, fftSize);
  const power1 = hasInst ? new Float32Array(arrayBuffer, off1, fftSize) : null;

  // drawSpectrum() expects normal JS arrays or array-likes; Float32Array works fine
  // Note: Binary frames don't include calibration metadata, so use global values
  return {
    ts,
    center_freq_hz: centerFreqHz,
    sample_rate_sps: sampleRateSps,
    fft_size: fftSize,
    power_dbfs: power0,
    power_inst_dbfs: power1,
    noise_floor_dbfs: noiseFloor,
    // Calibration metadata not in binary format - use global values set from hello message
    calibration_offset_db: globalCalibrationOffset,
    power_units: globalPowerUnits,
    // freqs_hz intentionally omitted (we'll compute axis from meta)
  };
}

function startFftWs() {
  // Avoid multiple connections or connecting while tab is hidden
  if (fftWs || (typeof document !== "undefined" && document.hidden)) {
    console.log("[FFT WS] WebSocket already exists, skipping");
    return;
  }
  console.log("[FFT WS] Creating new WebSocket connection to:", `ws://${location.host}/ws/live_fft`);
  fftWs = new WebSocket(`ws://${location.host}/ws/live_fft`);
  fftWs.binaryType = "arraybuffer";
  fftWs.onopen = () => {
    console.log("[FFT WS] WebSocket connected successfully!");
    setLed(wsLed, true);
    if (fftReconnectTimer) {
      clearTimeout(fftReconnectTimer);
      fftReconnectTimer = null;
    }
  };
  fftWs.onerror = (e) => {
    console.error("[FFT WS] WebSocket error:", e);
    setLed(wsLed, false);
  };
  fftWs.onmessage = (ev) => {
    try {
      if (!canvas || !ctx) {
        console.warn("[FFT WS] Canvas not available");
        return;
      }

      if (canvas.width < 50 || canvas.height < 50) resizeCanvas(true);

      // 1) Text messages (hello / fallback JSON)
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);

        // Handle hello message - store calibration metadata
        if (msg && msg.type === "hello") {
          if (msg.calibration_offset_db !== undefined) {
            globalCalibrationOffset = Number(msg.calibration_offset_db);
          }
          if (msg.power_units) {
            globalPowerUnits = msg.power_units;
          }
          console.log("[FFT WS] Calibration:", {
            offset: globalCalibrationOffset,
            units: globalPowerUnits
          });
          return;
        }

        // JSON fallback frame
        drawSpectrum(msg);
        return;
      }

      // 2) Binary frames
      const frame = parseBinarySpectrumFrame(ev.data);
      if (frame) {
        // Log received frame data for debugging (throttled to every 2 seconds)
        if (!window._lastFrameLog || (Date.now() - window._lastFrameLog) > 2000) {
          const fftSize = frame.power_dbfs ? frame.power_dbfs.length : (frame.power_inst_dbfs ? frame.power_inst_dbfs.length : 0);
          const hasInst = !!frame.power_inst_dbfs;
          const noiseFloor = frame.noise_floor_dbfs !== undefined ? frame.noise_floor_dbfs : null;
          
          if (fftSize > 0) {
            const arr = frame.power_inst_dbfs || frame.power_dbfs || [];
            const min = Math.min(...arr);
            const max = Math.max(...arr);
            const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
            
            console.log(`[FFT WS] Frame received: size=${fftSize}, has_inst=${hasInst}, ` +
                       `range=[${min.toFixed(1)}, ${max.toFixed(1)}] dBFS, mean=${mean.toFixed(1)} dBFS, ` +
                       `noise_floor=${noiseFloor !== null ? noiseFloor.toFixed(1) : 'N/A'} dBFS`);
          }
          window._lastFrameLog = Date.now();
        }
        drawSpectrum(frame);
      }
    } catch (e) {
      console.warn("[FFT WS] onmessage parse failed:", e);
    }
  };
  fftWs.onclose = () => {
    setLed(wsLed, false);
    fftWs = null;
    // Auto-reconnect if tab is visible
    if (typeof document !== "undefined" && !document.hidden) {
      if (fftReconnectTimer) {
        clearTimeout(fftReconnectTimer);
      }
      fftReconnectTimer = setTimeout(() => {
        fftReconnectTimer = null;
        startFftWs();
      }, 3000);
    }
  };
}

function stopFftWs() {
  if (fftWs) {
    fftWs.close();
    fftWs = null;
  }
  setLed(wsLed, false);
}

function startNotifyWs() {
  const url = `${window.location.protocol.replace("http", "ws")}//${window.location.host}/ws/notify`;
  console.log("[NOTIFY WS] Trying to connect to:", url);

  notifyWs = new WebSocket(url);

  notifyWs.onopen = () => {
    console.log("[NOTIFY WS] WebSocket connected to /ws/notify");
    console.log("[NOTIFY WS] SUCCESS: Connected! Ready for nodes, cues, and mode updates.");
    setLed("wsLed", true);
  };

  notifyWs.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      console.log("[NOTIFY WS] Received message type:", msg.type, "payload:", msg.payload);
      console.log("[NOTIFY WS] Received message:", msg);

      if (msg.type === "tripwire_nodes") {
        console.log("[NOTIFY WS] Updating node cards -", msg.payload.nodes.length, "nodes");
        renderTripwireNodes(msg.payload.nodes || []);

      } else if (msg.type === "edge_mode") {
        console.log("[NOTIFY WS] Edge mode changed to:", msg.payload.mode);
        setEdgeMode(msg.payload.mode);

      } else if (msg.type === "tripwire_cue") {
        console.log("[CUE] New RF cue received - adding to bottom panel:", msg.payload);
        addCue(msg.payload);      // ← correct function
        renderCues();             // ← refresh the cue list

      } else if (msg.type === "tripwire_auto_reject") {
        console.log("[CUE] Cue auto-rejected:", msg.payload.reason);
        // Optional: show rejected cues in the list (with different style)
        // For now, just add it so operator sees it was rejected
        const rejected = msg.payload.cue;
        rejected.rejected = true;
        rejected.reject_reason = msg.payload.reason;
        addCue(rejected);
        renderCues();

      } else if (msg.type === "capture_start") {
        console.log("[CAPTURE] *** CAPTURE STARTED ***", msg.payload);
        if (captureBanner) {
          // Force show the banner
          captureBanner.classList.remove("hidden");
          captureBanner.classList.add("active");
          captureBanner.style.display = "block";  // Direct style override
          console.log("[CAPTURE] Banner shown - classes:", captureBanner.className);
        }
        if (captureProgressBar) {
          captureProgressBar.style.width = "0%";
        }
        // Also update mode to tasked
        setEdgeMode("tasked");
        // Reduce UI polling during capture to reduce scheduler contention
        window._capture_in_progress = true;
        // Clear any existing timeout
        if (window._capture_timeout) {
          clearTimeout(window._capture_timeout);
          window._capture_timeout = null;
        }

      } else if (msg.type === "capture_progress") {
        const progress = msg.payload.progress_pct || 0;
        console.log("[CAPTURE] Progress:", progress.toFixed(1) + "%");
        if (captureProgressBar) {
          captureProgressBar.style.width = progress + "%";
        }
        // If progress reaches 100%, set a fallback timeout to hide banner
        if (progress >= 100.0) {
          if (window._capture_timeout) {
            clearTimeout(window._capture_timeout);
          }
          window._capture_timeout = setTimeout(() => {
            console.log("[CAPTURE] Fallback: Hiding banner after 100% progress");
            if (captureBanner) {
              captureBanner.classList.remove("active");
              captureBanner.classList.add("hidden");
            }
            window._capture_in_progress = false;
            window._capture_timeout = null;
          }, 2000); // 2 second fallback
        }

      } else if (msg.type === "capture_complete") {
        console.log("[CAPTURE] *** CAPTURE COMPLETE ***", msg.payload);
        if (captureProgressBar) {
          captureProgressBar.style.width = "100%";
        }
        // Resume normal UI polling after capture
        window._capture_in_progress = false;
        // Clear any fallback timeout
        if (window._capture_timeout) {
          clearTimeout(window._capture_timeout);
          window._capture_timeout = null;
        }
        // Hide banner immediately
        if (captureBanner) {
          console.log("[CAPTURE] Hiding banner (capture_complete event received)");
          captureBanner.classList.remove("active");
          captureBanner.classList.add("hidden");
          captureBanner.style.display = "none";
          // Remove inline style after a moment to let CSS take over
          setTimeout(() => {
            if (captureBanner) {
              captureBanner.style.display = "";
            }
          }, 100);
        }
        // Refresh mode (should return to armed/manual)
        refreshEdgeMode();
      }

      if (msg.type === "classification_result") {
        console.log("[NOTIFY WS] Received classification_result:", msg.payload);
        addClassification(msg.payload);
      }

    } catch (err) {
      console.warn("[NOTIFY WS] Failed to parse message:", err, e.data);
    }
  };

  notifyWs.onerror = (e) => {
    console.error("[NOTIFY WS] WebSocket error (check network/firewall/port 8000):", e);
  };

  notifyWs.onclose = (e) => {
    console.log("[NOTIFY WS] Connection closed (code:", e.code, "). Reconnecting in 3 seconds...");
    setLed("wsLed", false);
    setTimeout(startNotifyWs, 3000);  // auto-reconnect
  };
}

function stopNotifyWs() {
  if (notifyWs) {
    notifyWs.close();
    notifyWs = null;
  }
  setLed("wsLed", false);
}

async function startLive() {
  console.log("[Live] startLive() called");
  resizeCanvas(true);
  const cfg = readSdrForm();
  console.log("[Live] SDR config:", cfg);
  
  try {
    // CRITICAL: Apply SDR config FIRST (including gain, LNA, BT200)
    // This ensures gain settings are applied before starting the scan
    console.log("[Live] Applying SDR config (gain, LNA, BT200)...");
    await applySdrConfig();
    
    // Then start the live scan with FFT parameters
    console.log("[Live] Calling API.liveStart with payload:", {
      center_freq_hz: cfg.center_freq_hz,
      sample_rate_sps: cfg.sample_rate_sps,
      fft_size: cfg.fft_size,
      fps: cfg.fps,
    });
    const result = await API.liveStart({
      center_freq_hz: cfg.center_freq_hz,
      sample_rate_sps: cfg.sample_rate_sps,
      fft_size: cfg.fft_size,
      fps: cfg.fps,
    });
    console.log("[Live] API.liveStart succeeded:", result);
  } catch (e) {
    console.error("[Live] backend start failed:", e);
    alert(`Failed to start live scan: ${e.message || e}`);
    return; // Don't start WebSocket if backend failed
  }
  
  console.log("[Live] Starting WebSocket connections...");
  stopFftWs();
  startFftWs();
  startNotifyWs();
  refreshStatus();
  console.log("[Live] startLive() completed");
}

async function stopLive() {
  console.log("[Live] stopLive() called");
  
  // Stop backend scan first
  try {
    const response = await fetch("/live/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (response.ok) {
      const result = await response.json();
      console.log("[Live] Backend scan stopped:", result);
    } else {
      console.warn("[Live] Failed to stop backend scan:", response.status);
    }
  } catch (e) {
    console.error("[Live] Error stopping backend scan:", e);
  }
  
  // Then stop frontend WebSocket connections
  stopFftWs();
  stopNotifyWs();
  refreshStatus();
  console.log("[Live] stopLive() completed");
}

if (startBtn) {
  console.log("[Init] Start button found, wiring up click handler");
  startBtn.onclick = startLive;
} else {
  console.error("[Init] Start button NOT FOUND! ID='start'");
}
if (stopBtn) {
  console.log("[Init] Stop button found, wiring up click handler");
  stopBtn.onclick = stopLive;
} else {
  console.error("[Init] Stop button NOT FOUND! ID='stop'");
}

// ------------------------------
// STATUS & POLLING
// ------------------------------
function updateSdrStatus(info) {
  if (!info) return;
  const device = info?.device || {};
  const cfg = info?.current_config || {};
  if (sdrDriverEl) sdrDriverEl.textContent = device.driver || info.driver || "Unknown SDR";
  // RX port: prefer device.active_rx_channel (actual hardware state), fallback to cfg.rx_channel
  const rxChannel = device.active_rx_channel !== undefined ? device.active_rx_channel : (cfg.rx_channel !== undefined ? cfg.rx_channel : null);
  if (sdrRxPortEl) sdrRxPortEl.textContent = rxChannel !== null ? `RX${rxChannel}` : "—";
  if (sdrCenterEl) sdrCenterEl.textContent = cfg.center_freq_hz ? (cfg.center_freq_hz / 1e6).toFixed(3) + " MHz" : "—";
  if (sdrRateEl) sdrRateEl.textContent = cfg.sample_rate_sps ? (cfg.sample_rate_sps / 1e6).toFixed(2) + " MS/s" : "—";
  if (sdrGainEl) sdrGainEl.textContent = (cfg.gain_db !== undefined) ? (cfg.gain_db + " dB") : "—";
  if (sdrGainModeEl) sdrGainModeEl.textContent = cfg.gain_mode || "manual";
  
  // Update gain input from current_config, but not if user just moved the slider (avoids snap-back to stale value)
  if (cfg.gain_db !== undefined && gainSlider && (Date.now() - lastGainSliderInputAt) > GAIN_SLIDER_GRACE_MS) {
    gainSlider.value = cfg.gain_db;
    if (gainValue) {
      gainValue.textContent = cfg.gain_db;
    }
  }
  
  // Update LNA and BT200 controls from current_config (what we want) not device (what hardware has)
  // This prevents the UI from reverting user input before it's applied
  // LNA gain is now automatically optimized by main gain - no manual control
  // BT200: Only enable if user explicitly sets it to true
  // Default is always "false" (disabled) - hardware not connected
  if (cfg.bt200_enabled === true && bt200Select) {
    // Only set to true if explicitly enabled
    bt200Select.value = "true";
  } else if (bt200Select) {
    // Default to false (disabled) - BT200 not connected
    bt200Select.value = "false";
  }
}

async function refreshStatus() {
  // Rate-limit polling during capture to reduce scheduler contention
  if (window._capture_in_progress) {
    return; // Skip status polling during capture
  }
  try {
    const j = await API.status();
    if (statusEl) statusEl.textContent = safeJson(j);
    setEdgeMode(j?.mode || "manual");
    j?.task ? setActiveTask(j.task) : clearActiveTask();
    setLed(sdrLed, !!j?.sdr_open);
    setLed(gpsLed, !!j?.gps?.fix && j.gps.fix !== "NO FIX");
    setLed(takLed, !!j?.tak_connected);
    
    // Reconnect FFT WebSocket when scan is running but we have no connection (e.g. returned from ML page)
    if (j?.scan_running && (!fftWs || fftWs.readyState !== WebSocket.OPEN)) {
      startFftWs();
    }
    
    // Fallback: If banner is visible but we're not in capture mode, hide it
    if (captureBanner && captureBanner.classList.contains("active")) {
      const mode = j?.mode || "manual";
      // If mode is not "tasked" (which is the capture mode), hide banner
      if (mode !== "tasked" && !window._capture_in_progress) {
        console.log("[CAPTURE] Fallback: Hiding banner (mode is not 'tasked')");
        captureBanner.classList.remove("active");
        captureBanner.classList.add("hidden");
      }
    }
  } catch (_) {
    if (statusEl) statusEl.textContent = "Status unavailable";
  }
}

async function pollSdrInfo() {
  // Rate-limit polling during capture to reduce scheduler contention
  if (window._capture_in_progress) {
    return; // Skip SDR info polling during capture
  }
  try {
    const info = await API.sdrInfo();
    updateSdrStatus(info);
  } catch (_) {}
}

async function pollSdrHealth() {
  // Don't poll during capture to avoid contention
  if (window._capture_in_progress) {
    return;
  }
  
  try {
    const health = await API.sdrHealth();
    if (health && typeof health === "object") {
      updateSdrHealth(health);
    }
  } catch (e) {
    // Silently ignore health polling errors - don't break the UI
    // console.warn("[SDR Health] Poll error:", e);
  }
}

function updateSdrHealth(health) {
  if (!health || typeof health !== "object") {
    return; // Silently ignore invalid health data
  }

  try {
    const statusBtn = document.getElementById("sdrHealthStatus");
    const successRateEl = document.getElementById("healthSuccessRate");
    const throughputEl = document.getElementById("healthThroughput");
    const samplesSecEl = document.getElementById("healthSamplesSec");
    const avgReadTimeEl = document.getElementById("healthAvgReadTime");
    const errorsEl = document.getElementById("healthErrors");
    const readsEl = document.getElementById("healthReads");
    const streamEl = document.getElementById("healthStream");
    const usbSpeedEl = document.getElementById("healthUsbSpeed");

    if (!statusBtn) return;

    // Update status button
    const status = health.status || "unknown";
    statusBtn.className = `health-status-btn ${status}`;
    statusBtn.textContent = status.toUpperCase();

    // Update metrics with safe number handling
    if (successRateEl) {
      const val = health.success_rate_pct;
      successRateEl.textContent = (typeof val === "number" && !isNaN(val))
        ? `${val.toFixed(1)}%` : "—";
    }
    if (throughputEl) {
      const val = health.throughput_mbps;
      throughputEl.textContent = (typeof val === "number" && !isNaN(val))
        ? `${val.toFixed(2)} MBps` : "—";
    }
    if (samplesSecEl) {
      const val = health.samples_per_sec;
      samplesSecEl.textContent = (typeof val === "number" && !isNaN(val))
        ? `${val.toFixed(2)}M` : "—";
    }
    if (avgReadTimeEl) {
      const val = health.avg_read_time_ms;
      avgReadTimeEl.textContent = (typeof val === "number" && !isNaN(val))
        ? `${val.toFixed(2)} ms` : "—";
    }
    if (errorsEl) {
      const errors = (typeof health.errors === "number") ? health.errors : 0;
      errorsEl.textContent = `${errors}`;
    }
    if (readsEl) {
      const reads = health.reads || {};
      const total = (typeof reads.total === "number") ? reads.total : 0;
      const successful = (typeof reads.successful === "number") ? reads.successful : 0;
      readsEl.textContent = `${successful.toLocaleString()} / ${total.toLocaleString()}`;
    }
    if (streamEl) {
      streamEl.textContent = health.stream || "—";
    }
    if (usbSpeedEl) {
      usbSpeedEl.textContent = health.usb_speed || "Unknown";
    }
  } catch (e) {
    console.warn("[SDR Health] Error updating health panel:", e);
    // Don't throw - just log and continue
  }
}

// ------------------------------
// TRIPWIRE NODES (fixed 3 cards)
// ------------------------------

const MAX_TRIPWIRES = 3;

/**
 * Initialize 3 fixed Tripwire cards (always visible).
 * Call ONCE on page load.
 */
function initTripwireNodes() {
  if (!tripwireNodesEl) return;

  tripwireNodesEl.innerHTML = "";

  for (let i = 0; i < MAX_TRIPWIRES; i++) {
    const card = document.createElement("div");
    card.className = "tw-card idle";
    card.dataset.slot = i;

    card.innerHTML = `
      <div class="tw-header">
        <div class="tw-led"></div>
        <div class="tw-title">Tripwire ${i + 1}</div>
      </div>

      <div class="tw-line muted">Status: Waiting</div>
      <div class="tw-line">Callsign: —</div>
      <div class="tw-line">IP: —</div>
      <div class="tw-line">SDR: —</div>
      <div class="tw-line">GPS: —</div>
      <div class="tw-line">Node Activity: —</div>
      
      <div class="tw-scan-plan-controls" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--bg-panel-border);">
        <label style="display: block; margin-bottom: 6px; font-size: 11px; color: var(--text-muted);">
          Scan Plan:
          <select class="tw-scan-plan-select" data-slot="${i}" style="width: 100%; margin-top: 4px; padding: 4px; background: var(--bg-main); border: 1px solid var(--bg-panel-border); border-radius: 4px; color: var(--text-main); font-size: 11px;">
            <option value="">-- Select Plan --</option>
            <option value="survey_wide">RF Survey (Wide)</option>
            <option value="fhss_subghz">FHSS Control (Sub-GHz)</option>
            <option value="fhss_subghz_ibw">FHSS Control IBW (Sub-GHz)</option>
            <option value="fhss_eu_868_ibw">FHSS Control IBW (EU 868)</option>
            <option value="fhss_eu_433_ibw">FHSS Control IBW (EU 433)</option>
            <option value="fhss_24g">FHSS Control (2.4 GHz)</option>
            <option value="digital_vtx_5g8">Digital VTX (5.8 GHz)</option>
            <option value="analog_vtx_5g8">Analog FPV VTX (5.8 GHz)</option>
            <option value="voice_narrowband">Voice / PTT (Narrowband)</option>
            <option value="wifi_bt_24g">Wi-Fi / Bluetooth (2.4 GHz)</option>
            <option value="manual">Manual (Start/Stop MHz)</option>
          </select>
        </label>
        <button class="tw-send-plan-btn" data-slot="${i}" style="width: 100%; padding: 6px; background: rgba(0, 255, 136, 0.15); color: var(--accent-green); border: 1px solid rgba(0, 255, 136, 0.4); border-radius: 4px; font-size: 11px; font-weight: bold; cursor: pointer; transition: all 0.15s ease;">
          Send Scan Plan
        </button>
      </div>
    `;

    tripwireNodesEl.appendChild(card);
  }
}

function renderTripwireNodes(nodes) {
  if (!tripwireNodesEl) return;
  
  // Store nodes for scan plan button handlers
  window._lastTripwireNodes = nodes || [];
  
  const cards = tripwireNodesEl.querySelectorAll(".tw-card");
  const now = Date.now() / 1000;
  let connectedCount = 0;
  
  // Reset all cards
  cards.forEach(card => {
    card.classList.remove("connected");
    card.classList.add("idle");
    card.querySelector(".tw-led").classList.remove("on");
    const lines = card.querySelectorAll(".tw-line");
    if (lines[0]) lines[0].textContent = "Status: Waiting";
    if (lines[5]) lines[5].textContent = "Node Activity: —";
  });
  
  if (!Array.isArray(nodes) || nodes.length === 0) {
    updateTripwireLeds(0);
    return;
  }
  
  nodes.slice(0, MAX_TRIPWIRES).forEach((node, idx) => {
    const card = cards[idx];
    if (!card) return;
    const lastSeen = Number(node.last_seen || 0);
    const connected = (now - lastSeen) < CONNECTED_SECS;
    
    if (connected) connectedCount++;
    
    // Visual state
    card.classList.toggle("connected", connected);
    card.classList.remove("idle");
    const led = card.querySelector(".tw-led");
    if (connected) led.classList.add("on");
    const lines = card.querySelectorAll(".tw-line");
    // Status
    lines[0].textContent = `Status: ${connected ? "Online" : "Stale"}`;
    // Identity / metadata
    lines[1].innerHTML = `<strong>${node.callsign || node.node_id || "Unknown"}</strong>`;
    lines[2].textContent = `IP: ${node.ip || "—"}`;
    lines[3].textContent = `SDR: ${node.sdr || node.meta?.sdr_driver || "—"}`;
    lines[4].textContent = `GPS: ${node.gps && node.gps.lat ? "Fix" : "No fix"}`;
    // Transport-level activity only
    lines[5].textContent = connected
      ? "Node Activity: WS connected"
      : "Node Activity: —";
    
    // Update scan plan dropdown with current value
    const scanPlanSelect = card.querySelector(".tw-scan-plan-select");
    if (scanPlanSelect) {
      const currentPlan = node.scan_plan || node.meta?.scan_plan || "";
      scanPlanSelect.value = currentPlan;
    }
    
    // Wire up button handler
    const btn = card.querySelector(".tw-send-plan-btn");
    if (btn) {
      btn.onclick = () => {
        const scanPlan = scanPlanSelect?.value;
        if (scanPlan && node.node_id) {
          sendScanPlanToTripwire(node.node_id, scanPlan);
        }
      };
    }
  });
  
  updateTripwireLeds(connectedCount);
}

// ------------------------------
// AoA FUSION VISUALIZATION
// ------------------------------

// Node colors for visualization
const NODE_COLORS = [
  "#3cff9e", // green
  "#00b3ff", // blue
  "#ff8c00", // orange
];

/**
 * Calculate intersection point from two bearing lines (triangulation)
 * Returns {lat, lon} or null if calculation fails
 */
function calculateIntersection(node1, bearing1, node2, bearing2) {
  if (!node1.gps || !node2.gps || !node1.gps.lat || !node2.gps.lat) {
    return null;
  }
  
  const lat1 = node1.gps.lat * Math.PI / 180;
  const lon1 = node1.gps.lon * Math.PI / 180;
  const lat2 = node2.gps.lat * Math.PI / 180;
  const lon2 = node2.gps.lon * Math.PI / 180;
  
  const brg1 = bearing1 * Math.PI / 180;
  const brg2 = bearing2 * Math.PI / 180;
  
  // Calculate intersection using spherical trigonometry
  // Using Vincenty's formula for bearing intersection
  const dLat = lat2 - lat1;
  const dLon = lon2 - lon1;
  
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1) * Math.cos(lat2) *
            Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  const distance = 6371000 * c; // Earth radius in meters
  
  // Simplified intersection calculation
  // For small distances, use plane geometry approximation
  if (distance < 10000) { // Less than 10km
    // Convert to local coordinates (meters)
    const R = 6371000; // Earth radius in meters
    const x1 = R * Math.cos(lat1) * Math.cos(lon1);
    const y1 = R * Math.cos(lat1) * Math.sin(lon1);
    const x2 = R * Math.cos(lat2) * Math.cos(lon2);
    const y2 = R * Math.cos(lat2) * Math.sin(lon2);
    
    // Direction vectors
    const dx1 = Math.sin(brg1);
    const dy1 = Math.cos(brg1);
    const dx2 = Math.sin(brg2);
    const dy2 = Math.cos(brg2);
    
    // Find intersection of two lines
    // Line 1: (x1, y1) + t * (dx1, dy1)
    // Line 2: (x2, y2) + s * (dx2, dy2)
    const denom = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(denom) < 1e-6) {
      return null; // Lines are parallel
    }
    
    const t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom;
    const x = x1 + t * dx1;
    const y = y1 + t * dy1;
    
    // Convert back to lat/lon
    const lat = Math.asin(y / R) * 180 / Math.PI;
    const lon = Math.atan2(y, x) * 180 / Math.PI;
    
    return { lat, lon };
  }
  
  return null;
}

/**
 * Calculate fusion quality metric (0-1)
 * Based on intersection angles, cone widths, and confidences
 */
function calculateFusionQuality(cones) {
  if (cones.length < 2) return 0;
  
  let quality = 1.0;
  
  // Penalize wide cones
  cones.forEach(cone => {
    const width = cone.cone_width_deg || 45;
    if (width > 45) quality *= 0.8;
    if (width > 90) quality *= 0.5;
  });
  
  // Penalize low confidence
  cones.forEach(cone => {
    const conf = cone.confidence || 0.5;
    quality *= conf;
  });
  
  // Reward good intersection angles (60-120 degrees is ideal)
  if (cones.length >= 2) {
    // Simplified: assume nodes are reasonably spaced
    quality *= 0.9; // Good baseline
  }
  
  return Math.max(0, Math.min(1, quality));
}

/**
 * Convert GPS coordinates to canvas pixels
 * Note: canvas context is already scaled by DPR, so use display dimensions
 * Accounts for zoom and pan transformations
 */
function gpsToCanvas(lat, lon, bounds, canvasWidth, canvasHeight) {
  const latRange = bounds.maxLat - bounds.minLat;
  const lonRange = bounds.maxLon - bounds.minLon;
  
  // Calculate base coordinates (0-1 normalized)
  const normalizedX = (lon - bounds.minLon) / lonRange;
  const normalizedY = (bounds.maxLat - lat) / latRange;
  
  // Apply zoom and pan
  const centerX = canvasWidth / 2;
  const centerY = canvasHeight / 2;
  
  // Transform: scale around center, then translate
  const x = (normalizedX * canvasWidth - centerX) * aoaFusionZoom + centerX + aoaFusionPanX;
  const y = (normalizedY * canvasHeight - centerY) * aoaFusionZoom + centerY + aoaFusionPanY;
  
  return { x, y };
}

/**
 * Convert canvas pixel coordinates to GPS coordinates (for click-to-zoom/pan)
 */
function canvasToGps(canvasX, canvasY, bounds, canvasWidth, canvasHeight) {
  const centerX = canvasWidth / 2;
  const centerY = canvasHeight / 2;
  
  // Reverse transform: subtract pan, then unscale
  const normalizedX = ((canvasX - centerX - aoaFusionPanX) / aoaFusionZoom + centerX) / canvasWidth;
  const normalizedY = ((canvasY - centerY - aoaFusionPanY) / aoaFusionZoom + centerY) / canvasHeight;
  
  const latRange = bounds.maxLat - bounds.minLat;
  const lonRange = bounds.maxLon - bounds.minLon;
  
  const lon = bounds.minLon + normalizedX * lonRange;
  const lat = bounds.maxLat - normalizedY * latRange;
  
  return { lat, lon };
}

// Track if canvas has been initialized to avoid repeated scaling
let aoaFusionCanvasInitialized = false;

// Zoom and pan state for AoA fusion canvas
let aoaFusionZoom = 1.0;
let aoaFusionPanX = 0;
let aoaFusionPanY = 0;
let aoaFusionIsPanning = false;
let aoaFusionPanStartX = 0;
let aoaFusionPanStartY = 0;
let aoaFusionBounds = null; // Store bounds for coordinate transformation

/**
 * Resize AoA fusion canvas to match display size
 * Only resizes when actually needed (window resize or first time)
 */
function resizeAoAFusionCanvas() {
  if (!aoaFusionCanvas || !aoaFusionCtx) return;
  
  const dpr = window.devicePixelRatio || 1;
  const rect = aoaFusionCanvas.getBoundingClientRect();
  const displayWidth = rect.width;
  const displayHeight = rect.height;
  
  // Skip if dimensions are invalid
  if (displayWidth <= 0 || displayHeight <= 0) return;
  
  // Only resize if dimensions actually changed (with small threshold)
  if (aoaFusionCanvas.width > 0 && aoaFusionCanvas.height > 0) {
    const currentWidth = aoaFusionCanvas.width / dpr;
    const currentHeight = aoaFusionCanvas.height / dpr;
    
    if (Math.abs(currentWidth - displayWidth) < 2 && Math.abs(currentHeight - displayHeight) < 2 && aoaFusionCanvasInitialized) {
      return; // No resize needed
    }
  }
  
  // Set display size (CSS controls this)
  aoaFusionCanvas.style.width = displayWidth + "px";
  aoaFusionCanvas.style.height = displayHeight + "px";
  
  // Set actual canvas size (accounting for DPR)
  const newWidth = Math.floor(displayWidth * dpr);
  const newHeight = Math.floor(displayHeight * dpr);
  
  // Only resize if dimensions actually changed
  if (aoaFusionCanvas.width !== newWidth || aoaFusionCanvas.height !== newHeight) {
    aoaFusionCanvas.width = newWidth;
    aoaFusionCanvas.height = newHeight;
    
    // Reset transform and re-scale
    aoaFusionCtx.setTransform(1, 0, 0, 1, 0, 0);
    aoaFusionCtx.scale(dpr, dpr);
    
    aoaFusionCanvasInitialized = true;
  }
}

/**
 * Draw AoA fusion visualization on canvas
 */
function drawAoAFusion(cones) {
  if (!aoaFusionCanvas || !aoaFusionCtx) return;
  
  // Resize canvas if needed (only when dimensions change)
  resizeAoAFusionCanvas();
  
  const canvas = aoaFusionCanvas;
  const ctx = aoaFusionCtx;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  
  // Clear canvas (context is already scaled by DPR)
  ctx.fillStyle = "#070a0f";
  ctx.fillRect(0, 0, width, height);
  
  if (!cones || cones.length === 0) {
    ctx.fillStyle = "#7f8fa6";
    ctx.font = "14px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("Waiting for AoA cones...", width / 2, height / 2);
    return;
  }
  
  // Filter cones with GPS data
  const validCones = cones.filter(c => c.gps && c.gps.lat && c.gps.lon && c.bearing_deg != null);
  
  if (validCones.length === 0) {
    ctx.fillStyle = "#ff8c00";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("No GPS data available", width / 2, height / 2);
    return;
  }
  
  // Calculate bounds (with padding)
  let minLat = Infinity, maxLat = -Infinity;
  let minLon = Infinity, maxLon = -Infinity;
  
  validCones.forEach(cone => {
    const lat = cone.gps.lat;
    const lon = cone.gps.lon;
    minLat = Math.min(minLat, lat);
    maxLat = Math.max(maxLat, lat);
    minLon = Math.min(minLon, lon);
    maxLon = Math.max(maxLon, lon);
  });
  
  // Add padding (10% on each side)
  const latPadding = (maxLat - minLat) * 0.1 || 0.001;
  const lonPadding = (maxLon - minLon) * 0.1 || 0.001;
  minLat -= latPadding;
  maxLat += latPadding;
  minLon -= lonPadding;
  maxLon += lonPadding;
  
  const bounds = { minLat, maxLat, minLon, maxLon };
  aoaFusionBounds = bounds; // Store for coordinate transformations
  
  // Draw nodes and bearing lines (coordinates are already transformed in gpsToCanvas)
  validCones.forEach((cone, idx) => {
    const nodePos = gpsToCanvas(cone.gps.lat, cone.gps.lon, bounds, width, height);
    const nodeX = nodePos.x;
    const nodeY = nodePos.y;
    const bearing = cone.bearing_deg;
    const coneWidth = cone.cone_width_deg || 45;
    const color = NODE_COLORS[idx % NODE_COLORS.length];
    
    // Draw node point
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(nodeX, nodeY, 6, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw node label
    ctx.fillStyle = color;
    ctx.font = "10px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(cone.callsign || cone.node_id, nodeX + 8, nodeY - 4);
    
    // Draw bearing line (extend to edge of canvas)
    const bearingRad = (bearing - 90) * Math.PI / 180; // Convert to screen coordinates
    const lineLength = Math.max(width, height) * 1.5;
    const endX = nodeX + Math.cos(bearingRad) * lineLength;
    const endY = nodeY + Math.sin(bearingRad) * lineLength;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(nodeX, nodeY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    
    // Draw cone sector (semi-transparent) - draw first so TAI appears on top
    const halfCone = (coneWidth / 2) * Math.PI / 180;
    const brg1 = bearingRad - halfCone;
    const brg2 = bearingRad + halfCone;
    
    ctx.fillStyle = color + "20"; // 20 = ~12% opacity (reduced so TAI is more visible)
    ctx.beginPath();
    ctx.moveTo(nodeX, nodeY);
    ctx.lineTo(nodeX + Math.cos(brg1) * lineLength, nodeY + Math.sin(brg1) * lineLength);
    ctx.arc(nodeX, nodeY, lineLength, brg1, brg2);
    ctx.closePath();
    ctx.fill();
  });
  
  // Calculate and draw TAI (Targeted Area of Interest) FIRST, then intersection point
  if (validCones.length >= 2) {
    const intersections = [];
    
    // Calculate intersections between all pairs
    for (let i = 0; i < validCones.length; i++) {
      for (let j = i + 1; j < validCones.length; j++) {
        const cone1 = validCones[i];
        const cone2 = validCones[j];
        const node1 = { gps: cone1.gps };
        const node2 = { gps: cone2.gps };
        
        const intersection = calculateIntersection(
          node1, cone1.bearing_deg,
          node2, cone2.bearing_deg
        );
        
        if (intersection) {
          intersections.push(intersection);
        }
      }
    }
    
    // Draw intersection points and TAI (Targeted Area of Interest)
    if (intersections.length > 0) {
      // Average all intersections for final point
      const avgLat = intersections.reduce((sum, p) => sum + p.lat, 0) / intersections.length;
      const avgLon = intersections.reduce((sum, p) => sum + p.lon, 0) / intersections.length;
      
      const fusionPos = gpsToCanvas(avgLat, avgLon, bounds, width, height);
      const fusionX = fusionPos.x;
      const fusionY = fusionPos.y;
      
      // Calculate TAI size based on confidence and cone widths
      // Lower confidence = larger uncertainty area
      // Wider cones = larger uncertainty area
      const avgConfidence = validCones.reduce((sum, c) => sum + (c.confidence || 0.5), 0) / validCones.length;
      const avgConeWidth = validCones.reduce((sum, c) => sum + (c.cone_width_deg || 45), 0) / validCones.length;
      
      // Base radius on confidence (inverse relationship)
      // Confidence 1.0 = small radius (20px), Confidence 0.0 = large radius (80px)
      const baseRadius = 20 + (1.0 - avgConfidence) * 60;
      
      // Adjust for cone width (wider = larger)
      const widthMultiplier = 1.0 + (avgConeWidth / 90.0); // 45deg = 1.5x, 90deg = 2x
      const taiRadius = baseRadius * widthMultiplier;
      
      // Clamp radius to reasonable bounds (ensure minimum visibility)
      const minRadius = 25; // Increased minimum for better visibility
      const maxRadius = Math.min(width, height) * 0.4; // Max 40% of canvas
      const finalRadius = Math.max(minRadius, Math.min(maxRadius, taiRadius));
      
      // Draw TAI (Targeted Area of Interest) - yellow transparent area
      // Draw filled circle first
      ctx.fillStyle = "rgba(255, 255, 0, 0.4)"; // Yellow, 40% opacity (more visible)
      ctx.beginPath();
      ctx.arc(fusionX, fusionY, finalRadius, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw border
      ctx.strokeStyle = "rgba(255, 255, 0, 0.8)"; // Yellow border, 80% opacity
      ctx.lineWidth = 3; // Thicker border
      ctx.beginPath();
      ctx.arc(fusionX, fusionY, finalRadius, 0, Math.PI * 2);
      ctx.stroke();
      
      // Draw TAI label with background for visibility
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(fusionX - 20, fusionY - finalRadius - 25, 40, 18);
      ctx.fillStyle = "rgba(255, 255, 0, 1.0)"; // Yellow text, fully opaque
      ctx.font = "bold 12px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("TAI", fusionX, fusionY - finalRadius - 16);
      
      // Draw intersection point (center of TAI) - draw on top
      ctx.fillStyle = "#3cff9e";
      ctx.beginPath();
      ctx.arc(fusionX, fusionY, 5, 0, Math.PI * 2); // Slightly larger
      ctx.fill();
      
      // Draw crosshair
      ctx.strokeStyle = "#3cff9e";
      ctx.lineWidth = 2; // Thicker crosshair
      ctx.beginPath();
      ctx.moveTo(fusionX - 10, fusionY);
      ctx.lineTo(fusionX + 10, fusionY);
      ctx.moveTo(fusionX, fusionY - 10);
      ctx.lineTo(fusionX, fusionY + 10);
      ctx.stroke();
      
      // Store TAI data for ATAK integration
      window._lastTAI = {
        lat: avgLat,
        lon: avgLon,
        radius_m: finalRadius * (bounds.maxLat - bounds.minLat) * 111000 / height, // Approximate meters
        confidence: avgConfidence,
        quality: calculateFusionQuality(validCones),
        timestamp: Date.now() / 1000
      };
    }
  }
  
  // Draw zoom/pan controls overlay (not affected by transform)
  drawZoomPanControls(ctx, width, height);
}

/**
 * Draw zoom/pan control overlay
 */
function drawZoomPanControls(ctx, width, height) {
  // Reset button
  ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
  ctx.fillRect(width - 100, 10, 90, 30);
  ctx.strokeStyle = "#3cff9e";
  ctx.lineWidth = 1;
  ctx.strokeRect(width - 100, 10, 90, 30);
  ctx.fillStyle = "#3cff9e";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Reset View", width - 55, 25);
  
  // Zoom level indicator
  ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
  ctx.fillRect(10, 10, 80, 20);
  ctx.fillStyle = "#7f8fa6";
  ctx.font = "10px system-ui";
  ctx.textAlign = "left";
  ctx.fillText(`Zoom: ${(aoaFusionZoom * 100).toFixed(0)}%`, 15, 23);
}

/**
 * Reset zoom and pan to default
 */
function resetAoAFusionView() {
  aoaFusionZoom = 1.0;
  aoaFusionPanX = 0;
  aoaFusionPanY = 0;
  if (aoaFusionInterval) {
    updateAoAFusion(); // Redraw
  }
}

/**
 * Initialize zoom and pan event handlers for AoA fusion canvas
 */
function initAoAFusionZoomPan() {
  if (!aoaFusionCanvas) return;
  
  // Mouse wheel zoom
  aoaFusionCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    
    const rect = aoaFusionCanvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Zoom factor
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.5, Math.min(5.0, aoaFusionZoom * zoomFactor));
    
    // Zoom toward mouse position
    const zoomChange = newZoom / aoaFusionZoom;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    // Adjust pan to zoom toward mouse
    aoaFusionPanX = mouseX - (mouseX - centerX - aoaFusionPanX) * zoomChange - centerX;
    aoaFusionPanY = mouseY - (mouseY - centerY - aoaFusionPanY) * zoomChange - centerY;
    
    aoaFusionZoom = newZoom;
    
    if (aoaFusionInterval) {
      updateAoAFusion(); // Redraw
    }
  }, { passive: false });
  
  // Mouse pan
  aoaFusionCanvas.addEventListener("mousedown", (e) => {
    // Only pan with left button, and not on reset button
    if (e.button !== 0) return;
    const rect = aoaFusionCanvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    // Check if clicking on reset button
    if (clickX >= rect.width - 100 && clickX <= rect.width - 10 &&
        clickY >= 10 && clickY <= 40) {
      resetAoAFusionView();
      return;
    }
    
    aoaFusionIsPanning = true;
    aoaFusionPanStartX = e.clientX - aoaFusionPanX;
    aoaFusionPanStartY = e.clientY - aoaFusionPanY;
    aoaFusionCanvas.style.cursor = "grabbing";
  });
  
  aoaFusionCanvas.addEventListener("mousemove", (e) => {
    if (!aoaFusionIsPanning) {
      // Update cursor when hovering over reset button
      const rect = aoaFusionCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      if (mouseX >= rect.width - 100 && mouseX <= rect.width - 10 &&
          mouseY >= 10 && mouseY <= 40) {
        aoaFusionCanvas.style.cursor = "pointer";
      } else {
        aoaFusionCanvas.style.cursor = "grab";
      }
      return;
    }
    
    aoaFusionPanX = e.clientX - aoaFusionPanStartX;
    aoaFusionPanY = e.clientY - aoaFusionPanStartY;
    
    if (aoaFusionInterval) {
      updateAoAFusion(); // Redraw
    }
  });
  
  aoaFusionCanvas.addEventListener("mouseup", () => {
    aoaFusionIsPanning = false;
    aoaFusionCanvas.style.cursor = "grab";
  });
  
  aoaFusionCanvas.addEventListener("mouseleave", () => {
    aoaFusionIsPanning = false;
    aoaFusionCanvas.style.cursor = "crosshair";
  });
  
  // Double-click to reset
  aoaFusionCanvas.addEventListener("dblclick", (e) => {
    e.preventDefault();
    resetAoAFusionView();
  });
  
  // Set initial cursor
  aoaFusionCanvas.style.cursor = "grab";
}

/**
 * Update AoA fusion data display
 */
function updateAoAFusionData(cones) {
  if (!aoaStatusEl || !aoaConesListEl || !aoaFusionResultEl) return;
  
  if (!cones || cones.length === 0) {
    aoaStatusEl.textContent = "Waiting for cones...";
    aoaConesListEl.innerHTML = "";
    aoaFusionResultEl.innerHTML = "";
    return;
  }
  
  const validCones = cones.filter(c => c.gps && c.gps.lat && c.gps.lon);
  
  if (validCones.length === 0) {
    aoaStatusEl.textContent = "No GPS data available";
    aoaConesListEl.innerHTML = "";
    aoaFusionResultEl.innerHTML = "";
    return;
  }
  
  // Update status
  aoaStatusEl.textContent = `${validCones.length}/3 cones active`;
  
  // Update cones list
  aoaConesListEl.innerHTML = validCones.map((cone, idx) => {
    const bearing = (cone.bearing_deg || 0).toFixed(1);
    const width = (cone.cone_width_deg || 45).toFixed(1);
    const conf = ((cone.confidence || 0.5) * 100).toFixed(0);
    const color = NODE_COLORS[idx % NODE_COLORS.length];
    const age = cone.timestamp ? Math.round(Date.now() / 1000 - cone.timestamp) : 0;
    
    return `
      <div class="aoa-cone-item">
        <strong style="color: ${color}">${cone.callsign || cone.node_id}</strong>
        Bearing: ${bearing}° | Width: ${width}° | Conf: ${conf}% | Age: ${age}s
      </div>
    `;
  }).join("");
  
  // Calculate and display fusion result
  if (validCones.length >= 2) {
    const intersections = [];
    for (let i = 0; i < validCones.length; i++) {
      for (let j = i + 1; j < validCones.length; j++) {
        const cone1 = validCones[i];
        const cone2 = validCones[j];
        const node1 = { gps: cone1.gps };
        const node2 = { gps: cone2.gps };
        
        const intersection = calculateIntersection(
          node1, cone1.bearing_deg,
          node2, cone2.bearing_deg
        );
        
        if (intersection) {
          intersections.push(intersection);
        }
      }
    }
    
    if (intersections.length > 0) {
      const avgLat = intersections.reduce((sum, p) => sum + p.lat, 0) / intersections.length;
      const avgLon = intersections.reduce((sum, p) => sum + p.lon, 0) / intersections.length;
      const quality = calculateFusionQuality(validCones);
      const qualityPercent = (quality * 100).toFixed(0);
      
      aoaFusionResultEl.innerHTML = `
        <div class="fusion-label">Fused Location</div>
        <div class="fusion-coords">${avgLat.toFixed(6)}, ${avgLon.toFixed(6)}</div>
        <div class="fusion-quality">Quality: ${qualityPercent}%</div>
      `;
    } else {
      aoaFusionResultEl.innerHTML = `<div class="fusion-label">Unable to calculate intersection</div>`;
    }
  } else {
    aoaFusionResultEl.innerHTML = `<div class="fusion-label">Need 2+ cones for fusion</div>`;
  }
}

/**
 * Fetch and update AoA fusion data
 */
async function updateAoAFusion() {
  try {
    const response = await fetch("/api/tripwire/aoa-fusion");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    
    drawAoAFusion(data.cones || []);
    updateAoAFusionData(data.cones || []);
  } catch (err) {
    console.error("[AoA Fusion] Error fetching data:", err);
    if (aoaStatusEl) {
      aoaStatusEl.textContent = "Error loading data";
    }
  }
}

  // Start periodic updates (every 2 seconds)
let aoaFusionInterval = null;

function startAoAFusionUpdates() {
  if (aoaFusionInterval) return;
  // Initialize zoom/pan handlers
  initAoAFusionZoomPan();
  // Resize canvas on window resize (debounced)
  let resizeTimeout;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      resizeAoAFusionCanvas();
      // Redraw after resize
      if (aoaFusionInterval) {
        updateAoAFusion();
      }
    }, 100);
  });
  resizeAoAFusionCanvas(); // Initial resize
  updateAoAFusion(); // Initial update
  aoaFusionInterval = setInterval(updateAoAFusion, 2000);
}

function stopAoAFusionUpdates() {
  if (aoaFusionInterval) {
    clearInterval(aoaFusionInterval);
    aoaFusionInterval = null;
  }
}

function updateTripwireLeds(count) {
  if (!tripwireLedsEl) return;
  tripwireLedsEl.innerHTML = "";
  for (let i = 0; i < MAX_TRIPWIRES; i++) {
    const led = document.createElement("div");
    led.className = "tw-led-collapsed";
    if (i < count) {
      led.classList.add("on");
    }
    tripwireLedsEl.appendChild(led);
  }
}

function initTripwireCollapse() {
  if (!tripwireHeader || !tripwireBody || !tripwireCaret) {
    console.warn("[TW] Tripwire collapse elements not found:", {
      header: !!tripwireHeader,
      body: !!tripwireBody,
      caret: !!tripwireCaret
    });
    return;
  }
  
  // Start closed (collapsed) - matches top dropdown pattern
  tripwireBody.classList.remove("open");
  tripwireCaret.classList.remove("open");
  
  // Make LEDs non-clickable (pointer-events: none handled in CSS)
  if (tripwireLedsEl) {
    tripwireLedsEl.style.pointerEvents = "none";
  }
  
  tripwireHeader.addEventListener("click", (e) => {
    // Don't toggle if clicking directly on LEDs
    if (e.target === tripwireLedsEl || e.target.closest(".tripwire-leds-collapsed")) {
      return;
    }
    
    const open = tripwireBody.classList.contains("open");
    tripwireBody.classList.toggle("open", !open);
    tripwireCaret.classList.toggle("open", !open);
    
    // Close other top dropdowns when this one opens (like other top dropdowns)
    if (!open) {
      const otherDropdowns = document.querySelectorAll(".top-dropdown-content");
      otherDropdowns.forEach(dropdown => {
        if (dropdown !== tripwireBody && dropdown.classList.contains("open")) {
          dropdown.classList.remove("open");
          const otherHeader = dropdown.previousElementSibling;
          if (otherHeader) {
            const otherCaret = otherHeader.querySelector(".caret");
            if (otherCaret) {
              otherCaret.classList.remove("open");
            }
          }
        }
      });
    }
  });
  
  // Add click-outside-to-close listener (like other top dropdowns)
  document.addEventListener("click", (e) => {
    if (!tripwireHeader.contains(e.target) && !tripwireBody.contains(e.target)) {
      tripwireBody.classList.remove("open");
      tripwireCaret.classList.remove("open");
    }
  });
  
  console.log("[TW] Tripwire collapse initialized");
}

async function sendScanPlanToTripwire(nodeId, scanPlan) {
  if (!scanPlan || !nodeId) {
    console.warn("[TW] Cannot send scan plan: missing nodeId or scanPlan");
    return;
  }
  
  try {
    const response = await fetch(`/api/tripwire/scan-plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node_id: nodeId,
        scan_plan: scanPlan
      })
    });
    
    const result = await response.json();
    if (result.ok) {
      console.log(`[TW] Scan plan ${scanPlan} sent to ${nodeId}`);
    } else {
      console.error(`[TW] Failed to send scan plan:`, result.error);
    }
  } catch (e) {
    console.error(`[TW] Error sending scan plan:`, e);
  }
}

// ------------------------------
// CAPTURE LOG
// ------------------------------
function renderCaptureLog(log) {
  if (!captureLogEl) return;
  captureLogEl.innerHTML = "";
  if (!Array.isArray(log) || log.length === 0) {
    const empty = document.createElement("div");
    empty.className = "capture-empty";
    empty.textContent = "No captures yet";
    captureLogEl.appendChild(empty);
    return;
  }
  
  // Get class labels for dropdown
  const classLabels = [
    "elrs", "crossfire", "frsky", "flysky", "ghost", "redpine",
    "dji_mini_4_pro", "dji_avata", "dji_fpv", "dji_mini_3", "dji_air_3",
    "dji_mavic_3", "walksnail", "hdzero", "dji_o3", "dji_o4",
    "analog_fpv", "mavlink", "mavlink2", "lora_telemetry",
    "voice_analog", "voice_digital", "unknown"
  ];
  
  for (const item of log.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "capture-row";
    
    const freqMHz = (Number(item.freq_hz || 0) / 1e6).toFixed(3);
    const durMs = Math.round(Number(item.duration_s || 0) * 1000);
    const when = new Date(Number(item.ts || 0) * 1000).toLocaleTimeString();
    
    // Get current classification
    const currentLabel = item.classification?.label || "unknown";
    const currentConf = item.classification?.confidence || 0;
    const isManualLabel = item.classification?.model === "manual_label";
    
    // Create row content container
    const rowContent = document.createElement("div");
    rowContent.className = "capture-row-content";
    
    // Create info section
    const info = document.createElement("div");
    info.className = "capture-info";
    info.textContent = `[${when}] ${freqMHz} MHz · ${durMs} ms · ${item.reason || "capture"}` +
      (item.source_node ? ` · ${item.source_node}` : "") +
      (item.scan_plan ? ` · ${item.scan_plan}` : "");
    
    // Create label section
    const labelSection = document.createElement("div");
    labelSection.className = "capture-label-section";
    
    // Current label display (compact)
    const labelDisplay = document.createElement("span");
    labelDisplay.className = "capture-label-display";
    labelDisplay.textContent = currentLabel;
    if (isManualLabel) {
      labelDisplay.classList.add("manual-label");
    } else if (currentConf < 1.0 && currentConf > 0) {
      labelDisplay.textContent += ` (${(currentConf * 100).toFixed(0)}%)`;
    }
    
    // Label dropdown (compact)
    const labelSelect = document.createElement("select");
    labelSelect.className = "capture-label-select";
    labelSelect.value = currentLabel;
    labelSelect.title = "Change classification label";
    
    // Add options
    classLabels.forEach(label => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      labelSelect.appendChild(option);
    });
    
    // Handle label change
    labelSelect.addEventListener("change", async (e) => {
      const newLabel = e.target.value;
      if (!item.capture_dir) {
        alert("Cannot label: capture directory not found");
        labelSelect.value = currentLabel; // Revert
        return;
      }
      
      // Disable select while updating
      labelSelect.disabled = true;
      const originalValue = currentLabel;
      
      try {
        const response = await fetch("/api/capture/label", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            capture_dir: item.capture_dir,
            label: newLabel
          })
        });
        
        const result = await response.json();
        if (result.ok) {
          labelDisplay.textContent = newLabel;
          labelDisplay.classList.add("manual-label");
          // Refresh capture list to show updated label
          setTimeout(() => pollCaptures(), 500);
        } else {
          alert(`Failed to update label: ${result.error}`);
          labelSelect.value = originalValue; // Revert
        }
      } catch (err) {
        alert(`Error updating label: ${err.message}`);
        labelSelect.value = originalValue; // Revert
      } finally {
        labelSelect.disabled = false;
      }
    });
    
    labelSection.appendChild(labelDisplay);
    labelSection.appendChild(labelSelect);
    
    rowContent.appendChild(info);
    rowContent.appendChild(labelSection);
    row.appendChild(rowContent);
    captureLogEl.appendChild(row);
  }
}

async function pollCaptures() {
  try {
    const j = await API.captures();
    renderCaptureLog(j?.captures || []);
  } catch (_) {}
}

// ------------------------------
// ML CLASSIFICATIONS
// ------------------------------
function addClassification(classification) {
  if (!classification || typeof classification !== "object") {
    console.warn("[UI] Invalid classification object:", classification);
    return;
  }

  console.log("[UI] Adding classification to buffer:", classification);

  // Add timestamp if not present
  if (!classification.ts) {
    classification.ts = Date.now() / 1000;
  }

  // Add to buffer (newest first)
  classificationBuffer.unshift(classification);
  
  // Keep only last 15 classifications
  if (classificationBuffer.length > 15) {
    classificationBuffer.pop();
  }

  console.log("[UI] Classification buffer length:", classificationBuffer.length);
  renderClassifications();
}

function renderClassifications() {
  if (!mlClassificationsEl) {
    console.warn("[UI] mlClassificationsEl not found, cannot render classifications");
    return;
  }

  console.log("[UI] Rendering classifications, buffer length:", classificationBuffer.length);

  if (classificationBuffer.length === 0) {
    mlClassificationsEl.innerHTML = '<div class="muted">No classifications yet...</div>';
    return;
  }

  mlClassificationsEl.innerHTML = classificationBuffer.map((cls, idx) => {
    const label = (cls.label || "UNKNOWN").toUpperCase().replace(/_/g, " ");
    const confidence = (cls.confidence || 0) * 100;
    const freqMHz = cls.freq_hz ? (cls.freq_hz / 1e6).toFixed(3) : "—";
    const sourceNode = cls.source_node || "—";
    
    // Calculate time ago
    const now = Date.now() / 1000;
    const timeAgo = now - (cls.ts || now);
    let timeAgoStr = "";
    if (timeAgo < 60) {
      timeAgoStr = `${Math.round(timeAgo)}s ago`;
    } else if (timeAgo < 3600) {
      timeAgoStr = `${Math.round(timeAgo / 60)}m ago`;
    } else {
      timeAgoStr = `${Math.round(timeAgo / 3600)}h ago`;
    }

    // Determine confidence class
    let confidenceClass = "low-confidence";
    if (confidence >= 80) {
      confidenceClass = "high-confidence";
    } else if (confidence >= 50) {
      confidenceClass = "medium-confidence";
    }

    return `
      <div class="ml-classification-card ${confidenceClass}">
        <div class="ml-label">${label}</div>
        <div class="ml-details">
          <span>${freqMHz} MHz</span>
          <span class="ml-confidence">${confidence.toFixed(0)}%</span>
        </div>
        <div class="ml-meta">
          <span>${sourceNode}</span>
          <span>${timeAgoStr}</span>
        </div>
      </div>
    `;
  }).join("");
}

// ------------------------------
// TAK COLLAPSE & INIT
// ------------------------------
function initTakCollapse() {
  if (!takHeader || !takBody || !takCaret) return;
  takBody.classList.remove("open");
  takCaret.classList.remove("open");
  takHeader.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = takBody.classList.contains("open");
    takBody.classList.toggle("open", !open);
    takCaret.classList.toggle("open", !open);
  });
  
  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (takBody && takBody.classList.contains("open")) {
      const dropdown = takHeader.closest(".top-dropdown");
      if (dropdown && !dropdown.contains(e.target)) {
        takBody.classList.remove("open");
        takCaret.classList.remove("open");
      }
    }
  });
}

// NETWORK CONFIG COLLAPSE & INIT
// ------------------------------
function initNetworkCollapse() {
  if (!networkHeader || !networkBody || !networkCaret) return;
  networkBody.classList.remove("open");
  networkCaret.classList.remove("open");
  networkHeader.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = networkBody.classList.contains("open");
    networkBody.classList.toggle("open", !open);
    networkCaret.classList.toggle("open", !open);
    // Load current addresses when opening
    if (open === false) {
      loadNetworkConfig();
    }
  });
  
  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (networkBody && networkBody.classList.contains("open")) {
      const dropdown = networkHeader.closest(".top-dropdown");
      if (dropdown && !dropdown.contains(e.target)) {
        networkBody.classList.remove("open");
        networkCaret.classList.remove("open");
      }
    }
  });
}

async function loadNetworkConfig() {
  try {
    const config = await API.getNetworkConfig();
    if (config) {
      if (l4tbr0Input && config.l4tbr0) {
        l4tbr0Input.value = config.l4tbr0;
      }
      if (eth0Input && config.eth0) {
        eth0Input.value = config.eth0;
      }
    }
  } catch (e) {
    console.error("[NETWORK] Failed to load config:", e);
  }
}

async function setNetworkInterface(interfaceName) {
  const input = interfaceName === "l4tbr0" ? l4tbr0Input : eth0Input;
  if (!input) return;
  
  const address = input.value.trim();
  if (!address) {
    alert(`Please enter an address for ${interfaceName}`);
    return;
  }
  
  // Basic IP validation
  const ipRegex = /^(\d{1,3}\.){3}\d{1,3}(\/\d{1,2})?$/;
  if (!ipRegex.test(address)) {
    alert(`Invalid IP address format: ${address}`);
    return;
  }
  
  try {
    const result = await API.setNetworkInterface(interfaceName, address);
    if (result.ok) {
      console.log(`[NETWORK] Successfully set ${interfaceName} to ${address}`);
      alert(`Successfully set ${interfaceName} to ${address}`);
      // Reload config to show updated value
      await loadNetworkConfig();
    } else {
      console.error(`[NETWORK] Failed to set ${interfaceName}:`, result.error);
      alert(`Failed to set ${interfaceName}: ${result.error || "Unknown error"}`);
    }
  } catch (e) {
    console.error(`[NETWORK] Error setting ${interfaceName}:`, e);
    alert(`Error setting ${interfaceName}: ${e.message}`);
  }
}

function init() {
  // Initialize DOM element references (ensure DOM is ready)
  if (!mlClassificationsEl) {
    mlClassificationsEl = document.getElementById("mlClassificationsList");
    if (mlClassificationsEl) {
      console.log("[UI] ML classifications element found");
    } else {
      console.error("[UI] ML classifications element NOT found! Check HTML.");
    }
  }
  
  resizeCanvas(true);
  setEdgeMode("manual");
  initTakCollapse();
  initNetworkCollapse();
  initTripwireCollapse();
  updateGainUiLock();

  // -------------------------------------------------
  // INIT FIXED TRIPWIRE NODE CARDS (ALWAYS VISIBLE)
  // -------------------------------------------------
  initTripwireNodes();
  
  // -------------------------------------------------
  // INIT AoA FUSION VISUALIZATION
  // -------------------------------------------------
  startAoAFusionUpdates();

  // -------------------------------------------------
  // UI wiring
  // -------------------------------------------------
  if (btnApplySdr) btnApplySdr.addEventListener("click", applySdrConfig);
  if (btnManualCapture) btnManualCapture.addEventListener("click", startManualCapture);
  if (gainModeSelect) gainModeSelect.addEventListener("change", updateGainUiLock);
  
  // Frequency, sample rate, and bandwidth apply ONLY when "Apply SDR Settings" is clicked.
  // (Gain slider still applies in real time via its own debounced handler below.)
  if (btnManual) btnManual.addEventListener("click", () => apiSetEdgeMode("manual"));
  if (btnArmed) btnArmed.addEventListener("click", () => apiSetEdgeMode("armed"));
  
  if (btnSetL4tbr0) btnSetL4tbr0.addEventListener("click", () => setNetworkInterface("l4tbr0"));
  if (btnSetEth0) btnSetEth0.addEventListener("click", () => setNetworkInterface("eth0"));
  
  // Gain slider with debouncing for smooth updates
  let gainDebounceTimer = null;
  if (gainSlider) {
    gainSlider.addEventListener("input", (e) => {
      const value = parseInt(e.target.value);
      lastGainSliderInputAt = Date.now();
      // Update display immediately for smooth UI feedback
      if (gainValue) {
        gainValue.textContent = value;
      }
      
      // Clear existing debounce timer
      if (gainDebounceTimer) {
        clearTimeout(gainDebounceTimer);
      }
      
      // Debounce: Wait 200ms after last slider movement before applying
      // Use gain-only API so we never send partial freq/rate from the form (no tune on every stroke)
      gainDebounceTimer = setTimeout(() => {
        if (edgeMode !== "tasked") {
          const gainMode = gainModeSelect?.value || "manual";
          API.sdrGainOnly({ gain_mode: gainMode, gain_db: value }).then(() => {
            refreshStatus();
            pollSdrInfo();
          }).catch(() => {});
        }
        gainDebounceTimer = null;
      }, 200); // 200ms debounce delay
    });
  }
  
  // Waterfall brightness/contrast controls
  if (wfBrightnessSlider) {
    wfBrightnessSlider.addEventListener("input", (e) => {
      wfBrightness = parseFloat(e.target.value);
      if (wfBrightnessValue) {
        wfBrightnessValue.textContent = wfBrightness > 0 ? `+${wfBrightness}` : wfBrightness;
      }
    });
  }
  
  if (wfContrastSlider) {
    wfContrastSlider.addEventListener("input", (e) => {
      wfContrast = parseFloat(e.target.value);
      if (wfContrastValue) {
        wfContrastValue.textContent = wfContrast.toFixed(1);
      }
    });
  }
  
  // Spectrum display controls
  if (traceModeSelect) {
    traceModeSelect.addEventListener("change", (e) => {
      traceMode = e.target.value;
      // Clear trace buffers when switching modes
      if (traceMode !== "peak") peakHoldSpectrum = null;
      if (traceMode !== "average") averageSpectrum = null;
      if (traceMode !== "min") minHoldSpectrum = null;
    });
  }
  
  if (waterfallPaletteSelect) {
    waterfallPaletteSelect.addEventListener("change", (e) => {
      waterfallPalette = e.target.value;
      // Clear waterfall to apply new palette
      clearWaterfall();
    });
  }
  
  if (timeMarkersCheckbox) {
    timeMarkersCheckbox.addEventListener("change", (e) => {
      timeMarkersEnabled = e.target.checked;
      if (!timeMarkersEnabled) {
        waterfallTimestamps = [];  // Clear timestamps when disabled
      }
    });
  }

  // FFT Smoothing control
  const smoothingSlider = document.getElementById("smoothingSlider");
  const smoothingValue = document.getElementById("smoothingValue");
  let smoothingTimeout;
  
  if (smoothingSlider && smoothingValue) {
    // Convert slider value (0-100) to alpha (0.0-1.0)
    // Slider value 0 = alpha 0.0 (no smoothing), 100 = alpha 1.0 (full smoothing)
    // Actually, we want: slider 0 = alpha 0.01 (heavy), slider 100 = alpha 1.0 (none)
    // So: alpha = 0.01 + (slider / 100) * 0.99
    function sliderToAlpha(sliderVal) {
      return 0.01 + (sliderVal / 100) * 0.99;
    }
    
    function alphaToSlider(alpha) {
      return Math.round(((alpha - 0.01) / 0.99) * 100);
    }
    
    smoothingSlider.addEventListener("input", (e) => {
      const sliderVal = parseInt(e.target.value);
      const alpha = sliderToAlpha(sliderVal);
      smoothingValue.textContent = alpha.toFixed(2);
      
      // Debounce API call (200ms)
      clearTimeout(smoothingTimeout);
      smoothingTimeout = setTimeout(async () => {
        try {
          const response = await fetch("/api/live/smoothing", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ alpha: alpha }),
          });
          const result = await response.json();
          if (result.ok) {
            console.log(`[UI] FFT smoothing set to ${alpha.toFixed(3)}`);
          }
        } catch (err) {
          console.error("[UI] Failed to set smoothing:", err);
        }
      }, 200);
    });
  }

  // -------------------------------------------------
  // Initial state fetch
  // -------------------------------------------------
  refreshStatus();
  pollCaptures();
  pollSdrInfo();
  refreshEdgeMode();

  // -------------------------------------------------
  // Event-driven notifications (Tripwire, mode, tasks)
  // -------------------------------------------------
  startNotifyWs();

  // -------------------------------------------------
  // Polling loops (NO Tripwire polling)
  // -------------------------------------------------
  setInterval(refreshStatus, POLL_STATUS_MS);
  setInterval(pollCaptures, POLL_CAPTURES_MS);
  setInterval(pollSdrInfo, POLL_SDRINFO_MS);
  
  // Start health polling after a short delay to ensure everything is initialized
  setTimeout(() => {
    setInterval(pollSdrHealth, POLL_SDRHEALTH_MS);
    pollSdrHealth();
  }, 2000);
  
  // When user returns to this tab (e.g. from ML page), re-check status and reconnect FFT if scan is running
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      refreshStatus();
    }
  });
}

// Kick everything off
init();