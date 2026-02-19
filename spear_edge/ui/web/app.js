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
const tripwireNodesEl   = document.getElementById("tripwireNodes");
const tripwireHeader    = document.getElementById("tripwireHeader");
const tripwireBody      = document.getElementById("tripwireBody");
const tripwireCaret     = tripwireHeader ? tripwireHeader.querySelector(".caret") : null;
const tripwireLedsEl    = document.getElementById("tripwireLeds");
const cueListEl         = document.getElementById("cueList");
const captureLogEl      = document.getElementById("captureLog");
const taskBanner        = document.getElementById("taskBanner");
const taskFromEl        = document.getElementById("taskFrom");
const taskFreqEl        = document.getElementById("taskFreq");
const taskProfileEl     = document.getElementById("taskProfile");
const takHeader         = document.getElementById("takHeader");
const takBody           = document.getElementById("takBody");
const takCaret          = takHeader ? takHeader.querySelector(".caret") : null;
const modePill          = document.getElementById("modePill");
const armedBanner       = document.getElementById("armedBanner");
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
const gainSlider        = document.getElementById("gainSlider");      // dB
const wfBrightnessSlider = document.getElementById("wfBrightnessSlider");
const wfContrastSlider   = document.getElementById("wfContrastSlider");
const wfBrightnessValue  = document.getElementById("wfBrightnessValue");
const wfContrastValue    = document.getElementById("wfContrastValue");
const btnApplySdr       = document.getElementById("btnApplySdr");
const btnManualCapture  = document.getElementById("btnManualCapture");
const bandwidthInput    = document.getElementById("bandwidthInput");
const fpsSelect         = document.getElementById("fpsSelect");       // FPS

const sdrDriverEl       = document.getElementById("sdr-driver");
const sdrCenterEl       = document.getElementById("sdr-center");
const sdrRateEl         = document.getElementById("sdr-rate");
const sdrGainEl         = document.getElementById("sdr-gain");
const sdrGainModeEl     = document.getElementById("sdr-gain-mode");

const edgeModeLabel     = document.getElementById("edgeModeLabel");
const btnManual         = document.getElementById("btnManual");
const btnArmed          = document.getElementById("btnArmed");


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

// Waterfall display controls
let wfBrightness = 0;  // Offset in dB (-50 to +50)
let wfContrast = 1.0;  // Contrast multiplier (0.1 to 3.0)
let lastCanvasW         = 0;
let lastCanvasH         = 0;

// ------------------------------
// OPERATOR CUE QUEUE (fixed + robust)
// ------------------------------
const cueBuffer = []; // newest first, max 10

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
    const mode = j.mode || "unknown";
    if (edgeModeLabel) edgeModeLabel.textContent = `Mode: ${mode}`;
    if (btnManual) btnManual.classList.toggle("active", mode === "manual");
    if (btnArmed) btnArmed.classList.toggle("active", mode === "armed");
    if (armedBanner) armedBanner.style.display = mode === "armed" ? "block" : "none";
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
  if (modePill) {
    modePill.classList.remove("manual", "armed", "tasked");
    modePill.textContent = edgeMode === "manual" ? "MODE: MANUAL" :
                           edgeMode === "armed" ? "MODE: ARMED" : "MODE: TASKED";
    modePill.classList.add(edgeMode);
    edgeMode === "tasked" ? lockSdrControls() : unlockSdrControls();
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
  if (!gainSlider) return;
  const locked = edgeMode === "tasked" || gainModeSelect?.value === "agc";
  gainSlider.disabled = locked;
  gainSlider.classList.toggle("locked", locked);
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
}

// ------------------------------
// SDR CONFIG
// ------------------------------
function readSdrForm() {
  const mhz = Number(freqInput?.value ?? 915.0);
  const msps = Number(rateInput?.value ?? 2.0);
  const fftSize = Number(fftSizeSelect?.value ?? 1024);
  const bandwidthMhz = bandwidthInput?.value ? Number(bandwidthInput.value) : null;
  const fps = Number(fpsSelect?.value ?? 30);
  return {
    center_freq_hz: Math.round(mhz * 1e6),
    sample_rate_sps: Math.round(msps * 1e6),
    fft_size: Number.isFinite(fftSize) ? fftSize : 1024,
    fps: Number.isFinite(fps) ? fps : 30.0,
    gain_mode: gainModeSelect?.value || "manual",
    gain_db: Number(gainSlider?.value ?? 30),
    rx_channel: 0,
    bandwidth_hz: bandwidthMhz ? Math.round(bandwidthMhz * 1e6) : null,
  };
}

async function applySdrConfig() {
  if (edgeMode === "tasked") return;
  const cfg = readSdrForm();
  try {
    await API.sdrConfig({
      center_freq_hz: cfg.center_freq_hz,
      sample_rate_sps: cfg.sample_rate_sps,
      gain_mode: cfg.gain_mode,
      gain_db: cfg.gain_db,
      rx_channel: cfg.rx_channel,
      bandwidth_hz: cfg.bandwidth_hz,
    });
  } catch (e) {
    console.warn("[SDR] config failed:", e);
  }
  refreshStatus();
  pollSdrInfo();
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
function drawPowerAxis(ctx, fftH, w, dbMin, dbMax) {
  ctx.save();
  ctx.fillStyle = "rgba(0,255,136,0.9)";
  ctx.font = "11px monospace";
  ctx.textAlign = "left";
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const t = i / ticks;
    const db = dbMin + t * (dbMax - dbMin);
    const y = fftH - t * fftH;
    const label = db.toFixed(0) + " dBFS";
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
  const pxFftH = Math.floor(cssH * FFT_HEIGHT_FRAC * dpr);
  const pxWfH  = pxH - pxFftH;

  // CSS-space for FFT drawing
  const w = (canvas.width / dpr) || 1;
  const h = (canvas.height / dpr) || 1;
  const fftH = Math.floor(h * FFT_HEIGHT_FRAC);
  const wfH = h - fftH;

  // Pick sources: max-hold for FFT, instant for waterfall
  const fftArr = frame.power_dbfs;  // max-hold (for FFT line)
  const wfArr = frame.power_inst_dbfs || frame.power_dbfs;  // instant (for waterfall)

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
    if (smoothedNoiseFloor == null) {
      // Initialize from median-ish bin
      const midIdx = Math.floor(wfArr.length * 0.5);
      const mid = wfArr[midIdx];
      smoothedNoiseFloor = Number.isFinite(mid) ? Number(mid) : -90;
    } else {
      // Slow drift only
      const midIdx = Math.floor(wfArr.length * 0.5);
      if (midIdx < wfArr.length) {
        const mid = wfArr[midIdx];
        if (Number.isFinite(mid)) {
          smoothedNoiseFloor = 0.98 * smoothedNoiseFloor + 0.02 * Number(mid);
        }
      }
    }

    const wfDbMin = smoothedNoiseFloor - 10;
    const wfDbMax = smoothedNoiseFloor + 40;
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

      const o = x * 4;
      data[o + 0] = Math.floor(30 * t);
      data[o + 1] = Math.floor(255 * t);
      data[o + 2] = Math.floor(10 * t);
      data[o + 3] = 255;
    }

    ctx.putImageData(row, 0, pxFftH);
    ctx.restore();
  }

  // ---------
  // Visual leveling (stable / clamped) for FFT
  // - Estimate noise floor from FFT trace ONLY
  // - Use a low percentile so brief FHSS energy doesn't pull the floor
  // - Clamp offset motion per frame to prevent hopping
  // ---------
  const TARGET_FLOOR_DB = -80.0;
  const DB_MIN = -120.0;
  const floorSrc = fftArr;

  // robust percentile (5%)
  const sorted = floorSrc
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  const floorIdx = Math.max(0, Math.floor(sorted.length * 0.05));
  const noiseFloorRaw = sorted.length ? sorted[floorIdx] : -90;

  // desired offset to place floor at TARGET_FLOOR_DB
  const desiredOffset = TARGET_FLOOR_DB - noiseFloorRaw;

  if (window._fftVisOffset === undefined) {
    window._fftVisOffset = desiredOffset;
  } else {
    // clamp movement (dB per rendered frame)
    const delta = desiredOffset - window._fftVisOffset;
    const MAX_STEP_DB = 0.25; // smaller = more stable
    const step = Math.max(-MAX_STEP_DB, Math.min(MAX_STEP_DB, delta));
    window._fftVisOffset += step;
  }

  const visOffset = window._fftVisOffset || 0.0;

  // Apply offset ONLY for FFT drawing
  const p = fftArr.map(v => {
    const x = Number(v) + visOffset;
    return Number.isFinite(x) ? x : DB_MIN;
  });

  // Fixed display range for FFT
  const dbMin = -90;
  const dbMax = -20;

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

  // Smoothing for FFT trace
  const dbNow = p.map(v => clamp(v, dbMin, dbMax));
  if (!lastSpectrum || lastSpectrum.length !== dbNow.length) {
    lastSpectrum = dbNow.slice();
  } else {
    for (let i = 0; i < dbNow.length; i++) {
      lastSpectrum[i] = FFT_SMOOTH_ALPHA * dbNow[i] + (1 - FFT_SMOOTH_ALPHA) * lastSpectrum[i];
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
  drawPowerAxis(ctx, fftH, w, dbMin, dbMax);
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
  return {
    ts,
    center_freq_hz: centerFreqHz,
    sample_rate_sps: sampleRateSps,
    fft_size: fftSize,
    power_dbfs: power0,
    power_inst_dbfs: power1,
    noise_floor_dbfs: noiseFloor,
    // freqs_hz intentionally omitted (we'll compute axis from meta)
  };
}

function startFftWs() {
  if (fftWs) {
    console.log("[FFT WS] WebSocket already exists, skipping");
    return;
  }
  console.log("[FFT WS] Creating new WebSocket connection to:", `ws://${location.host}/ws/live_fft`);
  fftWs = new WebSocket(`ws://${location.host}/ws/live_fft`);
  fftWs.binaryType = "arraybuffer";
  fftWs.onopen = () => {
    console.log("[FFT WS] WebSocket connected successfully!");
    setLed(wsLed, true);
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

        // optional: handle hello
        if (msg && msg.type === "hello") {
          return;
        }

        // JSON fallback frame
        drawSpectrum(msg);
        return;
      }

      // 2) Binary frames
      const frame = parseBinarySpectrumFrame(ev.data);
      if (frame) {
        drawSpectrum(frame);
      }
    } catch (e) {
      console.warn("[FFT WS] onmessage parse failed:", e);
    }
  };
  fftWs.onerror = () => setLed(wsLed, false);
  fftWs.onclose = () => {
    setLed(wsLed, false);
    fftWs = null;
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
    console.log("[NOTIFY WS] SUCCESS: Connected! Ready for nodes, cues, and mode updates.");
    setLed("wsLed", true);
  };

  notifyWs.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      console.log("[NOTIFY WS] Received message:", msg);

      if (msg.type === "tripwire_nodes") {
        console.log("[NOTIFY WS] Updating node cards -", msg.payload.nodes.length, "nodes");
        renderTripwireNodes(msg.payload.nodes || []);

      } else if (msg.type === "edge_mode") {
        console.log("[NOTIFY WS] Edge mode changed to:", msg.payload.mode);
        refreshEdgeModeDisplay(msg.payload.mode);

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
        console.log("[CAPTURE] Capture started:", msg.payload);
        if (captureBanner) {
          captureBanner.classList.remove("hidden");
          captureBanner.classList.add("active");
        }
        if (captureProgressBar) {
          captureProgressBar.style.width = "0%";
        }
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
        console.log("[CAPTURE] Capture completed:", msg.payload);
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
          // Force a style update
          captureBanner.style.display = "none";
          // Remove inline style after a moment to let CSS take over
          setTimeout(() => {
            if (captureBanner) {
              captureBanner.style.display = "";
            }
          }, 100);
        }
      }

      // You can add more message types here later (e.g. classification_result)

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
  if (sdrCenterEl) sdrCenterEl.textContent = cfg.center_freq_hz ? (cfg.center_freq_hz / 1e6).toFixed(3) + " MHz" : "—";
  if (sdrRateEl) sdrRateEl.textContent = cfg.sample_rate_sps ? (cfg.sample_rate_sps / 1e6).toFixed(2) + " MS/s" : "—";
  if (sdrGainEl) sdrGainEl.textContent = (cfg.gain_db !== undefined) ? (cfg.gain_db + " dB") : "—";
  if (sdrGainModeEl) sdrGainModeEl.textContent = cfg.gain_mode || "manual";
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
      const timeouts = (typeof health.timeouts === "number") ? health.timeouts : 0;
      errorsEl.textContent = `${errors} err, ${timeouts} timeout`;
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
  
  // Start closed (collapsed) - panel hidden, LEDs visible
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
    console.log("[TW] Panel toggled, open:", !open);
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
  for (const item of log.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "capture-row";
    const freqMHz = (Number(item.freq_hz || 0) / 1e6).toFixed(3);
    const durMs = Math.round(Number(item.duration_s || 0) * 1000);
    const when = new Date(Number(item.ts || 0) * 1000).toLocaleTimeString();
    row.textContent =
      `[${when}] ${freqMHz} MHz · ${durMs} ms · ${item.reason || "capture"}` +
      (item.source_node ? ` · ${item.source_node}` : "") +
      (item.scan_plan ? ` · ${item.scan_plan}` : "");
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
// TAK COLLAPSE & INIT
// ------------------------------
function initTakCollapse() {
  if (!takHeader || !takBody || !takCaret) return;
  takBody.classList.remove("open");
  takCaret.classList.remove("open");
  takHeader.addEventListener("click", () => {
    const open = takBody.classList.contains("open");
    takBody.classList.toggle("open", !open);
    takCaret.classList.toggle("open", !open);
  });
}

function init() {
  resizeCanvas(true);
  setEdgeMode("manual");
  initTakCollapse();
  initTripwireCollapse();
  updateGainUiLock();

  // -------------------------------------------------
  // INIT FIXED TRIPWIRE NODE CARDS (ALWAYS VISIBLE)
  // -------------------------------------------------
  initTripwireNodes();

  // -------------------------------------------------
  // UI wiring
  // -------------------------------------------------
  if (btnApplySdr) btnApplySdr.addEventListener("click", applySdrConfig);
  if (btnManualCapture) btnManualCapture.addEventListener("click", startManualCapture);
  if (gainModeSelect) gainModeSelect.addEventListener("change", updateGainUiLock);
  if (btnManual) btnManual.addEventListener("click", () => apiSetEdgeMode("manual"));
  if (btnArmed) btnArmed.addEventListener("click", () => apiSetEdgeMode("armed"));
  
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
}

// Kick everything off
init();