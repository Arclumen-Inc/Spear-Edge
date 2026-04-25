// ======================================================
// SPEAR-EDGE ML DASHBOARD
// ML model management and capture labeling interface
// ======================================================

// ------------------------------
// DOM REFERENCES
// ------------------------------
const captureGrid = document.getElementById("captureGrid");
const filterSearch = document.getElementById("filterSearch");
const filterLabel = document.getElementById("filterLabel");
const filterSource = document.getElementById("filterSource");
const btnRefreshCaptures = document.getElementById("btnRefreshCaptures");
const btnBatchLabel = document.getElementById("btnBatchLabel");
const btnExportSelected = document.getElementById("btnExportSelected");
const btnDeleteSelected = document.getElementById("btnDeleteSelected");

const currentModelInfo = document.getElementById("currentModelInfo");
const modelList = document.getElementById("modelList");
const btnExportModel = document.getElementById("btnExportModel");
const fileImportModel = document.getElementById("fileImportModel");
const btnTestModel = document.getElementById("btnTestModel");

const trainLabel = document.getElementById("trainLabel");
const trainEpochs = document.getElementById("trainEpochs");
const btnQuickTrain = document.getElementById("btnQuickTrain");
const btnCancelTrain = document.getElementById("btnCancelTrain");
const trainProgress = document.getElementById("trainProgress");
const trainProgressFill = document.getElementById("trainProgressFill");
const trainStatus = document.getElementById("trainStatus");

const mlStats = document.getElementById("mlStats");
const imagePreviewModal = document.getElementById("imagePreviewModal");
const imagePreviewImg = document.getElementById("imagePreviewImg");
const imagePreviewTitle = document.getElementById("imagePreviewTitle");
const imagePreviewClose = document.getElementById("imagePreviewClose");

// ------------------------------
// STATE
// ------------------------------
let allCaptures = [];
let filteredCaptures = [];
let selectedCaptures = new Set();
let classLabels = {};
let currentModel = null;
let trainingJobId = null;
let trainingPollInterval = null;

// ------------------------------
// API FUNCTIONS
// ------------------------------
const API = {
  async getCaptures(params = {}) {
    const query = new URLSearchParams(params).toString();
    const r = await fetch(`/api/ml/captures?${query}`);
    if (!r.ok) throw new Error("Failed to fetch captures");
    return await r.json();
  },

  async getCapture(captureDir) {
    const r = await fetch(`/api/ml/captures/${captureDir}`);
    if (!r.ok) throw new Error("Failed to fetch capture");
    return await r.json();
  },

  async updateLabel(captureDir, label) {
    const r = await fetch(`/api/ml/captures/${captureDir}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    if (!r.ok) throw new Error("Failed to update label");
    return await r.json();
  },

  async batchUpdateLabels(captures, label) {
    const r = await fetch(`/api/ml/captures/batch-label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ captures: Array.from(captures), label }),
    });
    if (!r.ok) throw new Error("Failed to batch update labels");
    return await r.json();
  },

  async getModels() {
    const r = await fetch("/api/ml/models");
    if (!r.ok) throw new Error("Failed to fetch models");
    return await r.json();
  },

  async getCurrentModel() {
    const r = await fetch("/api/ml/models/current");
    if (!r.ok) throw new Error("Failed to fetch current model");
    return await r.json();
  },

  async exportModel() {
    const r = await fetch("/api/ml/models/export", { method: "POST" });
    if (!r.ok) throw new Error("Failed to export model");
    const blob = await r.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `spear_edge_model_${Date.now()}.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  },

  async importModel(file) {
    const formData = new FormData();
    formData.append("file", file);
    const r = await fetch("/api/ml/models/import", {
      method: "POST",
      body: formData,
    });
    if (!r.ok) {
      const error = await r.json();
      throw new Error(error.detail || "Failed to import model");
    }
    return await r.json();
  },

  async testModel(captureDir) {
    const r = await fetch("/api/ml/models/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ capture_dir: captureDir }),
    });
    if (!r.ok) throw new Error("Failed to test model");
    return await r.json();
  },

  async getClassLabels() {
    const r = await fetch("/api/ml/class-labels");
    if (!r.ok) throw new Error("Failed to fetch class labels");
    return await r.json();
  },

  async getStats() {
    const r = await fetch("/api/ml/stats");
    if (!r.ok) throw new Error("Failed to fetch stats");
    return await r.json();
  },

  async activateModel(modelPath) {
    const r = await fetch("/api/ml/models/activate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_path: modelPath }),
    });
    if (!r.ok) {
      const error = await r.json();
      throw new Error(error.detail || "Failed to activate model");
    }
    return await r.json();
  },

  async quickTrain(captures, label, epochs = 15) {
    const r = await fetch("/api/ml/train/quick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        captures: Array.from(captures),
        label: label,
        epochs: epochs,
        batch_size: 2,
        learning_rate: 1e-4,
      }),
    });
    if (!r.ok) {
      const error = await r.json();
      throw new Error(error.detail || "Failed to start training");
    }
    return await r.json();
  },

  async getTrainingStatus(jobId) {
    const r = await fetch(`/api/ml/train/status/${jobId}`);
    if (!r.ok) throw new Error("Failed to fetch training status");
    return await r.json();
  },

  async cancelTraining(jobId) {
    const r = await fetch(`/api/ml/train/cancel/${jobId}`, {
      method: "POST",
    });
    if (!r.ok) throw new Error("Failed to cancel training");
    return await r.json();
  },
};

// ------------------------------
// CAPTURE MANAGEMENT
// ------------------------------
async function loadCaptures() {
  try {
    captureGrid.innerHTML = '<div class="empty-state">Loading captures...</div>';
    const data = await API.getCaptures({ limit: 200 });
    allCaptures = data.captures || [];
    applyFilters();
  } catch (e) {
    console.error("[ML] Failed to load captures:", e);
    captureGrid.innerHTML = `<div class="empty-state">Error loading captures: ${e.message}</div>`;
  }
}

function applyFilters() {
  const search = filterSearch.value.toLowerCase();
  const labelFilter = filterLabel.value;
  const sourceFilter = filterSource.value;

  filteredCaptures = allCaptures.filter((cap) => {
    if (search && !cap.capture_dir.toLowerCase().includes(search)) {
      return false;
    }
    if (labelFilter && cap.label !== labelFilter) {
      return false;
    }
    if (sourceFilter && cap.source !== sourceFilter) {
      return false;
    }
    return true;
  });

  renderCaptures();
}

function renderCaptures() {
  if (filteredCaptures.length === 0) {
    captureGrid.innerHTML = '<div class="empty-state">No captures found</div>';
    return;
  }

  captureGrid.innerHTML = filteredCaptures
    .map((cap) => {
      const selected = selectedCaptures.has(cap.capture_dir) ? "selected" : "";
      const thumbUrl = cap.thumbnail_url || "";
      const label = cap.label || "unlabeled";
      const confidence = cap.confidence ? `${(cap.confidence * 100).toFixed(0)}%` : "—";
      const freq = (cap.freq_hz / 1e6).toFixed(3);
      const timestamp = new Date(cap.timestamp * 1000).toLocaleString();

      return `
        <div class="capture-card ${selected}" data-capture-dir="${cap.capture_dir}">
          <input type="checkbox" ${selected ? "checked" : ""} 
                 onchange="toggleCapture('${cap.capture_dir}')">
          ${thumbUrl ? `<img src="${thumbUrl}" class="capture-thumb" alt="Spectrogram">` : '<div class="capture-thumb" style="display: flex; align-items: center; justify-content: center; color: var(--text-muted);">No thumbnail</div>'}
          <div class="capture-info">
            <div style="font-weight: 600; color: var(--text-main); margin-bottom: 4px;">${cap.capture_dir.substring(0, 20)}...</div>
            <div>Freq: ${freq} MHz</div>
            <div>${timestamp}</div>
            <div class="capture-label">
              <select onchange="updateCaptureLabel('${cap.capture_dir}', this.value)" 
                      data-capture-dir="${cap.capture_dir}">
                <option value="">-- Select Label --</option>
                ${Object.keys(classLabels)
                  .map(
                    (labelId) =>
                      `<option value="${labelId}" ${label === labelId ? "selected" : ""}>${classLabels[labelId] || labelId}</option>`
                  )
                  .join("")}
              </select>
            </div>
            <div style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">
              ${label !== "unlabeled" ? `Label: ${label} (${confidence})` : "Unlabeled"}
            </div>
          </div>
        </div>
      `;
    })
    .join("");

  updateBatchButtons();
}

function toggleCapture(captureDir) {
  if (selectedCaptures.has(captureDir)) {
    selectedCaptures.delete(captureDir);
  } else {
    selectedCaptures.add(captureDir);
  }
  renderCaptures();
}

async function updateCaptureLabel(captureDir, label) {
  if (!label) return;

  try {
    await API.updateLabel(captureDir, label);
    // Reload captures to get updated data
    await loadCaptures();
  } catch (e) {
    console.error("[ML] Failed to update label:", e);
    alert(`Failed to update label: ${e.message}`);
  }
}

async function batchUpdateLabels() {
  if (selectedCaptures.size === 0) return;

  const label = prompt("Enter label for selected captures:");
  if (!label) return;

  try {
    const result = await API.batchUpdateLabels(selectedCaptures, label);
    alert(`Updated ${result.updated} captures. ${result.errors > 0 ? `${result.errors} errors.` : ""}`);
    selectedCaptures.clear();
    await loadCaptures();
  } catch (e) {
    console.error("[ML] Failed to batch update labels:", e);
    alert(`Failed to batch update labels: ${e.message}`);
  }
}

function updateBatchButtons() {
  const hasSelection = selectedCaptures.size > 0;
  btnBatchLabel.disabled = !hasSelection;
  btnExportSelected.disabled = !hasSelection;
  btnDeleteSelected.disabled = !hasSelection;
  btnTestModel.disabled = !hasSelection || selectedCaptures.size !== 1;
  
  // Update Quick Train button (API requires 2–12 captures; one may be held out for val)
  const canTrain = hasSelection &&
                   selectedCaptures.size >= 2 &&
                   selectedCaptures.size <= 12 &&
                   trainLabel.value !== "";
  btnQuickTrain.disabled = !canTrain || trainingJobId !== null;
}

function openImagePreview(imageUrl, captureDir) {
  if (!imagePreviewModal || !imagePreviewImg) return;

  imagePreviewImg.src = imageUrl;
  imagePreviewTitle.textContent = captureDir || "Spectrogram preview";
  imagePreviewModal.classList.add("open");
  imagePreviewModal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closeImagePreview() {
  if (!imagePreviewModal || !imagePreviewImg) return;

  imagePreviewModal.classList.remove("open");
  imagePreviewModal.setAttribute("aria-hidden", "true");
  imagePreviewImg.src = "";
  document.body.style.overflow = "";
}

// ------------------------------
// MODEL MANAGEMENT
// ------------------------------
async function loadCurrentModel() {
  try {
    const model = await API.getCurrentModel();
    currentModel = model;

    if (!model.active) {
      currentModelInfo.innerHTML = `
        <div class="model-info-item">
          <span class="model-info-label">Status:</span>
          <span class="model-info-value">No active model</span>
        </div>
      `;
      return;
    }

    currentModelInfo.innerHTML = `
      <div class="model-info-item">
        <span class="model-info-label">Status:</span>
        <span class="model-info-value">Active</span>
      </div>
      <div class="model-info-item">
        <span class="model-info-label">Type:</span>
        <span class="model-info-value">${model.type.toUpperCase()}</span>
      </div>
      <div class="model-info-item">
        <span class="model-info-label">Classes:</span>
        <span class="model-info-value">${model.num_classes || "—"}</span>
      </div>
      <div class="model-info-item">
        <span class="model-info-label">Path:</span>
        <span class="model-info-value" style="font-size: 10px; word-break: break-all;">${model.model_path || "—"}</span>
      </div>
    `;
  } catch (e) {
    console.error("[ML] Failed to load current model:", e);
    currentModelInfo.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
  }
}

async function loadModels() {
  try {
    const data = await API.getModels();
    const models = data.models || [];

    if (models.length === 0) {
      modelList.innerHTML = '<div class="empty-state">No models found</div>';
      return;
    }

    modelList.innerHTML = models
      .map((model) => {
        const isActive = currentModel?.model_path?.includes(model.name);
        const sizeMB = (model.size / 1024 / 1024).toFixed(2);
        const modified = new Date(model.modified * 1000).toLocaleString();
        const isPrimary = model.name === "rf_classifier";

        return `
          <div class="model-item ${isActive ? "active" : ""}">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 4px;">
              <div>
                <div style="font-weight: 600; color: var(--accent-green);">
                  ${model.name} ${isActive ? "(Active)" : ""}
                </div>
                <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">
                  Type: ${model.type.toUpperCase()} | Size: ${sizeMB} MB
                </div>
                <div style="font-size: 10px; color: var(--text-muted);">
                  Modified: ${modified}
                </div>
              </div>
              ${!isPrimary ? `<button class="activate-model-btn" data-model-path="${model.path}" style="padding: 4px 8px; font-size: 10px; background: transparent; border: 1px solid var(--accent-green-dim); color: var(--accent-green); border-radius: 4px; cursor: pointer; white-space: nowrap;">Activate</button>` : ""}
            </div>
          </div>
        `;
      })
      .join("");
    
    // Add event listeners to activate buttons
    document.querySelectorAll(".activate-model-btn").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const modelPath = e.target.getAttribute("data-model-path");
        await activateModel(modelPath);
      });
    });
  } catch (e) {
    console.error("[ML] Failed to load models:", e);
    modelList.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
  }
}

async function exportModel() {
  try {
    btnExportModel.disabled = true;
    btnExportModel.textContent = "Exporting...";
    await API.exportModel();
    btnExportModel.textContent = "Export Current Model";
    alert("Model exported successfully!");
  } catch (e) {
    console.error("[ML] Failed to export model:", e);
    alert(`Failed to export model: ${e.message}`);
    btnExportModel.textContent = "Export Current Model";
  } finally {
    btnExportModel.disabled = false;
  }
}

async function importModel(file) {
  if (!file) return;

  try {
    const result = await API.importModel(file);
    alert(`Model imported successfully!\n\n${result.message}`);
    await loadModels();
    await loadCurrentModel();
  } catch (e) {
    console.error("[ML] Failed to import model:", e);
    alert(`Failed to import model: ${e.message}`);
  }
}

async function testModel() {
  if (selectedCaptures.size !== 1) return;

  const captureDir = Array.from(selectedCaptures)[0];
  try {
    btnTestModel.disabled = true;
    btnTestModel.textContent = "Testing...";
    const result = await API.testModel(captureDir);
    alert(
      `Test Results:\n\nLabel: ${result.classification.label}\nConfidence: ${(result.classification.confidence * 100).toFixed(1)}%`
    );
    btnTestModel.textContent = "Test Model on Selected";
  } catch (e) {
    console.error("[ML] Failed to test model:", e);
    alert(`Failed to test model: ${e.message}`);
    btnTestModel.textContent = "Test Model on Selected";
  } finally {
    btnTestModel.disabled = false;
    updateBatchButtons();
  }
}

// ------------------------------
// STATISTICS
// ------------------------------
async function loadStats() {
  try {
    const stats = await API.getStats();
    const labels = stats.label_distribution || {};

    mlStats.innerHTML = `
      <div class="stat-card">
        <div class="stat-label">Total Captures</div>
        <div class="stat-value">${stats.total_captures || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Labeled</div>
        <div class="stat-value">${stats.labeled_count || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Unlabeled</div>
        <div class="stat-value">${stats.unlabeled_count || 0}</div>
      </div>
      <div class="stat-card" style="grid-column: 1 / -1;">
        <div class="stat-label">Label Distribution</div>
        <div class="label-distribution">
          ${Object.entries(labels)
            .sort((a, b) => b[1] - a[1])
            .map(
              ([label, count]) => `
            <div class="label-item">
              <span class="label-name">${classLabels[label] || label}</span>
              <span class="label-count">${count}</span>
            </div>
          `
            )
            .join("")}
          ${Object.keys(labels).length === 0 ? '<div class="empty-state" style="padding: 20px;">No labels yet</div>' : ""}
        </div>
      </div>
    `;
  } catch (e) {
    console.error("[ML] Failed to load stats:", e);
    mlStats.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
  }
}

// ------------------------------
// CLASS LABELS
// ------------------------------
async function loadClassLabels() {
  try {
    const data = await API.getClassLabels();
    const mapping = data.class_mapping || {};

    // Build label lookup
    classLabels = {};
    for (const classInfo of Object.values(mapping)) {
      classLabels[classInfo.id] = classInfo.name || classInfo.id;
    }

    // Populate filter dropdown
    filterLabel.innerHTML = '<option value="">All Labels</option>';
    for (const [id, name] of Object.entries(classLabels)) {
      filterLabel.innerHTML += `<option value="${id}">${name}</option>`;
    }
    
    // Populate train label dropdown
    trainLabel.innerHTML = '<option value="">Select Label</option>';
    for (const [id, name] of Object.entries(classLabels)) {
      trainLabel.innerHTML += `<option value="${id}">${name}</option>`;
    }
  } catch (e) {
    console.error("[ML] Failed to load class labels:", e);
  }
}

// ------------------------------
// EVENT HANDLERS
// ------------------------------
btnRefreshCaptures.addEventListener("click", loadCaptures);
filterSearch.addEventListener("input", applyFilters);
filterLabel.addEventListener("change", applyFilters);
filterSource.addEventListener("change", applyFilters);

btnBatchLabel.addEventListener("click", batchUpdateLabels);
btnExportSelected.addEventListener("click", async () => {
  if (selectedCaptures.size === 0) return;
  
  // For now, just show info - full export can be added later
  alert(`Export functionality for ${selectedCaptures.size} captures coming soon!\n\nThis will export selected captures with their labels for training.`);
});
btnDeleteSelected.addEventListener("click", async () => {
  if (selectedCaptures.size === 0) return;
  
  if (!confirm(`Delete ${selectedCaptures.size} selected captures? This cannot be undone.`)) {
    return;
  }

  try {
    const response = await fetch("/api/ml/captures/batch-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ captures: Array.from(selectedCaptures) }),
    });
    
    if (!response.ok) throw new Error("Failed to delete captures");
    
    const result = await response.json();
    alert(`Deleted ${result.deleted} captures. ${result.errors > 0 ? `${result.errors} errors.` : ""}`);
    selectedCaptures.clear();
    await loadCaptures();
    await loadStats();
  } catch (e) {
    console.error("[ML] Failed to delete captures:", e);
    alert(`Failed to delete captures: ${e.message}`);
  }
});

btnExportModel.addEventListener("click", exportModel);
fileImportModel.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    importModel(e.target.files[0]);
    e.target.value = ""; // Reset input
  }
});
btnTestModel.addEventListener("click", testModel);

btnQuickTrain.addEventListener("click", startQuickTrain);
btnCancelTrain.addEventListener("click", cancelTraining);
trainLabel.addEventListener("change", updateBatchButtons);

captureGrid.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;

  if (!target.classList.contains("capture-thumb")) return;

  const imageUrl = target.getAttribute("src");
  if (!imageUrl) return;

  const captureCard = target.closest(".capture-card");
  const captureDir = captureCard?.getAttribute("data-capture-dir") || "Spectrogram preview";
  openImagePreview(imageUrl, captureDir);
});

if (imagePreviewClose) {
  imagePreviewClose.addEventListener("click", closeImagePreview);
}

if (imagePreviewModal) {
  imagePreviewModal.addEventListener("click", (event) => {
    if (event.target === imagePreviewModal) {
      closeImagePreview();
    }
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && imagePreviewModal?.classList.contains("open")) {
    closeImagePreview();
  }
});

// ------------------------------
// QUICK TRAINING
// ------------------------------
async function startQuickTrain() {
  if (selectedCaptures.size < 2 || selectedCaptures.size > 12) {
    alert("Please select 2–12 labeled captures for quick training (one is used for validation when possible).");
    return;
  }

  const label = trainLabel.value;
  if (!label) {
    alert("Please select a label for training");
    return;
  }

  const epochs = parseInt(trainEpochs.value) || 15;
  if (epochs < 5 || epochs > 30) {
    alert("Epochs must be between 5 and 30");
    return;
  }

  try {
    // Disable controls
    btnQuickTrain.disabled = true;
    trainLabel.disabled = true;
    trainEpochs.disabled = true;
    btnCancelTrain.disabled = false;
    btnCancelTrain.style.display = "block";
    
    // Show progress
    trainProgress.classList.add("active");
    trainProgressFill.style.width = "0%";
    trainProgressFill.textContent = "0%";
    trainStatus.textContent = "Starting training...";

    // Start training
    const result = await API.quickTrain(selectedCaptures, label, epochs);
    trainingJobId = result.job_id;

    // Start polling for status
    startTrainingPoll();
  } catch (e) {
    console.error("[ML] Failed to start training:", e);
    alert(`Failed to start training: ${e.message}`);
    resetTrainingUI();
  }
}

function startTrainingPoll() {
  if (trainingPollInterval) {
    clearInterval(trainingPollInterval);
  }

  trainingPollInterval = setInterval(async () => {
    if (!trainingJobId) {
      clearInterval(trainingPollInterval);
      return;
    }

    try {
      const status = await API.getTrainingStatus(trainingJobId);

      // Update progress
      const progress = status.progress || 0;
      const percent = Math.round(progress * 100);
      trainProgressFill.style.width = `${percent}%`;
      trainProgressFill.textContent = `${percent}%`;

      // Update status text
      let statusText = "";
      if (status.status === "running") {
        statusText = `Epoch ${status.epoch || 0}/${status.total_epochs || 0}`;
        if (status.loss !== null) {
          statusText += ` | Loss: ${status.loss.toFixed(4)}`;
        }
        if (status.accuracy !== null) {
          statusText += ` | Acc: ${status.accuracy.toFixed(1)}%`;
        }
      } else if (status.status === "completed") {
        statusText = "Training completed!";
        trainProgressFill.style.width = "100%";
        trainProgressFill.textContent = "100%";
        clearInterval(trainingPollInterval);
        trainingPollInterval = null;
        
        // Show success message
        setTimeout(() => {
          const ae = status.activation_eligible === true;
          const actHint = ae
            ? "Validation passed — you can Activate then POST /api/ml/models/reload (or restart) to load weights."
            : "Validation did not pass default threshold — use Activate with allow_unvalidated if you accept the risk.";
          alert(
            `Training completed.\n\nModel: ${status.output_path}\nactivation_eligible: ${ae}\n\n${actHint}`
          );
          resetTrainingUI();
          // Reload models list
          loadModels();
        }, 500);
      } else if (status.status === "failed") {
        statusText = `Training failed: ${status.error || "Unknown error"}`;
        clearInterval(trainingPollInterval);
        trainingPollInterval = null;
        alert(`Training failed: ${status.error || "Unknown error"}`);
        resetTrainingUI();
      } else if (status.status === "cancelled") {
        statusText = "Training cancelled";
        clearInterval(trainingPollInterval);
        trainingPollInterval = null;
        resetTrainingUI();
      }

      trainStatus.textContent = statusText;
    } catch (e) {
      console.error("[ML] Failed to poll training status:", e);
      // Continue polling even on error
    }
  }, 2000); // Poll every 2 seconds
}

async function cancelTraining() {
  if (!trainingJobId) return;

  if (!confirm("Cancel training? This cannot be undone.")) {
    return;
  }

  try {
    await API.cancelTraining(trainingJobId);
    if (trainingPollInterval) {
      clearInterval(trainingPollInterval);
      trainingPollInterval = null;
    }
    trainingJobId = null;
    resetTrainingUI();
    alert("Training cancelled");
  } catch (e) {
    console.error("[ML] Failed to cancel training:", e);
    alert(`Failed to cancel training: ${e.message}`);
  }
}

function resetTrainingUI() {
  trainingJobId = null;
  btnQuickTrain.disabled = false;
  trainLabel.disabled = false;
  trainEpochs.disabled = false;
  btnCancelTrain.disabled = true;
  btnCancelTrain.style.display = "none";
  trainProgress.classList.remove("active");
  updateBatchButtons();
}

// ------------------------------
// ACTIVATE MODEL
// ------------------------------
async function activateModel(modelPath) {
  if (!confirm(`Activate this model?\n\nThis will:\n- Backup current model\n- Copy this model to rf_classifier.pth\n- Require application restart to take effect\n\nModel: ${modelPath}`)) {
    return;
  }

  try {
    const result = await API.activateModel(modelPath);
    alert(`Model activated successfully!\n\n${result.message}\n\nBackup saved to: ${result.backup_path || "N/A"}\n\nPlease restart the application to use the new model.`);
    // Reload model info and list
    await loadCurrentModel();
    await loadModels();
  } catch (e) {
    console.error("[ML] Failed to activate model:", e);
    alert(`Failed to activate model: ${e.message}`);
  }
}

// Make functions global for inline handlers
window.toggleCapture = toggleCapture;
window.updateCaptureLabel = updateCaptureLabel;

// ------------------------------
// INITIALIZATION
// ------------------------------
async function init() {
  console.log("[ML Dashboard] Initializing...");
  
  // Load class labels first (needed for filters and labels)
  await loadClassLabels();
  
  // Load all data
  await Promise.all([
    loadCaptures(),
    loadCurrentModel(),
    loadModels(),
    loadStats(),
  ]);

  // Auto-refresh stats every 30 seconds
  setInterval(loadStats, 30000);
  
  console.log("[ML Dashboard] Initialized");
}

// Start when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
