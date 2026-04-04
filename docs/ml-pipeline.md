# SPEAR-Edge ML pipeline (operator)

## Preprocessing contract

- **Schema:** `spear_ml_spec_v1` (see `spear_edge/ml/preprocess.py`).
- **Input tensor:** 512×512 `float32` spectrogram from `compute_spectrogram_chunked()` (median noise floor removed). Training and inference use the **same** normalization (no per-sample min–max in training).

## Artifacts

| Artifact | Role |
|----------|------|
| `spear_edge/ml/models/class_labels.json` | Label ontology (shipped) |
| `spear_edge/ml/models/rf_classifier.pth` | Primary weights (**not** shipped by default; train locally) |
| `rf_classifier.validation.json` | Pass/fail for `POST /api/ml/models/activate` |
| `rf_classifier.modelcard.json` | Dataset fingerprint, torch version, metrics |
| `rf_classifier.calibration.json` | Temperature scaling (`spear_edge.calibration.v1`); optional but recommended |

Large/binary files are typically **gitignored** (`.pth`, `.onnx`, sidecars under `ml/models/`).

### Calibration (temperature scaling)

- **Full train** and **fine-tune** fit a scalar **T** on the **validation** logits and write `*.calibration.json` next to the `.pth`.
- **Inference:** `PyTorchRfClassifier` loads `rf_classifier.calibration.json` if present and divides logits by **T** before softmax.
- **Retrofit** (already have `.pth` from before this feature):

```bash
./venv/bin/python scripts/fit_calibration.py \
  --checkpoint spear_edge/ml/models/rf_classifier.pth \
  --dataset-dir data/dataset
```

Then **reload** the classifier (`POST /api/ml/models/reload`) or restart.

### Uncertain predictions (optional)

Set **either** env var to flag low-trust outputs (`classification_result` includes `uncertain: true`):

- `SPEAR_ML_UNCERTAIN_MIN_CONFIDENCE` — e.g. `0.55` → uncertain if max calibrated prob is below this.
- `SPEAR_ML_UNCERTAIN_MIN_ENTROPY_NATS` — e.g. `1.5` → uncertain if predictive entropy (nats) is **above** this.

If unset, `uncertain` is normally `false` (backward compatible).

### ONNX export

```bash
./venv/bin/python scripts/export_rf_classifier_onnx.py \
  --checkpoint spear_edge/ml/models/rf_classifier.pth \
  --output spear_edge/ml/models/rf_classifier.onnx
```

ONNX outputs **logits**; apply the same preprocessing as PyTorch, then softmax (and optional **T** from calibration if you replicate in your runtime).

## End-to-end flow

1. **Capture** — IQ → `features/spectrogram.npy`; `capture.json` includes `ml_features.preprocess_schema`.
2. **Label** — Web UI or `POST /api/ml/captures/.../label`.
3. **Prepare dataset** — `scripts/prepare_training_dataset.py` (optional `--group-by capture` to reduce split leakage for SPEAR exports).
4. **Full train** — `scripts/train_rf_classifier.py` (writes `.pth` + `.validation.json` + `.modelcard.json`). Use `--no-gpu-batch-clamp` on desktop GPUs if you need large batches.
5. **Activate** — `POST /api/ml/models/activate` (requires passing validation unless `allow_unvalidated=true`).
6. **Reload (no restart)** — `POST /api/ml/models/reload` after activate, unless `SPEAR_ML_ALLOW_HOT_RELOAD=0`.

## Quick train

- API: `POST /api/ml/train/quick` — requires **2–12** captures; holds one capture out for validation when possible.
- Needs an existing PyTorch checkpoint path on the capture manager (train or `make_dummy_pytorch_model.py` for dev).

## Dev: dummy weights

```bash
./venv/bin/python spear_edge/ml/models/make_dummy_pytorch_model.py
```

Produces `rf_classifier_dummy.pth` so the stack loads a network without a full train.
