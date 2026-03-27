from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import json
import zipfile
import shutil
import time
import io
import subprocess
import asyncio
from typing import List, Optional, Dict, Any

router = APIRouter(prefix="/api/ml", tags=["ml"])

# Base paths
BASE_DIR = Path(__file__).resolve().parents[3]
CAPTURES_DIR = BASE_DIR / "data" / "artifacts" / "captures"
MODELS_DIR = BASE_DIR / "spear_edge" / "ml" / "models"
CLASS_LABELS_PATH = MODELS_DIR / "class_labels.json"


@router.get("/captures")
async def list_captures(
    limit: int = 100,
    offset: int = 0,
    label: Optional[str] = None,
    min_confidence: Optional[float] = None
):
    """List captures with optional filtering."""
    if not CAPTURES_DIR.exists():
        return {"captures": [], "total": 0}
    
    captures = []
    capture_dirs = sorted([d for d in CAPTURES_DIR.iterdir() if d.is_dir()], reverse=True)
    
    for cap_dir in capture_dirs:
        json_path = cap_dir / "capture.json"
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Apply filters
            classification = data.get("classification", {})
            capture_label = classification.get("label")
            confidence = classification.get("confidence", 0.0)
            
            if label and capture_label != label:
                continue
            if min_confidence is not None and confidence < min_confidence:
                continue
            
            # Get thumbnail path
            thumb_path = cap_dir / "thumbnails" / "spectrogram.png"
            thumbnail_url = f"/api/ml/captures/{cap_dir.name}/thumbnail" if thumb_path.exists() else None
            
            capture_info = {
                "capture_dir": cap_dir.name,
                "timestamp": data.get("timing", {}).get("capture_timestamp", 0),
                "freq_hz": data.get("rf_configuration", {}).get("center_freq_hz", 0),
                "sample_rate_sps": data.get("rf_configuration", {}).get("sample_rate_sps", 0),
                "duration_s": data.get("timing", {}).get("duration_s", 0),
                "label": capture_label,
                "confidence": confidence,
                "source": data.get("request_provenance", {}).get("reason", "unknown"),
                "thumbnail_url": thumbnail_url,
                "classification": classification,
            }
            captures.append(capture_info)
        except Exception as e:
            print(f"[ML API] Error reading capture {cap_dir.name}: {e}")
            continue
    
    total = len(captures)
    captures = captures[offset:offset + limit]
    
    return {
        "captures": captures,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/captures/{capture_dir}")
async def get_capture(capture_dir: str):
    """Get detailed capture information."""
    cap_dir = CAPTURES_DIR / capture_dir
    if not cap_dir.exists():
        raise HTTPException(status_code=404, detail="Capture not found")
    
    json_path = cap_dir / "capture.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="capture.json not found")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data


@router.get("/captures/{capture_dir}/thumbnail")
async def get_capture_thumbnail(capture_dir: str):
    """Get capture thumbnail image."""
    cap_dir = CAPTURES_DIR / capture_dir
    thumb_path = cap_dir / "thumbnails" / "spectrogram.png"
    
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(thumb_path, media_type="image/png")


@router.post("/captures/{capture_dir}/label")
async def update_capture_label(capture_dir: str, request: Request):
    """Update capture classification label."""
    body = await request.json()
    label = body.get("label")
    
    if not label:
        raise HTTPException(status_code=400, detail="label is required")
    
    cap_dir = CAPTURES_DIR / capture_dir
    if not cap_dir.exists():
        raise HTTPException(status_code=404, detail="Capture not found")
    
    json_path = cap_dir / "capture.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="capture.json not found")
    
    # Validate label exists in class_labels.json
    if CLASS_LABELS_PATH.exists():
        with open(CLASS_LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        
        valid_labels = []
        for class_info in class_labels.get("class_mapping", {}).values():
            valid_labels.append(class_info.get("id"))
        
        if label not in valid_labels:
            raise HTTPException(status_code=400, detail=f"Invalid label: {label}. Valid labels: {valid_labels}")
    
    # Update capture.json
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if "classification" not in data:
        data["classification"] = {}
    
    old_label = data["classification"].get("label", "none")
    data["classification"]["label"] = label
    data["classification"]["confidence"] = 1.0  # Manual label = 100% confidence
    data["classification"]["model"] = "manual_label"
    data["classification"]["labeled_at"] = time.time()
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Also update dataset_raw copy if it exists
    dataset_raw_dir = BASE_DIR / "data" / "dataset_raw" / capture_dir
    if dataset_raw_dir.exists():
        dataset_json = dataset_raw_dir / "capture.json"
        if dataset_json.exists():
            with open(dataset_json, 'r') as f:
                dataset_data = json.load(f)
            dataset_data["classification"] = data["classification"]
            with open(dataset_json, 'w') as f:
                json.dump(dataset_data, f, indent=2)
    
    return {"ok": True, "label": label, "capture_dir": capture_dir, "old_label": old_label}


@router.post("/captures/batch-label")
async def batch_update_labels(request: Request):
    """Batch update labels for multiple captures."""
    body = await request.json()
    capture_dirs = body.get("captures", [])
    label = body.get("label")
    
    if not label:
        raise HTTPException(status_code=400, detail="label is required")
    
    if not capture_dirs:
        raise HTTPException(status_code=400, detail="captures list is required")
    
    results = []
    errors = []
    
    for capture_dir in capture_dirs:
        try:
            cap_dir = CAPTURES_DIR / capture_dir
            json_path = cap_dir / "capture.json"
            
            if not json_path.exists():
                errors.append({"capture_dir": capture_dir, "error": "capture.json not found"})
                continue
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if "classification" not in data:
                data["classification"] = {}
            
            data["classification"]["label"] = label
            data["classification"]["confidence"] = 1.0
            data["classification"]["model"] = "manual_label"
            data["classification"]["labeled_at"] = time.time()
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            results.append({"capture_dir": capture_dir, "ok": True})
        except Exception as e:
            errors.append({"capture_dir": capture_dir, "error": str(e)})
    
    return {
        "ok": len(errors) == 0,
        "updated": len(results),
        "errors": len(errors),
        "results": results,
        "error_details": errors
    }


@router.get("/models")
async def list_models():
    """List all available models."""
    if not MODELS_DIR.exists():
        return {"models": []}
    
    models = []
    
    # Find PyTorch models
    for pth_file in MODELS_DIR.glob("*.pth"):
        models.append({
            "path": str(pth_file.relative_to(BASE_DIR)),
            "name": pth_file.stem,
            "type": "pytorch",
            "size": pth_file.stat().st_size,
            "modified": pth_file.stat().st_mtime
        })
    
    # Find ONNX models
    for onnx_file in MODELS_DIR.glob("*.onnx"):
        models.append({
            "path": str(onnx_file.relative_to(BASE_DIR)),
            "name": onnx_file.stem,
            "type": "onnx",
            "size": onnx_file.stat().st_size,
            "modified": onnx_file.stat().st_mtime
        })
    
    # Sort by modified time (newest first)
    models.sort(key=lambda x: x["modified"], reverse=True)
    
    return {"models": models}


@router.get("/models/current")
async def get_current_model(request: Request):
    """Get information about currently active model."""
    orch = request.app.state.orchestrator
    capture_mgr = orch.capture_mgr
    
    classifier = capture_mgr.classifier
    if classifier is None:
        return {
            "active": False,
            "type": "none",
            "model_path": None
        }
    
    # Determine model type and path
    model_type = "unknown"
    model_path = None
    
    # Check for model_path attribute (now stored in classifiers)
    if hasattr(classifier, "model_path") and classifier.model_path:
        model_path = classifier.model_path
    
    # Determine type from path if not already set
    if model_path:
        model_path_obj = Path(model_path)
        if model_path_obj.exists():
            if model_path_obj.suffix == ".pth":
                model_type = "pytorch"
            elif model_path_obj.suffix == ".onnx":
                model_type = "onnx"
            # Convert to relative path for display
            try:
                model_path = str(model_path_obj.relative_to(BASE_DIR))
            except ValueError:
                # If not relative, use absolute
                model_path = str(model_path_obj)
    
    # Try to get class count
    num_classes = None
    if hasattr(classifier, "num_classes"):
        num_classes = classifier.num_classes
    
    return {
        "active": True,
        "type": model_type,
        "model_path": model_path,
        "num_classes": num_classes
    }


@router.post("/models/export")
async def export_model(request: Request):
    """Export current model as zip file."""
    orch = request.app.state.orchestrator
    capture_mgr = orch.capture_mgr
    
    classifier = capture_mgr.classifier
    if classifier is None:
        raise HTTPException(status_code=400, detail="No active model to export")
    
    # Get model path
    model_path = None
    if hasattr(classifier, "model_path"):
        model_path = classifier.model_path
    
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    model_path = Path(model_path)
    
    # Create zip in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add model file
        zip_file.write(model_path, model_path.name)
        
        # Add class_labels.json
        if CLASS_LABELS_PATH.exists():
            zip_file.write(CLASS_LABELS_PATH, "class_labels.json")
        
        # Create metadata
        metadata = {
            "export_timestamp": time.time(),
            "model_path": str(model_path),
            "model_name": model_path.stem,
            "model_type": "pytorch" if model_path.suffix == ".pth" else "onnx",
            "num_classes": getattr(classifier, "num_classes", None),
        }
        
        zip_file.writestr("model_metadata.json", json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    
    # Return as download
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filename = f"spear_edge_model_{model_path.stem}_{timestamp}.zip"
    
    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/models/import")
async def import_model(file: UploadFile = File(...)):
    """Import new model from zip file."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a zip archive")
    
    # Read zip file
    zip_data = await file.read()
    zip_buffer = io.BytesIO(zip_data)
    
    # Extract to temporary directory
    temp_dir = BASE_DIR / "data" / "temp_import"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            zip_file.extractall(temp_dir)
        
        # Find model file and class_labels.json
        model_file = None
        class_labels_file = None
        
        for item in temp_dir.rglob("*"):
            if item.is_file():
                if item.suffix in [".pth", ".onnx"]:
                    model_file = item
                elif item.name == "class_labels.json":
                    class_labels_file = item
        
        if not model_file:
            raise HTTPException(status_code=400, detail="No model file (.pth or .onnx) found in zip")
        
        # Copy model to models directory
        dest_model = MODELS_DIR / model_file.name
        shutil.copy2(model_file, dest_model)
        
        # Copy class_labels.json if present
        if class_labels_file:
            dest_labels = MODELS_DIR / "class_labels.json"
            shutil.copy2(class_labels_file, dest_labels)
        
        return {
            "ok": True,
            "model_path": str(dest_model.relative_to(BASE_DIR)),
            "model_name": dest_model.stem,
            "message": "Model imported successfully. Restart application to activate."
        }
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@router.post("/models/test")
async def test_model(request: Request):
    """Test model on a specific capture."""
    body = await request.json()
    capture_dir = body.get("capture_dir")
    
    if not capture_dir:
        raise HTTPException(status_code=400, detail="capture_dir is required")
    
    orch = request.app.state.orchestrator
    capture_mgr = orch.capture_mgr
    
    # Get capture
    cap_dir = CAPTURES_DIR / capture_dir
    if not cap_dir.exists():
        raise HTTPException(status_code=404, detail="Capture not found")
    
    # Find spectrogram
    spec_path = cap_dir / "features" / "spectrogram.npy"
    if not spec_path.exists():
        raise HTTPException(status_code=404, detail="Spectrogram not found")
    
    # Load spectrogram
    import numpy as np
    spec = np.load(spec_path)
    
    # Run classification
    classifier = capture_mgr.classifier
    if classifier is None:
        raise HTTPException(status_code=400, detail="No classifier available")
    
    try:
        result = classifier.classify(spec)
        return {
            "ok": True,
            "capture_dir": capture_dir,
            "classification": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/class-labels")
async def get_class_labels():
    """Get class labels mapping."""
    if not CLASS_LABELS_PATH.exists():
        return {"error": "class_labels.json not found"}
    
    with open(CLASS_LABELS_PATH, 'r') as f:
        data = json.load(f)
    
    return data


@router.post("/captures/{capture_dir}/delete")
async def delete_capture(capture_dir: str):
    """Delete a capture directory."""
    cap_dir = CAPTURES_DIR / capture_dir
    if not cap_dir.exists():
        raise HTTPException(status_code=404, detail="Capture not found")
    
    try:
        shutil.rmtree(cap_dir)
        return {"ok": True, "capture_dir": capture_dir, "message": "Capture deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete capture: {str(e)}")


@router.post("/captures/batch-delete")
async def batch_delete_captures(request: Request):
    """Batch delete multiple captures."""
    body = await request.json()
    capture_dirs = body.get("captures", [])
    
    if not capture_dirs:
        raise HTTPException(status_code=400, detail="captures list is required")
    
    results = []
    errors = []
    
    for capture_dir in capture_dirs:
        try:
            cap_dir = CAPTURES_DIR / capture_dir
            if cap_dir.exists():
                shutil.rmtree(cap_dir)
                results.append({"capture_dir": capture_dir, "ok": True})
            else:
                errors.append({"capture_dir": capture_dir, "error": "Not found"})
        except Exception as e:
            errors.append({"capture_dir": capture_dir, "error": str(e)})
    
    return {
        "ok": len(errors) == 0,
        "deleted": len(results),
        "errors": len(errors),
        "results": results,
        "error_details": errors
    }


@router.get("/stats")
async def get_ml_stats():
    """Get ML statistics."""
    if not CAPTURES_DIR.exists():
        return {
            "total_captures": 0,
            "label_distribution": {},
            "unlabeled_count": 0
        }
    
    label_counts = {}
    total = 0
    unlabeled = 0
    
    for cap_dir in CAPTURES_DIR.iterdir():
        if not cap_dir.is_dir():
            continue
        
        json_path = cap_dir / "capture.json"
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            total += 1
            classification = data.get("classification", {})
            label = classification.get("label")
            
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
            else:
                unlabeled += 1
        except Exception:
            continue
    
    return {
        "total_captures": total,
        "labeled_count": total - unlabeled,
        "unlabeled_count": unlabeled,
        "label_distribution": label_counts
    }


# Training job storage (in-memory, simple implementation)
_training_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/train/quick")
async def quick_train(request: Request):
    """Quick fine-tune on 1-2 captures."""
    body = await request.json()
    captures = body.get("captures", [])
    label = body.get("label")
    epochs = body.get("epochs", 15)
    batch_size = body.get("batch_size", 2)
    learning_rate = body.get("learning_rate", 1e-4)
    
    if len(captures) < 1 or len(captures) > 2:
        raise HTTPException(status_code=400, detail="Must select 1-2 captures")
    
    if not label:
        raise HTTPException(status_code=400, detail="Label is required")
    
    # Get orchestrator and current model
    orch = request.app.state.orchestrator
    capture_mgr = orch.capture_mgr
    
    classifier = capture_mgr.classifier
    if classifier is None or not hasattr(classifier, "model_path") or not classifier.model_path:
        raise HTTPException(status_code=400, detail="No active PyTorch model to fine-tune")
    
    model_path = Path(classifier.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    
    # Validate label exists in class_labels.json
    if not CLASS_LABELS_PATH.exists():
        raise HTTPException(status_code=400, detail="class_labels.json not found")
    
    with open(CLASS_LABELS_PATH, 'r') as f:
        class_labels = json.load(f)
    
    class_to_index = class_labels.get("class_to_index", {})
    if label not in class_to_index:
        raise HTTPException(status_code=400, detail=f"Label '{label}' not found in class_labels.json")
    
    # Create job ID
    job_id = f"train_{int(time.time())}"
    
    # Create temporary dataset directory
    temp_dataset_dir = BASE_DIR / "data" / "temp_training" / job_id
    temp_dataset_dir.mkdir(parents=True, exist_ok=True)
    train_dir = temp_dataset_dir / "train" / label
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract spectrograms from captures
    copied = 0
    for capture_dir in captures:
        cap_dir = CAPTURES_DIR / capture_dir
        spec_path = cap_dir / "features" / "spectrogram.npy"
        
        if not spec_path.exists():
            print(f"[QUICK TRAIN] Warning: Spectrogram not found for {capture_dir}")
            continue
        
        # Copy spectrogram to temp dataset
        dest_path = train_dir / f"{capture_dir}.npy"
        shutil.copy2(spec_path, dest_path)
        copied += 1
    
    if copied == 0:
        shutil.rmtree(temp_dataset_dir)
        raise HTTPException(status_code=400, detail="No valid spectrograms found in selected captures")
    
    # Create output model path
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_model_path = MODELS_DIR / f"rf_classifier_quicktrain_{timestamp}.pth"
    
    # Initialize job status
    _training_jobs[job_id] = {
        "status": "starting",
        "progress": 0.0,
        "epoch": 0,
        "total_epochs": epochs,
        "loss": None,
        "accuracy": None,
        "error": None,
        "output_path": str(output_model_path),
        "started_at": time.time()
    }
    
    # Start training in background
    asyncio.create_task(run_quick_train(
        job_id=job_id,
        model_path=model_path,
        dataset_dir=temp_dataset_dir,
        label=label,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_path=output_model_path,
        class_labels_path=CLASS_LABELS_PATH
    ))
    
    return {
        "ok": True,
        "job_id": job_id,
        "captures_processed": copied,
        "label": label,
        "epochs": epochs
    }


async def run_quick_train(
    job_id: str,
    model_path: Path,
    dataset_dir: Path,
    label: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_path: Path,
    class_labels_path: Path
):
    """Run fine-tuning in background."""
    try:
        _training_jobs[job_id]["status"] = "running"
        
        # Build command
        script_path = BASE_DIR / "scripts" / "fine_tune_new_class.py"
        
        cmd = [
            "python3",
            str(script_path),
            "--model-path", str(model_path),
            "--dataset-dir", str(dataset_dir),
            "--new-class-id", label,
            "--output-path", str(output_path),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--class-labels", str(class_labels_path),
            "--device", "cuda"
        ]
        
        # Note: We're using fine_tune_new_class.py even for existing classes
        # It will fine-tune the model on the new data for that class
        # The script handles both new and existing classes
        
        # Run training script
        # Set environment variables to suppress ONNX warnings
        import os
        env = os.environ.copy()
        env["ORT_LOGGING_LEVEL"] = "3"  # Suppress ONNX warnings (0=verbose, 3=errors only)
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(BASE_DIR),
            env=env
        )
        
        # Monitor output (simple - could be improved with regex parsing)
        stdout_lines = []
        stderr_lines = []
        
        # Read stdout and stderr concurrently
        async def read_stream(stream, is_stderr=False):
            lines = []
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='ignore').strip()
                # Filter out ONNX warnings and ANSI codes
                if line_str and not any(skip in line_str for skip in [
                    '[W:onnxruntime',
                    'device_discovery',
                    'ReadFileContents',
                    '/sys/class/drm',
                    '[0;93m',  # ANSI color codes
                    '[m'  # ANSI reset
                ]):
                    lines.append(line_str)
            return lines
        
        # Read both streams in parallel
        stdout_task = asyncio.create_task(read_stream(process.stdout))
        stderr_task = asyncio.create_task(read_stream(process.stderr, is_stderr=True))
        
        stdout_lines = await stdout_task
        stderr_lines = await stderr_task
        
        # Process stdout lines for progress
        for line_str in stdout_lines:
            
            # Try to extract progress info (basic parsing)
            if "Epoch" in line_str and "/" in line_str:
                try:
                    # Parse "Epoch 5/15:" format
                    parts = line_str.split("Epoch")[1].split("/")
                    if len(parts) >= 2:
                        epoch = int(parts[0].strip().split(":")[0])
                        total = int(parts[1].split(":")[0])
                        _training_jobs[job_id]["epoch"] = epoch
                        _training_jobs[job_id]["progress"] = epoch / total
                except:
                    pass
            
            if "Train Acc:" in line_str:
                try:
                    # Parse "Train Acc: 85.50%" format
                    acc_str = line_str.split("Train Acc:")[1].split("%")[0].strip()
                    _training_jobs[job_id]["accuracy"] = float(acc_str)
                except:
                    pass
            
            if "Train Loss:" in line_str:
                try:
                    # Parse "Train Loss: 0.1234" format
                    loss_str = line_str.split("Train Loss:")[1].split(",")[0].strip()
                    _training_jobs[job_id]["loss"] = float(loss_str)
                except:
                    pass
        
        # Wait for process to complete
        return_code = await process.wait()
        
        if return_code == 0:
            _training_jobs[job_id]["status"] = "completed"
            _training_jobs[job_id]["progress"] = 1.0
        else:
            # Collect all error information
            error_parts = []
            
            # Add stderr (filtered)
            if stderr_lines:
                error_parts.extend(stderr_lines)
            
            # Add stdout errors (if any)
            for line in stdout_lines:
                if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                    error_parts.append(line)
            
            # Combine error message
            if error_parts:
                error_msg = '\n'.join(error_parts[-10:])  # Last 10 lines
            else:
                # If no error captured, try to get last stdout lines
                if stdout_lines:
                    error_msg = '\n'.join(stdout_lines[-5:])  # Last 5 lines
                else:
                    error_msg = f"Training failed with return code {return_code}. No error output captured."
            
            # Clean up error message (remove ANSI codes)
            import re
            error_msg = re.sub(r'\x1b\[[0-9;]*m', '', error_msg)  # Remove ANSI codes
            
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["error"] = error_msg[:1000]  # Increased limit for debugging
        
    except Exception as e:
        _training_jobs[job_id]["status"] = "failed"
        _training_jobs[job_id]["error"] = str(e)
    finally:
        # Cleanup temp dataset (keep for a bit in case user wants to inspect)
        # Could add cleanup task here if needed
        pass


@router.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return _training_jobs[job_id]


@router.post("/train/cancel/{job_id}")
async def cancel_training(job_id: str):
    """Cancel training job (placeholder - actual cancellation would need process management)."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if _training_jobs[job_id]["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Training job already finished")
    
    # Note: Actual cancellation would require process management
    # For now, just mark as cancelled
    _training_jobs[job_id]["status"] = "cancelled"
    
    return {"ok": True, "job_id": job_id, "message": "Training job marked for cancellation"}


@router.post("/models/activate")
async def activate_model(request: Request):
    """Activate a model by copying it to the primary model path."""
    body = await request.json()
    model_path = body.get("model_path")
    allow_unvalidated = bool(body.get("allow_unvalidated", False))
    
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
    
    # Resolve model path
    model_file = Path(model_path)
    if not model_file.is_absolute():
        # Assume relative to models directory
        model_file = MODELS_DIR / model_path
    
    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_file}")
    
    # Verify it's a .pth file
    if not model_file.suffix == ".pth":
        raise HTTPException(status_code=400, detail="Only .pth model files can be activated")

    # Require a passing validation artifact by default.
    # Operators may explicitly bypass with allow_unvalidated=true for emergency use.
    validation_path = model_file.with_suffix(".validation.json")
    validation_summary = {
        "path": str(validation_path),
        "exists": validation_path.exists(),
        "validated": False,
        "reason": "missing_validation_artifact",
    }
    if validation_path.exists():
        try:
            validation_data = json.loads(validation_path.read_text(encoding="utf-8"))
            validation_summary["validated"] = bool(validation_data.get("validated", False))
            validation_summary["reason"] = validation_data.get("reason", "unknown")
            metrics = validation_data.get("metrics", {})
            if isinstance(metrics, dict):
                validation_summary["val_accuracy_pct"] = metrics.get("val_accuracy_pct")
                validation_summary["macro_f1"] = metrics.get("macro_f1")
        except Exception as e:
            validation_summary["reason"] = f"invalid_validation_artifact: {e}"

    if not allow_unvalidated:
        if not validation_summary["exists"]:
            raise HTTPException(
                status_code=400,
                detail="Model activation blocked: missing .validation.json artifact. "
                       "Run evaluation first or set allow_unvalidated=true to override.",
            )
        if not validation_summary["validated"]:
            raise HTTPException(
                status_code=400,
                detail=f"Model activation blocked: validation did not pass ({validation_summary['reason']}). "
                       "Set allow_unvalidated=true to override.",
            )
    
    # Backup current model if it exists
    primary_model = MODELS_DIR / "rf_classifier.pth"
    if model_file.resolve() == primary_model.resolve():
        return {
            "ok": True,
            "message": "Model is already active.",
            "model_path": str(primary_model),
            "backup_path": None,
            "validation": validation_summary,
            "allow_unvalidated": allow_unvalidated,
            "no_op": True,
        }

    backup_path = None
    if primary_model.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        backup_path = MODELS_DIR / f"rf_classifier_backup_{timestamp}.pth"
        shutil.copy2(primary_model, backup_path)
        print(f"[ML API] Backed up current model to: {backup_path}")
    
    # Copy new model to primary location
    try:
        shutil.copy2(model_file, primary_model)
        print(f"[ML API] Activated model: {model_file} -> {primary_model}")
        
        return {
            "ok": True,
            "message": "Model activated successfully. Restart application to use new model.",
            "model_path": str(primary_model),
            "backup_path": str(backup_path) if backup_path else None,
            "validation": validation_summary,
            "allow_unvalidated": allow_unvalidated,
        }
    except Exception as e:
        # Restore backup if copy failed
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, primary_model)
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")
