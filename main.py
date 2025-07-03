import os
import json
import uuid
import shutil
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from src.model_inference import (
    run_brain_tumor,
    run_pneumonia,
    run_skin_disease,
    run_vascular,
    run_bone_fracture
)
from src.db_connection import initialize_db, save_inference_result,test_db_connection,fetch_all_records

# Initialize database at startup
initialize_db()
test_db_connection()  #

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="DeepMediTech: AI-Driven Imaging for Enhanced Healthcare")

# Directories setup
UPLOAD_DIR = Path("static/uploads")
RESULTS_DIR = Path("static/results")
TEMPLATES_DIR = Path("templates")
DATA_DIR = Path("data")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load medical class information from JSON
CLASS_INFO_FILE = Path("static/class_info.json")
CLASS_INFO = {}

if CLASS_INFO_FILE.exists():
    try:
        with CLASS_INFO_FILE.open("r", encoding="utf-8") as f:
            CLASS_INFO = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse class_info.json: {e}")
    except Exception as e:
        logging.error(f"Error loading class_info.json: {e}")
else:
    logging.warning("class_info.json file not found!")

# Mount static files directory
app.mount("/static", StaticFiles(directory=UPLOAD_DIR.parent), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Available tasks
TASKS = {
    "brain_tumor": {
        "name": "Brain Tumor Boundary Extraction",
        "function": run_brain_tumor,
        "description": "Detects and delineates brain tumor boundaries in MRI images"
    },
    "pneumonia": {
        "name": "Pneumonia Segmentation",
        "function": run_pneumonia,
        "description": "Segments lung areas in X-ray images to detect pneumonia"
    },
    "skin_disease": {
        "name": "Skin Disease Detection",
        "function": run_skin_disease,
        "description": "Identifies and classifies skin lesions and abnormalities"
    },
    "vascular": {
        "name": "Vascular Segmentation",
        "function": run_vascular,
        "description": "Segments blood vessels in medical images"
    },
    "bone_fracture": {
        "name": "X-ray for Bone Fracture Detection",
        "function": run_bone_fracture,
        "description": "Segments bones in X-ray images to detect fractures"
    }
}

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    input_image: str = None,
    output_image: str = None,
    error: str = None,
    predictions: list = None
):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tasks": TASKS,
            "input_image": input_image,
            "output_image": output_image,
            "error": error,
            "predictions": predictions or []
        }
    )
@app.post("/analyze")
async def analyze(
    request: Request,
    task: str = Form(...),
    file: UploadFile = File(...)
):
    if task not in TASKS:
        raise HTTPException(status_code=400, detail="Invalid task selected")

    # Generate unique filenames
    input_filename = f"{uuid.uuid4()}{Path(file.filename).suffix}"
    input_path = UPLOAD_DIR / input_filename
    output_dir = RESULTS_DIR / str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file securely
    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logging.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Error saving file")

    # Run the selected model
    try:
        result = TASKS[task]["function"](str(input_path), str(output_dir))
        output_path = result.get("output_image")
        raw_predictions = result.get("predictions", [])

        if not output_path:
            raise HTTPException(status_code=500, detail="No output image was generated")

        # Process predictions
        processed_predictions = []
        seen_classes = set()

        for pred in raw_predictions:
            pred_lower = pred.lower()
            if pred_lower in seen_classes:
                continue  # Skip duplicate classes
            seen_classes.add(pred_lower)

            info = CLASS_INFO.get(pred_lower, {
                "description": "No description available",
                "prescription": "No prescription available",
                "suggestion": "No suggestions available"
            })
            processed_predictions.append({
                "class": pred_lower,
                "description": info["description"],
                "prescription": info["prescription"],
                "suggestion": info["suggestion"]
            })

        logging.info(f"Saving result to database: {task}, {input_path}, {output_path}, {processed_predictions}")

        save_inference_result(
            task,
            str(input_path),
            output_path,
            json.dumps(processed_predictions)
        )

        logging.info("Database save function executed successfully")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "tasks": TASKS,
                "input_image": str(input_path),
                "output_image": output_path,
                "selected_task": task,
                "predictions": processed_predictions
            }
        )
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.get("/get_all_records")
async def get_all_records():
    """API endpoint to retrieve all inference records."""
    records = fetch_all_records()
    
    if not records:
        raise HTTPException(status_code=404, detail="No records found")
    
    return {"data": records}

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)