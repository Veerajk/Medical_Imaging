# Import YOLO from ultralytics to load and train the model
from ultralytics import YOLO
import logging
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Configure logging for detailed tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants for configuration
MODEL_PATH = "yolo11x-bone.pt"
DATA_CONFIG_PATH = r"model_training\bone_fracture_detection\data\data.yaml"
EPOCHS = 150
IMG_SIZE = 640
DEVICE = 0  # Use GPU device 0 or 'cpu' if no GPU is available
OUTPUT_DIR = Path("bone_fracture_training_results")
MODEL_EXPORT_DIR = OUTPUT_DIR / "exported_model"

# Create directories for output if they don't exist
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {OUTPUT_DIR}")

if not MODEL_EXPORT_DIR.exists():
    MODEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created model export directory: {MODEL_EXPORT_DIR}")

# Load the YOLO model
logging.info(f"Attempting to load YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    logging.info(f"Model loaded successfully: {MODEL_PATH}")
except FileNotFoundError:
    logging.error(f"Model file not found: {MODEL_PATH}. Please verify the file path.")
    raise

# Log training parameters in a structured format
def log_training_config():
    training_params = {
        "Model Path": MODEL_PATH,
        "Dataset Config Path": DATA_CONFIG_PATH,
        "Epochs": EPOCHS,
        "Image Size": IMG_SIZE,
        "Device": "GPU" if DEVICE >= 0 else "CPU",
        "Output Directory": str(OUTPUT_DIR),
    }
    logging.info("Training Configuration:")
    logging.info(json.dumps(training_params, indent=4))

# Log configuration details
log_training_config()

# Record the start time for performance tracking
start_time = time.time()

# Train the YOLO model for bone fracture detection
logging.info("Starting training process...")
train_results = model.train(
    data=DATA_CONFIG_PATH,  # Path to dataset YAML
    epochs=EPOCHS,  # Number of epochs
    imgsz=IMG_SIZE,  # Image size
    device=DEVICE,  # GPU/CPU selection
    project=str(OUTPUT_DIR),  # Save training results in the output directory
    name="bone_fracture_model"  # Name for this training run
)

# Calculate and log training duration
training_duration = time.time() - start_time
logging.info(f"Training completed in {training_duration:.2f} seconds.")

# Display training results if available
def visualize_training_results(results_dir):
    metrics_image_path = results_dir / "results.png"
    if metrics_image_path.exists():
        logging.info(f"Displaying training metrics from: {metrics_image_path}")
        img = plt.imread(str(metrics_image_path))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        logging.warning(f"No training metrics image found at: {metrics_image_path}")

# Call the function to display results
visualize_training_results(OUTPUT_DIR / "bone_fracture_model")

# Export the trained model
exported_model_path = MODEL_EXPORT_DIR / "final_bone_fracture_model.pt"
logging.info("Exporting trained model...")
model.export(
    format="torchscript",  # Export format for deployment
    dynamic=True,  # Enable dynamic input dimensions
    simplify=True,  # Simplify model for faster inference
    imgsz=IMG_SIZE,  # Image size
    device=DEVICE  # Device for exporting
)
logging.info(f"Model exported successfully to: {exported_model_path}")

# Log final completion message
logging.info("Bone fracture detection model training and export process completed successfully.")