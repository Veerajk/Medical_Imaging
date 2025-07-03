from ultralytics import YOLO

# Load the trained YOLO model
#model_path = "src/models/brain_tumor_segmention_v0.1.pt"
#model_path = "src/models/bone_fracture_detection.pt"
#model_path = "src/models/blood_vessels_segmentation.pt"
#model_path = "src/models/pneumonia_segmentation.pt"
model_path = "src/models/skin_disease_detection.pt"
# Adjust as needed
model = YOLO(model_path)

# Print basic model info
print(model.info())

# Count parameters
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

# Get number of classes and class names
num_classes = model.model.model[-1].nc
class_names = model.names

print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Number of Classes: {num_classes}")
print(f"Class Names: {class_names}")
