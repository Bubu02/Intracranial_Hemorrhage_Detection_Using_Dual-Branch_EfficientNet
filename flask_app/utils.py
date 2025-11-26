from ultralytics import YOLO
import os
import cv2
from pathlib import Path

# Load the YOLO model
MODEL_PATH = os.path.join('..', 'Jupyter Notebook', 'brain_hemorrhage_project', 'yolov8n_run2', 'weights', 'best.pt')

# Fallback to base model if trained model not found
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join('..', 'Jupyter Notebook', 'yolov8n.pt')

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

def process_image(image_path, result_folder):
    """
    Process an image with YOLO model and return the result path and detections.
    
    Args:
        image_path: Path to the input image
        result_folder: Folder to save the result image
    
    Returns:
        tuple: (result_image_path, detections_list)
    """
    try:
        # Run inference
        results = model(image_path, conf=0.25)
        
        # Get the result image with bounding boxes
        result = results[0]
        
        # Save the annotated image
        result_filename = f"result_{Path(image_path).name}"
        result_path = os.path.join(result_folder, result_filename)
        
        # Plot and save the result
        annotated_img = result.plot()
        cv2.imwrite(result_path, annotated_img)
        
        # Extract detection information
        detections = []
        for box in result.boxes:
            detection = {
                'class': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            }
            detections.append(detection)
        
        return result_path, detections
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, []
