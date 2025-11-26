
from ultralytics import YOLO
import os

def main():
    # Load a model
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # load a pretrained model

    # Define path to data.yaml
    # Assuming this script is in 'c:\Users\mebub_9a7jdi8\Desktop\Brain Stroke BloodClot Detection\Jupyter Notebook'
    # and datasets are in 'c:\Users\mebub_9a7jdi8\Desktop\Brain Stroke BloodClot Detection\datasets'
    data_path = os.path.abspath(os.path.join('..', 'datasets', 'brain_hemorrhage', 'data.yaml'))
    
    if not os.path.exists(data_path):
        print(f"Error: data.yaml not found at {data_path}")
        return

    print(f"Training with data config: {data_path}")

    # Train the model
    # Using a small number of epochs for demonstration/quick feedback, user can increase later
    results = model.train(data=data_path, epochs=10, imgsz=640, project='brain_hemorrhage_project', name='yolov8n_run')
    
    print("Training complete.")

    # Validate the model
    print("Validating model...")
    metrics = model.val()
    print("Validation complete.")

if __name__ == "__main__":
    main()
