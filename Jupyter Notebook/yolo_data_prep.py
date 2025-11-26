
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# Define paths
# Script is in Jupyter Notebook folder, so .. goes to root
dataset_base_path = os.path.join('..', 'Dataset', 'Patients_CT')
output_base_path = os.path.join('..', 'datasets', 'brain_hemorrhage')

# Create YOLO directory structure
images_train_dir = os.path.join(output_base_path, 'images', 'train')
images_val_dir = os.path.join(output_base_path, 'images', 'val')
labels_train_dir = os.path.join(output_base_path, 'labels', 'train')
labels_val_dir = os.path.join(output_base_path, 'labels', 'val')

os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)

print(f"Created directories at {output_base_path}")

def get_bbox_from_mask(mask_path):
    """
    Reads a mask image and returns the bounding box in YOLO format.
    YOLO format: class x_center y_center width height (normalized 0-1)
    """
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    except:
        mask = mpimg.imread(mask_path)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    h, w = mask.shape
    
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 5 or bh < 5:
            continue
            
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h
        
        bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
    return bboxes

data_pairs = []
print("Scanning dataset...")
if os.path.exists(dataset_base_path):
    for patient_id in os.listdir(dataset_base_path):
        patient_dir = os.path.join(dataset_base_path, patient_id, 'brain')
        if not os.path.isdir(patient_dir):
            continue
            
        files = os.listdir(patient_dir)
        images = [f for f in files if f.endswith('.jpg') and '_Seg' not in f]
        
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            mask_name = f"{base_name}_HGE_Seg.jpg"
            
            if mask_name in files:
                data_pairs.append({
                    'image_path': os.path.join(patient_dir, img_name),
                    'mask_path': os.path.join(patient_dir, mask_name),
                    'base_name': f"{patient_id}_{base_name}"
                })
else:
    print(f"Dataset path not found: {dataset_base_path}")

print(f"Found {len(data_pairs)} image-mask pairs.")

if len(data_pairs) > 0:
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)

    def process_batch(pairs, img_dest_dir, lbl_dest_dir):
        for item in tqdm(pairs):
            bboxes = get_bbox_from_mask(item['mask_path'])
            dest_img_path = os.path.join(img_dest_dir, item['base_name'] + '.jpg')
            shutil.copy(item['image_path'], dest_img_path)
            dest_lbl_path = os.path.join(lbl_dest_dir, item['base_name'] + '.txt')
            with open(dest_lbl_path, 'w') as f:
                f.write('\n'.join(bboxes))

    print("Processing training data...")
    process_batch(train_pairs, images_train_dir, labels_train_dir)

    print("Processing validation data...")
    process_batch(val_pairs, images_val_dir, labels_val_dir)

    # Create data.yaml
    yaml_content = f"""
path: {os.path.abspath(output_base_path)}
train: images/train
val: images/val

nc: 1
names: ['hemorrhage']
"""
    yaml_path = os.path.join(output_base_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created data.yaml at {yaml_path}")
else:
    print("No data found to process.")
