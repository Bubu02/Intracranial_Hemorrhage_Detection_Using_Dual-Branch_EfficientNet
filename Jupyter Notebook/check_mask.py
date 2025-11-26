
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Path to a segmentation image
seg_image_path = r'c:\Users\mebub_9a7jdi8\Desktop\Brain Stroke BloodClot Detection\Dataset\Patients_CT\049\brain\14_HGE_Seg.jpg'

if os.path.exists(seg_image_path):
    img = mpimg.imread(seg_image_path)
    print(f"Image shape: {img.shape}")
    print(f"Unique values: {np.unique(img)}")
    
    plt.imshow(img, cmap='gray')
    plt.title("Segmentation Mask Check")
    plt.show()
else:
    print("File not found")
