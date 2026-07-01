import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / ".." / "dataset" / "raw"
dataset_path = dataset_path.resolve()

classes = os.listdir(dataset_path) ## raw, sample, processed
#dataset_path = "../dataset/raw"
#classes = os.listdir(dataset_path)  

print("Classes found:", classes)
print("Number of classes:", len(classes))


for cls in classes:
    path = os.path.join(dataset_path, cls)
    print(f"{cls}: {len(os.listdir(path))} images")#Count Images per Class


#VisualizeSampleImages
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
#from PIL import Image

plt.figure(figsize=(12, 8))

valid_extensions = ('.jpg', '.jpeg', '.png')

for i, cls in enumerate(classes):
    folder = os.path.join(dataset_path, cls)
   
    
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
    
    if len(files) == 0:
        print("No images in:", folder)
        continue
    
    
    img_path = os.path.join(folder, files[0])  
    
    print("Loading:", img_path)
    img = cv2.imread(img_path)
    
    
    if img is None:
        print("Skipping invalid image:", img_path)
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")
    plt.show()
