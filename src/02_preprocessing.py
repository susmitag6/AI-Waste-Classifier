#Full Preprocessing Loop
import os
import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 224

data = []
labels = []

BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / ".." / "dataset" / "raw"
dataset_path = dataset_path.resolve()

classes = sorted(os.listdir(dataset_path))  # sorted so class -> index stays consistent everywhere

class_to_index = {cls: idx for idx, cls in enumerate(classes)}
print(class_to_index)
count = 0
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
#        print(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print("No Image In")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # normalize
        img = img.astype(np.float32) / 255.0
#        img = img / 255.0
        data.append(img)
        labels.append(class_to_index[cls])
        count += 1
        if count % 500 == 0:
            print("Processed:", count)

print("LOOP FINISHED")
print("Before conversion")
print("Data samples:", len(data))
print("Labels samples:", len(labels))
print("Sample shape:", data[0].shape)
data = np.array(data)

labels = np.array(labels)

#print("Data shape:", data.shape)
#print("Labels shape:", labels.shape) 

np.save("data.npy", data)
np.save("labels.npy", labels)

print("Saved dataset!")

#assert 'data' in globals()
#assert 'labels' in globals()



