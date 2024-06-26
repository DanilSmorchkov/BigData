import os
import numpy as np

from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\User\PycharmProjects\BigData\Lab3\best-2.pt")  # pretrained YOLOv8n model

imgs_path = []

for dirpath, _, filenames in os.walk(r'C:\Users\User\PycharmProjects\BigData\Lab3\train\images'):
    for f in filenames:
        imgs_path.append(os.path.abspath(os.path.join(dirpath, f)))

indexes = np.random.randint(0, len(imgs_path), size=10)

# Run batched inference on a list of images
results = model(list(np.array(imgs_path)[indexes]))  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print(boxes)
    result.show()  # display to screen

