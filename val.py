import cv2
from ultralytics import YOLO
import os
import torch

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Paths
data_yaml_path = "/home/dataset/data.yaml"
image_path = "/home/image.jpg"
trained_model_path = "/home/trained_model.pt" 

# Load the trained YOLO model

model = YOLO(trained_model_path)  # 使用训练后的模型进行预测


# Clean up memory cache and evaluate the model
torch.cuda.empty_cache()

results = model.val(data=data_yaml_path)


# Perform object detection on an image using the model

results = model(image_path,conf=0.40, iou=0.50)

# print(results)
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Iterate through detection results and draw bounding boxes
for result in results:
    for box in result.boxes:
        # Extract bounding box information
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Top-left and bottom-right coordinates
        confidence = box.conf.item()  # Confidence score
        class_id = int(box.cls.item())  # Class ID
        class_name = result.names[class_id]  # Class name

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save the annotated image
cv2.imwrite("annotated_image14.jpg", image)

torch.cuda.empty_cache()