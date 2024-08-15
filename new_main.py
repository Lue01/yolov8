import cv2
from ultralytics import YOLO
import os
import torch

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Paths
data_yaml_path = "/home/dataset/data.yaml"
image_path = "/home/image.png"
trained_model_path = "/home/trained_model.pt" 

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")  # 使用预训练模型进行训练

# Train the model using the 'data.yaml' dataset for 3 epochs
results = model.train(data=data_yaml_path, epochs=3, batch=2, imgsz=512)

# Save the trained model
model.save(trained_model_path)   

"""
# Load the trained YOLO model
model = YOLO(trained_model_path)  # 使用训练后的模型进行预测

# Clean up memory cache and evaluate the model
torch.cuda.empty_cache()
results = model.val()

# Perform object detection on an image using the model
results = model(image_path)
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
cv2.imwrite("annotated_image4.jpg", image)

# Export the model to ONNX format
success = model.export(format="onnx")
files = os.listdir(".")
print(files)
print(success)

torch.cuda.empty_cache()
"""