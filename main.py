import cv2
from ultralytics import YOLO
import os
import torch
import yaml

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/home/dataset/data.yaml", epochs=3,batch=2,imgsz=512)


# Evaluate the model's performance on the validation set
torch.cuda.empty_cache()
results = model.val()

# Perform object detection on an image using the model

image_path = "/home/image.png"
results = model(image_path)
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Failed to load image from {image_path}")

# 遍历检测结果并绘制边界框
for result in results:
    for box in result.boxes:
        # 提取边界框信息
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 左上角和右下角的坐标
        confidence = box.conf.item()  # 置信度
        class_id = int(box.cls.item())  # 类别ID
        class_name = result.names[class_id]  # 类别名称

        # 绘制边界框和标签
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 保存带有标注的图像
cv2.imwrite("annotated_image3.jpg", image)

# Export the model to ONNX format
success = model.export(format="onnx")
files = os.listdir(".")
print(files)
print(success)

torch.cuda.empty_cache()
