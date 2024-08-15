import cv2
from ultralytics import YOLO
import os
import torch

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 配置路径
data_yaml_path = "/home/dataset/data.yaml"
input_folder = "/home/test"  # 输入图片的文件夹路径
output_folder = "/home/output_test"  # 输出图片的文件夹路径
trained_model_path = "/home/trained_model.pt" 

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model = YOLO(trained_model_path)

torch.cuda.empty_cache()

# 遍历输入文件夹中的所有图片
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    results = model(image_path, conf=0.40, iou=0.50)

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
        continue

    # 绘制检测框
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            class_name = result.names[class_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 保存标注后的图片到输出文件夹
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)
    print(f"已处理并保存: {output_path}")

torch.cuda.empty_cache()
