import cv2
import os
from glob import glob

def resize_images_and_labels(input_dir, output_dir, label_dir, target_size=(640, 640)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, 'labels')):
        os.makedirs(os.path.join(output_dir, 'labels'))

    image_paths = glob(os.path.join(input_dir, '*.*'))
    
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w, _ = img.shape
        scale = min(target_size[0] / h, target_size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (nw, nh))

        # 创建带有填充的目标图像
        top, bottom = (target_size[0] - nh) // 2, (target_size[0] - nh + 1) // 2
        left, right = (target_size[1] - nw) // 2, (target_size[1] - nw + 1) // 2
        color = [0, 0, 0]
        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # 保存调整大小后的图像
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, padded_img)
        print(f'Resized and saved image to: {output_path}')

        # 调整标签
        label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            new_label_path = os.path.join(output_dir, 'labels', os.path.basename(label_path))
            with open(new_label_path, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    center_x = float(parts[1]) * w * scale + left
                    center_y = float(parts[2]) * h * scale + top
                    bbox_width = float(parts[3]) * w * scale
                    bbox_height = float(parts[4]) * h * scale

                    # 归一化新的标签坐标
                    center_x /= target_size[1]
                    center_y /= target_size[0]
                    bbox_width /= target_size[1]
                    bbox_height /= target_size[0]

                    f.write(f'{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n')
            print(f'Adjusted and saved label to: {new_label_path}')

input_dir = '/home/dataset/images/train'
output_dir = '/home/dataset/images/new_train'
label_dir = '/home/dataset/labels/train'
resize_images_and_labels(input_dir, output_dir, label_dir)
