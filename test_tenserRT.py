from ultralytics import YOLO
import os
import glob
import time
import cv2
import numpy as np

# -----------------------------
# 參數設定
# -----------------------------
engine_path = "yolov8m.engine"  # 動態 batch engine
base_dir = "../Data/data/baseball0/mocap/images"
cams = ["cam1", "cam3", "cam4"]
batch_size = 3  # 每次推理 3 張圖

# -----------------------------
# 載入模型
# -----------------------------
model = YOLO(engine_path)

# -----------------------------
# 遍歷每個 batch (cam1, cam3, cam4)
# -----------------------------
num_batches = len(sorted(glob.glob(os.path.join(base_dir, cams[0], "*.*"))))
total_inference_time = 0

for i in range(num_batches):
    batch_imgs = []
    batch_names = []
    
    # 每個 batch 從各相機取 1 張圖
    for cam in cams:
        cam_dir = os.path.join(base_dir, cam)
        img_paths = sorted(glob.glob(os.path.join(cam_dir, "*.*")))
        if i < len(img_paths):
            img_path = img_paths[i]
            img = cv2.imread(img_path)
            batch_imgs.append(img)
            batch_names.append(f"{cam} image {i+1}")
    
    if not batch_imgs:
        continue
    
    start_time = time.time()
    results_list = model(batch_imgs)  # dynamic batch inference
    elapsed = time.time() - start_time
    total_inference_time += elapsed
    
    for j, results in enumerate(results_list):
        boxes = results.boxes
        classes = [results.names[int(c)] for c in boxes.cls]
        # 只保留 person
        person_boxes = [boxes.xyxy[k] for k, c in enumerate(classes) if c == 'person']
        print(f"{batch_names[j]}: Detected {len(person_boxes)} persons")
    
avg_time = total_inference_time / num_batches
print(f"\nAverage inference time per image: {avg_time*1000:.2f} ms")
