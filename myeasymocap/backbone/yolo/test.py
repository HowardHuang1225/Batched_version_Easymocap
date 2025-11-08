# File: /yolo-trt-test/yolo-trt-test/src/test_trt_yolo_speed_compare.py

import cv2
import time
import os
import shutil
from yolo import BaseYOLOv5, BaseYOLOv5rt  # 同時 import 兩個版本

def benchmark_model(yolo_model, image_folder):
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))]
    if not images:
        print(f"No images found in {image_folder}")
        return None, None

    timings = []
    all_results = []

    for img_name in images:
        image_path = os.path.join(image_folder, img_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping.")
            continue

        start_time = time.time()
        results = yolo_model.detect(image, image_path)
        end_time = time.time()

        timings.append(end_time - start_time)
        all_results.append((img_name, results))

    if timings:
        avg_time = sum(timings) / len(timings)
        fps = 1.0 / avg_time
        return avg_time, fps, all_results
    else:
        return None, None, None

def main():
    image_folder = 'cam3'
    
    # ---------------- Benchmark BaseYOLOv5 ----------------
    print("=== Benchmark BaseYOLOv5 (PyTorch) ===")
    yolo_v5 = BaseYOLOv5(model='yolov5m', name='person')
    avg_time_v5, fps_v5, results_v5 = benchmark_model(yolo_v5, image_folder)
    if avg_time_v5:
        print(f"BaseYOLOv5 avg time: {avg_time_v5*1000:.2f} ms | FPS ≈ {fps_v5:.2f}")


    # 刪除 output 資料夾
    output_folder = 'output'
    if os.path.exists(output_folder):
        print(f"Removing folder '{output_folder}' before TensorRT benchmark...")
        shutil.rmtree(output_folder)
    # ---------------- Benchmark BaseYOLOv5rt ----------------
    print("\n=== Benchmark BaseYOLOv5rt (TensorRT) ===")
    yolo_rt = BaseYOLOv5rt(engine_path='yolov5m.engine', name='person')
    avg_time_rt, fps_rt, results_rt = benchmark_model(yolo_rt, image_folder)
    if avg_time_rt:
        print(f"BaseYOLOv5rt avg time: {avg_time_rt*1000:.2f} ms | FPS ≈ {fps_rt:.2f}")

    # ---------------- Sample detection results ----------------
    print("\n===== Sample Detection Results (first 3 images) =====")
    for i in range(min(3, len(results_v5))):
        print(f"\n--- Image: {results_v5[i][0]} ---")

        print("[PyTorch results]:")
        print(results_v5[i][1]['results'])

        print("\n[TensorRT results]:")
        print(results_rt[i][1]['results'])


    # 刪除 output 資料夾
    output_folder = 'output'
    if os.path.exists(output_folder):
        print(f"Removing folder '{output_folder}' before TensorRT benchmark...")
        shutil.rmtree(output_folder)

if __name__ == "__main__":
    main()
