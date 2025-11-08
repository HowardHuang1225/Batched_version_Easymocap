# benchmark_tensorrt_yolo.py
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os
from glob import glob

# -------------------------------
# 配置區
# -------------------------------
CONF_THRESH = 0.25
IOU_THRESH = 0.45
INPUT_SIZE = 640
PERSON_CLASS_ID = 0
repeat = 3  # forward benchmark 重複次數
img_dir = "../Data/data/baseball0/mocap/images/cam3"
engine_paths = {
    "yolov5m": "yolov5m.engine"
}

# -------------------------------
# NMS 函數
# -------------------------------
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, class_id=0):
    boxes = []
    for det in prediction:
        scores = det[5:]
        cid = np.argmax(scores)
        conf = det[4] * scores[cid]
        if conf > conf_thres and cid == class_id:
            x, y, w, h = det[:4]
            boxes.append([x, y, w, h, conf, cid])
    if not boxes:
        return []
    boxes = np.array(boxes)
    indices = cv2.dnn.NMSBoxes(
        boxes[:, :4].tolist(), boxes[:, 4].tolist(), conf_thres, iou_thres
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    return boxes[indices]

# -------------------------------
# 讀取圖片
# -------------------------------
img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))
if len(img_paths) == 0:
    raise FileNotFoundError(f"No images found in {img_dir}")

imgs = []
orig_shapes = []
for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        continue
    orig_shapes.append(img.shape[:2])
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_input = img_resized[:, :, ::-1].transpose(2,0,1)  # BGR->RGB, HWC->CHW
    img_input = np.ascontiguousarray(img_input, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    imgs.append(img_input)

print(f"Loaded {len(imgs)} images from {img_dir}")

# -------------------------------
# Benchmark 每個 engine
# -------------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

for name, engine_path in engine_paths.items():
    print(f"\n===== Benchmark {name} =====")

    # Load TensorRT engine
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Allocate buffers
    output_shape = (1, 25200, 85)  # 根據 engine 調整
    d_input = cuda.mem_alloc(imgs[0].nbytes)
    output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    bindings = [int(d_input), int(d_output)]

    # ---------- Forward benchmark ----------
    start = time.time()
    for _ in range(repeat):
        for img in imgs:
            cuda.memcpy_htod(d_input, img)
            context.execute_v2(bindings)
            cuda.memcpy_dtoh(output, d_output)
    end = time.time()
    total_frames = len(imgs) * repeat
    avg_forward_time = (end - start) / total_frames
    fps = 1 / avg_forward_time
    print(f"Avg raw forward time: {avg_forward_time*1000:.2f} ms | FPS ≈ {fps:.2f}")

    # ---------- Accuracy benchmark (人物) ----------
    total_people = 0
    total_conf = 0.0
    total_boxes = 0
    for img_input, orig_shape in zip(imgs, orig_shapes):
        pred = output[0]
        detections = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, PERSON_CLASS_ID)
        for det in detections:
            x, y, w, h, conf, cid = det
            total_people += 1
            total_boxes += 1
            total_conf += conf
    avg_conf = total_conf / total_boxes if total_boxes > 0 else 0.0
    print(f"Detected {total_people} people in {len(imgs)} images, avg confidence = {avg_conf:.2f}")

    # Clean up
    del context
    del engine

print("\n✅ Benchmark complete.")
