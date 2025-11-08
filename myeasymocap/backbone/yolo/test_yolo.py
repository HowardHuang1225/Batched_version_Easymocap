# benchmark_yolo_pt_and_trt.py
import torch
import time
from ultralytics import YOLO
import cv2
import os
from glob import glob
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# -------------------------------
# 配置區
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
repeat = 3  # forward benchmark 重複次數
INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
PERSON_CLASS_ID = 0

# 模型列表：pt 或 engine
models = {
    "yolov5m": "./yolov5m.pt",
    "yolov5mrt": "./yolov5m.engine"
}

# -------------------------------
# Non-Max Suppression for TensorRT
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
# 載入圖片
# -------------------------------
img_dir = "./cam3"
img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))
if len(img_paths) == 0:
    raise FileNotFoundError(f"No images found in {img_dir}")

imgs_pt = []        # for torch model
imgs_trt = []       # for tensorrt model
orig_shapes = []

for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        continue
    orig_shapes.append(img.shape[:2])
    # Torch preprocessing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)
    imgs_pt.append(img_t)
    # TensorRT preprocessing
    img_input = img_resized[:, :, ::-1].transpose(2,0,1)  # BGR->RGB, HWC->CHW
    img_input = np.ascontiguousarray(img_input, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    imgs_trt.append(img_input)

print(f"Loaded {len(imgs_pt)} images from {img_dir}")

# -------------------------------
# Benchmark loop
# -------------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

for name, path in models.items():
    print(f"\n===== Benchmark {name} =====")

    is_trt = path.endswith(".engine")
    # ---------- Load model ----------
    if is_trt:
        with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
        d_input = cuda.mem_alloc(imgs_trt[0].nbytes)
        output_shape = (1, 25200, 85)  # adjust for your engine
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        bindings = [int(d_input), int(d_output)]
        imgs = imgs_trt
    else:
        model = YOLO(path)
        model.to(device)
        forward_model = model.model
        forward_model.eval()
        imgs = imgs_pt

    # ---------- Forward benchmark ----------
    start = time.time()
    for _ in range(repeat):
        for img in imgs:
            if is_trt:
                cuda.memcpy_htod(d_input, img)
                context.execute_v2(bindings)
                cuda.memcpy_dtoh(output, d_output)
            else:
                with torch.no_grad():
                    _ = forward_model(img)
                    if device=="cuda":
                        torch.cuda.synchronize()
    end = time.time()
    total_frames = len(imgs) * repeat
    avg_forward_time = (end - start) / total_frames
    fps = 1 / avg_forward_time
    print(f"Avg raw forward time: {avg_forward_time*1000:.2f} ms | FPS ≈ {fps:.2f}")

    # ---------- Accuracy benchmark (人物) ----------
    total_people = 0
    total_conf = 0.0
    total_boxes = 0
    for i, img in enumerate(imgs):
        if is_trt:
            pred = output[0]
            detections = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, PERSON_CLASS_ID)
            for det in detections:
                x, y, w, h, conf, cid = det
                total_people += 1
                total_boxes += 1
                total_conf += conf
        else:
            img_np = img[0].permute(1,2,0).cpu().numpy() * 255.0
            img_np = img_np.astype('uint8')
            results = model(img_np, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                person_boxes = [b for b in boxes if int(b.cls[0])==0]
                total_people += len(person_boxes)
                total_boxes += len(person_boxes)
                total_conf += sum(float(b.conf[0]) for b in person_boxes)

    avg_conf = total_conf / total_boxes if total_boxes>0 else 0.0
    print(f"Detected {total_people} people in {len(imgs)} images, avg confidence = {avg_conf:.2f}")

    if is_trt:
        del context
        del engine

print("\n✅ Benchmark complete.")
