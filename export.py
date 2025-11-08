from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# 1. 載入原始 YOLOv8 模型
model = YOLO("yolov8m.pt") 

# 2. 匯出為 TensorRT engine
model.export(format="engine",batch=40,dynamic=True , half=True, nms=True,task="detect")  # 會生成 yolov8m.engine

# 3. 載入 TensorRT engine
tensorrt_model = YOLO("yolov8m.engine") 

# 4. 推理
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")
