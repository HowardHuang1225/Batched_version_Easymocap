# easymocap_nvtx_test.py
import torch
import torch.cuda.nvtx as nvtx
import time

# 建立一個 GPU tensor 模擬輸入影像
x = torch.randn(32, 3, 256, 256).cuda()  # batch_size=32, RGB, 256x256

# -------------------------------
# 模擬三個 stage
# -------------------------------

def detect(images):
    with nvtx.range("detect"):
        time.sleep(0.2)  # 模擬 CPU/GPU 運算
        return images * 0.5  # 模擬輸出

def keypoints2d(images):
    with nvtx.range("keypoints2d"):
        time.sleep(0.3)
        return images + 1  # 模擬輸出

def lbfgs(images):
    with nvtx.range("lbfgs"):
        time.sleep(0.4)
        y = torch.sum(images)  # 模擬優化計算
        return y

# -------------------------------
# 主流程
# -------------------------------
with nvtx.range("EasyMocap_pipeline"):
    det_out = detect(x)
    kp_out = keypoints2d(det_out)
    result = lbfgs(kp_out)

print("Done! Result =", result.item())
