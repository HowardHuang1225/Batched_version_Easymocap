"""
Identify which 51-face-landmark layout your SMPL-X regressor uses.

Run inside the EasyMocap conda env. Adjust the model loading lines
to match your config (model_path, regressor_path).

Output:
  - Prints 3D positions of indices 67-117 in the keypoint output
  - Saves a scatter plot of those 51 points (front view)
  - Tells you whether the layout includes jaw contour
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# ── Adjust these to your setup ────────────────────────────────────────────────
import sys
sys.path.append('/workspace/EasyMocap')   # adjust if needed
from easymocap.bodymodel.smplx import SMPLXModel  # or wherever your model lives

MODEL_PATH = 'data/smplx/smplx/SMPLX_NEUTRAL.pkl'
REGRESSOR_PATH = 'data/smplx/J_regressor_body25_smplx.txt'
NUM_PCA_COMPS = 12
# ──────────────────────────────────────────────────────────────────────────────

model = SMPLXModel(
    model_path=MODEL_PATH,
    regressor_path=REGRESSOR_PATH,
    num_pca_comps=NUM_PCA_COMPS,
    use_pca=True,
    NUM_SHAPES=10,
    num_expression_coeffs=10,
)
model.eval()

# Neutral pose
B = 1
params = {
    'poses':      torch.zeros(B, 165),
    'shapes':     torch.zeros(B, 10),
    'expression': torch.zeros(B, 10),
    'Rh':         torch.zeros(B, 3),
    'Th':         torch.zeros(B, 3),
}

with torch.no_grad():
    keypoints = model.keypoints(params, return_tensor=True)

kp = keypoints[0].cpu().numpy()
print(f'Total keypoints output: {kp.shape[0]}')
print()

# Extract face range (67-117)
face_kp = kp[67:118]
print(f'Face range 67-117 stats:')
print(f'  X (left-right):  min={face_kp[:, 0].min():.3f}  max={face_kp[:, 0].max():.3f}  range={np.ptp(face_kp[:, 0]):.3f}')
print(f'  Y (up-down):     min={face_kp[:, 1].min():.3f}  max={face_kp[:, 1].max():.3f}  range={np.ptp(face_kp[:, 1]):.3f}')
print(f'  Z (front-back):  min={face_kp[:, 2].min():.3f}  max={face_kp[:, 2].max():.3f}  range={np.ptp(face_kp[:, 2]):.3f}')
print()

# Find lowest-Y points (likely chin)
y_coords = face_kp[:, 1]
sorted_idx = np.argsort(y_coords)
print('Lowest 5 Y-coordinates in face range (likely chin if contour included):')
for rank in range(5):
    local_idx = sorted_idx[rank]
    global_idx = 67 + local_idx
    print(f'  #{rank+1}: keypoint[{global_idx}]  y={y_coords[local_idx]:.4f}  '
          f'x={face_kp[local_idx, 0]:+.4f}  z={face_kp[local_idx, 2]:+.4f}')
print()

# Plot front view (X vs Y)
fig, ax = plt.subplots(figsize=(10, 12))
ax.scatter(face_kp[:, 0], face_kp[:, 1], s=20, c='steelblue')
for local_idx, (x, y, z) in enumerate(face_kp):
    global_idx = 67 + local_idx
    ax.annotate(str(global_idx), (x, y), fontsize=7, ha='center', va='center')
ax.set_xlabel('X (left -> right)')
ax.set_ylabel('Y (down -> up)')
ax.set_aspect('equal')
ax.set_title('SMPL-X face landmarks (indices 67-117), neutral pose, front view')
ax.grid(True, alpha=0.3)
fig.savefig('face_layout.png', dpi=120, bbox_inches='tight')
print('Saved face_layout.png in current directory')
print()

# Heuristic: layout detection
# - Layout B (with chin contour): ~17 points form a U-shape at the bottom of the face
#   The lowest ~17 points span a wide X range (cheek to cheek)
# - Layout A (no chin): mouth is the lowest, ~5-10 points at the bottom, narrow X range
lowest_17 = sorted_idx[:17]
x_spread_low17 = np.ptp(face_kp[lowest_17, 0])
total_x_spread = np.ptp(face_kp[:, 0])
ratio = x_spread_low17 / total_x_spread

print(f'Spread of lowest 17 points (X): {x_spread_low17:.3f}')
print(f'Total face X spread:            {total_x_spread:.3f}')
print(f'Ratio: {ratio:.2f}')
print()
if ratio > 0.85:
    print('>>> LIKELY LAYOUT B: face landmarks INCLUDE jaw contour')
    print('>>> Lowest 17 points span almost full face width = chin curve from ear to ear')
    print('>>> You CAN add jaw points to fitting directly.')
else:
    print('>>> LIKELY LAYOUT A: face landmarks DO NOT include jaw contour')
    print('>>> Lowest points are concentrated around the mouth, not spreading sideways')
    print('>>> Cannot directly add jaw landmarks; would need vertex-based approach.')