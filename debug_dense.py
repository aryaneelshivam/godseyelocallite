import torch
from transformers import pipeline
import pycolmap
import numpy as np
import scipy.stats
from PIL import Image

reconstruction = pycolmap.Reconstruction("outputs/sample/reconstruction/0")
images = list(reconstruction.images.values())
image = images[5]

depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
pil_img = Image.open(f"outputs/sample/keyframes/{image.name}").convert("RGB")
depth_output = depth_estimator(pil_img)
mono_disp = np.array(depth_output["depth"])
mono_disp = np.clip(mono_disp, 1e-6, None)
mono_depth = 1.0 / mono_disp

camera = reconstruction.cameras[image.camera_id]
R = image.cam_from_world().rotation.matrix()
t = image.cam_from_world().translation

sparse_zs = []
mono_zs = []
H, W = mono_depth.shape

for point2D in image.points2D:
    if point2D.has_point3D():
        p3D = reconstruction.points3D[point2D.point3D_id]
        X_cam = R @ p3D.xyz + t
        z_metric = X_cam[2]
        if z_metric <= 0: continue
        x, y = point2D.xy
        ix = int(x * (W / camera.width))
        iy = int(y * (H / camera.height))
        if 0 <= ix < W and 0 <= iy < H:
            sparse_zs.append(z_metric)
            mono_zs.append(mono_depth[iy, ix])

print(f"Num sparse points: {len(sparse_zs)}")
res = scipy.stats.linregress(mono_zs, sparse_zs)
print(f"Inverse Slope (s): {res.slope}, Intercept (c): {res.intercept}")

mono_raw = []
for point2D in image.points2D:
    if point2D.has_point3D():
        p3D = reconstruction.points3D[point2D.point3D_id]
        X_cam = R @ p3D.xyz + t
        if X_cam[2] > 0:
            x, y = point2D.xy
            ix = int(x * (W / camera.width))
            iy = int(y * (H / camera.height))
            if 0 <= ix < W and 0 <= iy < H:
                mono_raw.append(mono_disp[iy, ix])

res_raw = scipy.stats.linregress(mono_raw, sparse_zs)
print(f"Raw Slope: {res_raw.slope}, Raw Intercept: {res_raw.intercept}")

