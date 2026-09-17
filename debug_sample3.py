import torch
from transformers import pipeline
import pycolmap
import numpy as np
import scipy.stats
from PIL import Image

reconstruction = pycolmap.Reconstruction("outputs/sample3/reconstruction/0")
images = list(reconstruction.images.values())

depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)

for image in images:
    if not getattr(image, "has_pose", False) and not getattr(image, "is_registered", False):
        if hasattr(image, "is_registered") and callable(image.is_registered):
            if not image.is_registered(): continue
        else:
            continue
            
    pil_img = Image.open(f"outputs/sample3/keyframes/{image.name}").convert("RGB")
    depth_output = depth_estimator(pil_img)
    mono_depth = np.array(depth_output["depth"], dtype=float)
    
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
                
    if len(sparse_zs) < 10:
        print(f"{image.name}: skipped due to <10 sparse points ({len(sparse_zs)})")
        continue
        
    res = scipy.stats.linregress(mono_zs, sparse_zs)
    print(f"{image.name}: slope={res.slope}, intercept={res.intercept}, rvalue={res.rvalue}")
