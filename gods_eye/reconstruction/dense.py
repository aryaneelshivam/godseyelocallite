import logging
from pathlib import Path
import numpy as np

try:
    import torch
    from transformers import pipeline
except ImportError:
    pass

import pycolmap
import scipy.stats
from tqdm import tqdm

class MonocularDenseReconstructor:
    def __init__(self, db_path: Path, frames_dir: Path, output_dir: Path, config):
        self.db_path = db_path
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.config = config
        self.reconstruction_dir = output_dir
        
    def run(self):
        if "torch" not in globals():
            logging.error("Dense reconstruction requires torch and transformers.")
            return None
            
        device = -1 # Force CPU execution to prevent Apple Silicon MPS SIGSEGV crashes
        
        logging.info("Loading DepthAnything-v2 model...")
        depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        
        model_path = self.reconstruction_dir / "0"
        if not model_path.exists():
            logging.error("No valid sparse reconstruction found to densify.")
            return None
            
        reconstruction = pycolmap.Reconstruction(model_path)
        
        all_points = []
        all_colors = []
        
        images = list(reconstruction.images.values())
        
        for image in tqdm(images, desc="AI Dense Reconstruction", position=1, leave=False):
            if not getattr(image, "has_pose", False) and not getattr(image, "is_registered", False):
                if hasattr(image, "is_registered") and callable(image.is_registered):
                    if not image.is_registered(): continue
                else:
                    continue
                
            frame_path = self.frames_dir / image.name
            if not frame_path.exists():
                continue
                
            from PIL import Image
            pil_img = Image.open(frame_path).convert("RGB")
            
            # 1. Predict Monocular Depth
            depth_output = depth_estimator(pil_img)
            
            # Depth-Anything outputs relative disparity (0-255, where brighter = closer)
            # We normalize to 0-1 and strictly bound to prevent extreme outlier 1/x spikes
            mono_disp = np.array(depth_output["depth"], dtype=float) / 255.0
            mono_disp = np.clip(mono_disp, 0.05, 1.0) # Max inverse depth multiplier is 20x
            
            # Convert disparity to true relative depth
            mono_depth = 1.0 / mono_disp
            
            # 2. Extract Sparse Points for Alignment
            camera = reconstruction.cameras[image.camera_id]
            cam_pose = image.cam_from_world()
            R = cam_pose.rotation.matrix()
            t = cam_pose.translation
            
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
                continue
                
            # 3. Align Metric Scale
            mono_zs = np.array(mono_zs)
            sparse_zs = np.array(sparse_zs)
            
            res = scipy.stats.linregress(mono_zs, sparse_zs)
            s, c = res.slope, res.intercept
            
            # Prevent negative or extremely broken scales
            if s <= 0:
                continue
                
            metric_depth = s * mono_depth + c
            valid_mask = metric_depth > 0
            
            v, u = np.indices((H, W))
            u = u[valid_mask]
            v = v[valid_mask]
            Z = metric_depth[valid_mask]
            
            u_cam = u * (camera.width / W)
            v_cam = v * (camera.height / H)
                
            focal_x = camera.focal_length_x
            focal_y = camera.focal_length_y
            cx = camera.principal_point_x
            cy = camera.principal_point_y
                
            X = (u_cam - cx) * Z / focal_x
            Y = (v_cam - cy) * Z / focal_y
            
            pts_cam = np.stack([X, Y, Z], axis=1)
            
            # Subsample for performance/memory (e.g., skip 64 pixels at a time)
            step = 16
            pts_cam = pts_cam[::step]
            u_img = u[::step]
            v_img = v[::step]
            
            # Transform to World Space
            pts_world = (pts_cam - t) @ R
            
            rgb = np.array(pil_img)
            # Map back to original image resolution for color extraction
            u_orig = (u_img * (pil_img.width / W)).astype(int)
            v_orig = (v_img * (pil_img.height / H)).astype(int)
            u_orig = np.clip(u_orig, 0, pil_img.width - 1)
            v_orig = np.clip(v_orig, 0, pil_img.height - 1)
            
            colors = rgb[v_orig, u_orig]
            
            all_points.append(pts_world)
            all_colors.append(colors)
            
        if not all_points:
            logging.error("Failed to extract any dense points.")
            return None
            
        dense_pts = np.vstack(all_points)
        dense_colors = np.vstack(all_colors)
        
        out_ply = self.reconstruction_dir / "scene_dense.ply"
        
        logging.info(f"Writing {len(dense_pts)} dense points to {out_ply}...")
        with open(out_ply, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(dense_pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(dense_pts, dense_colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
                
        return out_ply

if __name__ == "__main__":
    import argparse
    from gods_eye.config.settings import Config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args()
    
    config = Config(extraction_fps=1, features=2000, matching_window=3, blur_threshold=100.0, similarity_threshold=0.85, max_image_size=args.resolution)
    
    reconstructor = MonocularDenseReconstructor(
        Path(args.db_path), Path(args.frames_dir), Path(args.output_dir), config
    )
    reconstructor.run()
