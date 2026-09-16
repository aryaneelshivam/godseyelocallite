import json
import trimesh
import numpy as np
from pathlib import Path

class InteractiveViewer:
    def __init__(self, scene_path: str, trajectory_path: str):
        self.scene_path = Path(scene_path)
        self.trajectory_path = Path(trajectory_path)
        
    def load_scene(self):
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene file not found: {self.scene_path}")
            
        print(f"Loading point cloud from {self.scene_path}...")
        pcd = trimesh.load(self.scene_path)
        return pcd
        
    def load_trajectory(self):
        if not self.trajectory_path.exists():
            print(f"Trajectory file not found: {self.trajectory_path}")
            return []
            
        print(f"Loading camera trajectory from {self.trajectory_path}...")
        with open(self.trajectory_path, "r") as f:
            trajectory_data = json.load(f)
            
        camera_geometries = []
        for frame in trajectory_data:
            pos = np.array(frame["position"])
            rot = np.array(frame["rotation"])
            
            extents = [0.2, 0.2, 0.4]
            cam_mesh = trimesh.creation.box(extents=extents)
            
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = pos
            cam_mesh.apply_transform(transform)
            
            cam_mesh.visual.vertex_colors = [255, 0, 0, 255]
            
            camera_geometries.append(cam_mesh)
            
        return camera_geometries
        
    def show(self):
        scene_elements = []
        
        try:
            pcd = self.load_scene()
            scene_elements.append(pcd)
        except Exception as e:
            print(f"Error loading scene: {e}")
            
        try:
            cameras = self.load_trajectory()
            scene_elements.extend(cameras)
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            
        if not scene_elements:
            print("Nothing to display.")
            return
            
        print("Starting interactive viewer... (Close the window to exit)")
        scene = trimesh.Scene(scene_elements)
        scene.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Path to scene.ply")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to camera_trajectory.json")
    args = parser.parse_args()
    
    viewer = InteractiveViewer(args.scene, args.trajectory)
    viewer.show()
