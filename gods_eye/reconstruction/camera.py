import json
import pycolmap
from pathlib import Path
import logging

def export_camera_trajectory(reconstruction: pycolmap.Reconstruction, output_path: Path):
    """Exports the camera trajectory in a JSON format."""
    trajectory = []
    
    # Iterate through all registered images
    for image_id, image in reconstruction.images.items():
        if getattr(image, "has_pose", False) == False and getattr(image, "is_registered", False) == False:
            if hasattr(image, "is_registered") and callable(image.is_registered):
                if not image.is_registered(): continue
            else:
                continue
            
        # Get camera center and rotation
        cam_center = image.projection_center()
        cam_rot = image.cam_from_world().rotation.matrix()
        
        entry = {
            "image_id": image_id,
            "name": image.name,
            "position": cam_center.tolist(),
            "rotation": cam_rot.tolist()
        }
        trajectory.append(entry)
        
    # Sort by image name assuming frames are named sequentially
    trajectory = sorted(trajectory, key=lambda x: x["name"])
    
    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=4)
        
    logging.info(f"Exported camera trajectory to {output_path}")
