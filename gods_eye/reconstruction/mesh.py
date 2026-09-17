import open3d as o3d
import argparse
from pathlib import Path
import logging
import numpy as np

def create_mesh(dense_ply_path: Path, output_mesh_path: Path, depth: int = 9, density_threshold: float = 0.05):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not dense_ply_path.exists():
        logging.error(f"Input point cloud {dense_ply_path} does not exist.")
        return
        
    logging.info(f"Loading dense point cloud from {dense_ply_path}...")
    pcd = o3d.io.read_point_cloud(str(dense_ply_path))
    
    logging.info(f"Loaded {len(pcd.points)} points.")
    
    # Downsample to a memory-safe resolution for meshing
    logging.info("Downsampling and cleaning point cloud...")
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    if not pcd.has_normals():
        logging.info("Estimating surface normals (this may take a moment)...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # Use an instantaneous camera-oriented approach instead of the RAM-heavy MST graph
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        
    logging.info(f"Running Poisson Surface Reconstruction (Depth: {depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    # Remove low density vertices (the strange "bubble" that Poisson puts around the edges)
    logging.info("Cropping extrapolated boundary artifacts...")
    densities = np.asarray(densities)
    density_val = np.quantile(densities, density_threshold)
    vertices_to_remove = densities < density_val
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    logging.info(f"Writing fully formed mesh to {output_mesh_path}...")
    o3d.io.write_triangle_mesh(str(output_mesh_path), mesh)
    logging.info("Meshing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to scene_dense.ply")
    parser.add_argument("--output", type=str, required=True, help="Path to save scene_mesh.ply")
    parser.add_argument("--depth", type=int, default=9, help="Poisson tree depth (higher = more detail)")
    args = parser.parse_args()
    
    create_mesh(Path(args.input), Path(args.output), args.depth)
