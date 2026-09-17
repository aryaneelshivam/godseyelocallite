import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("outputs/sample3/scene/scene.ply")
pcd = pcd.voxel_down_sample(voxel_size=0.1)

print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

print("Orienting normals...")
pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

print("Running Poisson...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
print(f"Created mesh with {len(mesh.vertices)} vertices!")

