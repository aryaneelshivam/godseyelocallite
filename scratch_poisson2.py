import numpy as np
from scipy.spatial import cKDTree
import pycolmap
import trimesh

print("Loading points...")
pcd = trimesh.load("outputs/sample3/reconstruction/scene_sparse.ply")
points = pcd.vertices
colors = pcd.colors

print("Computing KNN...")
tree = cKDTree(points)
k = 15
distances, indices = tree.query(points, k=k)

print("Estimating normals via PCA...")
normals = np.zeros_like(points)
for i in range(len(points)):
    neighbors = points[indices[i]]
    centroid = np.mean(neighbors, axis=0)
    cov = np.cov((neighbors - centroid).T)
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]
    normals[i] = normal

# Ensure consistent normal orientation
centroid = np.mean(points, axis=0)
for i in range(len(points)):
    if np.dot(normals[i], points[i] - centroid) < 0:
        normals[i] = -normals[i]

print("Writing new PLY with normals...")
with open("outputs/sample3/reconstruction/scene_sparse_normals.ply", "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(points)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property float nx\nproperty float ny\nproperty float nz\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    for p, n, c in zip(points, normals, colors):
        f.write(f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]} {c[0]} {c[1]} {c[2]}\n")

print("Running PyColmap Poisson Meshing...")
options = pycolmap.PoissonMeshingOptions()
try:
    pycolmap.poisson_meshing("outputs/sample3/reconstruction/scene_sparse_normals.ply", "outputs/sample3/scene/scene_mesh.ply", options)
    print("Success!")
except Exception as e:
    print("Error:", e)
