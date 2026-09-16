import pycolmap

print("Running PyColmap Poisson Meshing...")
options = pycolmap.PoissonMeshingOptions()
# We might need to adjust options, e.g. depth
try:
    pycolmap.poisson_meshing("outputs/sample3/reconstruction/scene_sparse.ply", "outputs/sample3/scene/scene_mesh.ply", options)
    print("Success!")
except Exception as e:
    print("Error:", e)
