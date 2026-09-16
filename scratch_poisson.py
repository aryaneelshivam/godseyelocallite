import pymeshlab

print("Loading point cloud...")
ms = pymeshlab.MeshSet()
ms.load_new_mesh("outputs/sample3/scene/scene.ply")

print("Computing normals...")
ms.compute_normal_for_point_clouds(k=15, smoothiter=0)

print("Running Poisson Surface Reconstruction...")
# Screened Poisson Surface Reconstruction
ms.generate_surface_reconstruction_screened_poisson(depth=8, preclean=True)

print("Exporting mesh...")
ms.save_current_mesh("outputs/sample3/scene/scene_mesh.ply")
print("Done!")
