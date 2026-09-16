import pycolmap
from pathlib import Path
import logging

class SparseReconstructor:
    def __init__(self, database_path: Path, image_dir: Path, output_dir: Path):
        self.database_path = database_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def reconstruct(self):
        logging.info(f"Starting incremental mapping from {self.database_path}")
        
        mapper_options = pycolmap.IncrementalPipelineOptions()
        
        reconstructions = pycolmap.incremental_mapping(
            database_path=self.database_path,
            image_path=self.image_dir,
            output_path=self.output_dir,
            options=mapper_options
        )
        
        if not reconstructions:
            logging.error("Failed to reconstruct any models.")
            return None, None
            
        logging.info(f"Successfully reconstructed {len(reconstructions)} models.")
        
        # Typically the first reconstruction is the largest/best
        # Pycolmap returns a dict mapping model_id to Reconstruction object
        best_model_id = max(reconstructions.keys(), key=lambda k: len(reconstructions[k].images))
        best_reconstruction = reconstructions[best_model_id]
        
        # Export as PLY
        ply_path = self.output_dir / "scene_sparse.ply"
        best_reconstruction.export_PLY(ply_path)
        logging.info(f"Exported sparse point cloud to {ply_path}")
        
        return best_reconstruction, ply_path
