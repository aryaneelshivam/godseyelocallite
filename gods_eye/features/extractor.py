import pycolmap
from pathlib import Path
from gods_eye.config.settings import Config
import logging

class FeatureExtractor:
    def __init__(self, config: Config, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.database_path = output_dir / "database.db"
        
    def extract(self, image_dir: Path):
        logging.info(f"Extracting features from {image_dir} to {self.database_path}")
        
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.use_gpu = False
        extraction_options.max_image_size = 3840
        extraction_options.num_threads = 2
        extraction_options.sift.max_num_features = self.config.features
        extraction_options.sift.peak_threshold = 0.002
        extraction_options.sift.edge_threshold = 16.0
        
        if self.database_path.exists():
            logging.info("Database already exists. Features may already be extracted.")
            # If we want to support true resuming, we might just return here if it's completely done.
            # For now, pycolmap will skip existing images in the database.
            
        pycolmap.extract_features(
            self.database_path,
            image_dir,
            extraction_options=extraction_options
        )
        return self.database_path
