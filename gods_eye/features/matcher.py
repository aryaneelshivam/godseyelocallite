import pycolmap
from pathlib import Path
from gods_eye.config.settings import Config
import logging

class FeatureMatcher:
    def __init__(self, config: Config, database_path: Path):
        self.config = config
        self.database_path = database_path
        
    def match_sequential(self):
        logging.info(f"Sequential matching on {self.database_path} with window {self.config.matching_window}")
        
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.use_gpu = False
        
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = self.config.matching_window
        pairing_options.quadratic_overlap = 0
        
        pycolmap.match_sequential(
            self.database_path,
            matching_options=matching_options,
            pairing_options=pairing_options
        )
