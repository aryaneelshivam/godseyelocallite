import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    extraction_fps: int
    features: int
    matching_window: int
    blur_threshold: float
    similarity_threshold: float

def load_config(mode: str = "FAST", config_path: str = "config.yaml") -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        
    modes = data.get("modes", {})
    if mode not in modes:
        mode = data.get("default_mode", "FAST")
        
    mode_config = modes.get(mode, {})
    
    return Config(
        extraction_fps=mode_config.get("extraction_fps", 1),
        features=mode_config.get("features", 2000),
        matching_window=mode_config.get("matching_window", 3),
        blur_threshold=data.get("blur_threshold", 100.0),
        similarity_threshold=data.get("similarity_threshold", 0.85)
    )
