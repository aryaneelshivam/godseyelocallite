import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from gods_eye.config.settings import Config

@dataclass
class Keyframe:
    id: int
    image: np.ndarray
    blur_score: float
    timestamp: float

class KeyframeSelector:
    def __init__(self, config: Config, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selected_keyframes: List[Keyframe] = []
        
    def _compute_blur_score(self, image: np.ndarray) -> float:
        """Computes variance of Laplacian to estimate blur. Higher is sharper."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
        
    def _compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Computes a simple structural similarity or pixel-wise difference."""
        # Simple MSE or SSIM-like approach. For speed, resizing and MSE is fast.
        small1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (64, 64))
        small2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (64, 64))
        mse = np.mean((small1.astype(np.float32) - small2.astype(np.float32)) ** 2)
        # Normalize roughly (0 to 1), where 1 is identical. Max MSE is 255^2 = 65025
        similarity = 1.0 - (mse / 65025.0)
        return similarity

    def process_frame(self, frame_id: int, image: np.ndarray, timestamp: float) -> Optional[Keyframe]:
        blur_score = self._compute_blur_score(image)
        
        # Blur filter
        if blur_score < self.config.blur_threshold:
            return None
            
        # Similarity filter against the last selected keyframe
        if self.selected_keyframes:
            last_kf = self.selected_keyframes[-1]
            sim = self._compute_similarity(last_kf.image, image)
            if sim > self.config.similarity_threshold:
                # Too similar
                return None
                
        kf = Keyframe(id=frame_id, image=image, blur_score=blur_score, timestamp=timestamp)
        self.selected_keyframes.append(kf)
        
        # Save keyframe
        kf_path = self.output_dir / f"frame_{kf.id:05d}.jpg"
        cv2.imwrite(str(kf_path), image)
        
        return kf
