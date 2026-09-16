import cv2
import os
from pathlib import Path
from typing import Generator, Tuple
import numpy as np

class VideoExtractor:
    def __init__(self, video_path: str, target_fps: float = 1.0):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.original_fps if self.original_fps > 0 else 0
        
        # Calculate frame skip interval
        if target_fps >= self.original_fps or target_fps <= 0:
            self.frame_interval = 1
        else:
            self.frame_interval = int(round(self.original_fps / target_fps))

    def extract_frames(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """Yields (frame_index, frame_image, timestamp) based on the target FPS interval."""
        frame_idx = 0
        extracted_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_idx % self.frame_interval == 0:
                timestamp = frame_idx / self.original_fps if self.original_fps > 0 else 0.0
                yield extracted_idx, frame, timestamp
                extracted_idx += 1
                
            frame_idx += 1
            
    def release(self):
        self.cap.release()
