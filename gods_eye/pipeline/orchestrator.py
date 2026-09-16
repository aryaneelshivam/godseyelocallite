import logging
from pathlib import Path
from typing import Optional

from gods_eye.config.settings import Config
from gods_eye.video.extractor import VideoExtractor
from gods_eye.video.keyframes import KeyframeSelector
from gods_eye.features.extractor import FeatureExtractor
from gods_eye.features.matcher import FeatureMatcher
from gods_eye.reconstruction.sfm import SparseReconstructor
from gods_eye.reconstruction.camera import export_camera_trajectory

class PipelineOrchestrator:
    def __init__(self, config: Config, video_path: str, output_root: str = "outputs"):
        self.config = config
        self.video_path = Path(video_path)
        
        # Setup output directories
        video_name = self.video_path.stem
        self.output_dir = Path(output_root) / video_name
        self.frames_dir = self.output_dir / "keyframes"
        self.features_dir = self.output_dir / "features"
        self.reconstruction_dir = self.output_dir / "reconstruction"
        self.scene_dir = self.output_dir / "scene"
        self.camera_traj_dir = self.output_dir / "camera_trajectory"
        
        for d in [self.frames_dir, self.features_dir, self.reconstruction_dir, self.scene_dir, self.camera_traj_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
    def run(self):
        print(f"[1/8] Extracting frames and [2/8] Selecting keyframes ...")
        self._extract_and_select_keyframes()
        
        print(f"[3/8] Extracting features ...")
        db_path = self._extract_features()
        
        print(f"[4/8] Sequential matching ...")
        self._match_features(db_path)
        
        print(f"[5/8] Sparse reconstruction ...")
        reconstruction, ply_path = self._reconstruct(db_path)
        
        if reconstruction:
            print(f"-> Exporting camera trajectory ...")
            export_camera_trajectory(reconstruction, self.camera_traj_dir / "camera_trajectory.json")
            
            # Copy or move the sparse ply to the scene directory as the final representation for Phase 1
            import shutil
            final_scene_path = self.scene_dir / "scene.ply"
            shutil.copy(ply_path, final_scene_path)
            print(f"-> Final sparse scene saved to {final_scene_path}")
            
        print("Pipeline Complete!")

    def _extract_and_select_keyframes(self):
        # Skip if keyframes exist (basic caching)
        if any(self.frames_dir.iterdir()):
            logging.info("Keyframes already exist, skipping extraction.")
            return
            
        extractor = VideoExtractor(self.video_path, self.config.extraction_fps)
        selector = KeyframeSelector(self.config, self.frames_dir)
        
        total_extracted = extractor.frame_count // extractor.frame_interval
        from tqdm import tqdm
        
        for frame_id, image, timestamp in tqdm(extractor.extract_frames(), total=total_extracted, desc="Extracting Keyframes", unit="frame"):
            selector.process_frame(frame_id, image, timestamp)
            
        extractor.release()
        logging.info(f"Selected {len(selector.selected_keyframes)} keyframes.")
        
    def _extract_features(self) -> Path:
        extractor = FeatureExtractor(self.config, self.features_dir)
        return extractor.extract(self.frames_dir)
        
    def _match_features(self, db_path: Path):
        matcher = FeatureMatcher(self.config, db_path)
        matcher.match_sequential()
        
    def _reconstruct(self, db_path: Path):
        reconstructor = SparseReconstructor(db_path, self.frames_dir, self.reconstruction_dir)
        return reconstructor.reconstruct()
