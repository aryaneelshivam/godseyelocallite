import argparse
import logging
from gods_eye.config.settings import load_config
from gods_eye.pipeline.orchestrator import PipelineOrchestrator

def main():
    parser = argparse.ArgumentParser(description="God's Eye: Monocular Drone Video to 3D Environment")
    parser.add_argument("--input", type=str, required=True, help="Path to the input drone video")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory root")
    parser.add_argument("--mode", type=str, default="FAST", choices=["FAST", "BALANCED", "QUALITY"], help="Reconstruction quality mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = load_config(mode=args.mode)
    logging.info(f"Loaded config mode: {args.mode}")
    
    orchestrator = PipelineOrchestrator(config, args.input, args.output)
    orchestrator.run()

if __name__ == "__main__":
    main()
