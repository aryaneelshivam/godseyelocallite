import argparse
import logging
import os
from gods_eye.config.settings import load_config
from gods_eye.pipeline.orchestrator import PipelineOrchestrator

def interactive_prompt(args):
    print("====================================")
    print("          GOD'S EYE CLI")
    print("====================================")
    
    # 1. Ask for filename
    mp4_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if mp4_files:
        print("\nAvailable video files:")
        for i, f in enumerate(mp4_files, 1):
            print(f"{i}. {f}")
        
        while True:
            val = input(f"Select a file (1-{len(mp4_files)}) or type path: ").strip()
            if val.isdigit() and 1 <= int(val) <= len(mp4_files):
                args.input = mp4_files[int(val)-1]
                break
            elif val:
                args.input = val
                break
    else:
        while True:
            val = input("Enter path to input drone video: ").strip()
            if val:
                args.input = val
                break
        
    # 2. Ask for type
    print("\nSelect reconstruction quality mode:")
    print("1. FAST (Low Density, Very Quick)")
    print("2. BALANCED (Medium Density)")
    print("3. QUALITY (High Density, Slower)")
    val = input("Select [1-3] (default 1): ").strip()
    mode_map = {"1": "FAST", "2": "BALANCED", "3": "QUALITY"}
    args.mode = mode_map.get(val, "FAST")
    
    # 3. Ask for tweakable params
    print("\nSelect internal resolution limit (controls how many pixels SIFT can analyze):")
    print("1. Low (1024px) - Safest for low RAM")
    print("2. Medium (1920px) - Good default")
    print("3. Max (3840px / 4K) - Extreme detail")
    val = input("Select [1-3] (default 3): ").strip()
    res_map = {"1": 1024, "2": 1920, "3": 3840}
    args.resolution = res_map.get(val, 3840)
    
    print("\n")
    return args

def main():
    parser = argparse.ArgumentParser(description="God's Eye: Monocular Drone Video to 3D Environment")
    parser.add_argument("--input", type=str, help="Path to the input drone video")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory root")
    parser.add_argument("--mode", type=str, choices=["FAST", "BALANCED", "QUALITY"], help="Reconstruction quality mode")
    parser.add_argument("--resolution", type=int, help="Max image resolution for SIFT")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Enter interactive mode if no input is provided
    if not args.input:
        args = interactive_prompt(args)
    
    if not args.mode:
        args.mode = "FAST"
    if not args.resolution:
        args.resolution = 3840
        
    import pycolmap
    if args.debug:
        level = logging.DEBUG
        pycolmap.logging.minloglevel = 0
    else:
        level = logging.INFO
        pycolmap.logging.minloglevel = 2
        
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = load_config(mode=args.mode)
    config.max_image_size = args.resolution
    
    logging.info(f"Loaded config mode: {args.mode} (Resolution limit: {args.resolution}px)")
    
    orchestrator = PipelineOrchestrator(config, args.input, args.output)
    
    # Prompt for meshing
    if not args.input:
        print("\nDo you want to generate a photorealistic solid Mesh (Poisson)?")
        print("This takes a few minutes, but creates a solid video-game style surface.")
        ans = input("Enable Poisson Meshing? (y/n, default n): ").strip().lower()
        run_mesh = ans == 'y'
    else:
        run_mesh = False
        
    orchestrator.run()
    
    if run_mesh:
        import subprocess
        print("\n=== Running Poisson Surface Reconstruction ===")
        mesh_in = orchestrator.scene_dir / "scene.ply"
        mesh_out = orchestrator.scene_dir / "scene_mesh.ply"
        subprocess.run(["python3", "-m", "gods_eye.reconstruction.mesh", "--input", str(mesh_in), "--output", str(mesh_out)])
        print("Meshing Complete!")

if __name__ == "__main__":
    main()
