import torch
from transformers import pipeline
from PIL import Image

device = -1
print("Loading model...")
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
print("Model loaded.")

pil_img = Image.open("outputs/sample/keyframes/frame_00000.jpg").convert("RGB")
print("Predicting...")
depth_output = depth_estimator(pil_img)
print("Predicted.")
