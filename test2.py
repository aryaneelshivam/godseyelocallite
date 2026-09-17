import torch
from transformers import pipeline
import pycolmap
from PIL import Image
p = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
p(Image.new("RGB", (224, 224)))
print("Test 2 OK")
