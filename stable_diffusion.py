import os
os.environ["HF_HOME"] = "./hf_cache"

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id)

pipe = pipe.to("cpu")

prompts = [
    "anime girl with blue hair, fantasy style",
    "anime warrior with sword, dark theme",
    "anime cyberpunk city, neon lights"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=20).images[0]
    image.save(f"outputs/anime_output_{i}.png")

print("Image generated successfully!")