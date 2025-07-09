# STEP 1: Install packages (only if needed)
# pip install diffusers transformers accelerate torch

from diffusers import StableDiffusionPipeline
import torch

# STEP 2: Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# STEP 3: Input prompt
prompt = input("Enter your image prompt: ")

# STEP 4: Generate image
image = pipe(prompt).images[0]

# STEP 5: Show and save image
image.show()
image.save("generated_image.png")
