import torch
from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline
from diffusers.utils import load_image

controlnet = SD3ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
).to("cuda")

control_image = load_image("outputs/preprocessed/1.jpg")
prompt = "A luxurious, high-end, and modern interior design with a touch of luxury and elegance."

generator = torch.Generator(device="cpu").manual_seed(0)
image = pipe(
    prompt,
    control_image=control_image,
    guidance_scale=3.5,
    num_inference_steps=60,
    generator=generator,
    max_sequence_length=77,
    width=1024,
    height=672,
).images[0]
image.save("canny-8b.jpg")
