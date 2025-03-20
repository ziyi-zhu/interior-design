import os
from dataclasses import dataclass
from typing import Literal

import torch
from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

from preprocess import apply_blur, apply_canny


class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)

    def preprocess(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5
        return image

    def postprocess(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image


INSTRUCTION = """
This is an interior room image. Output a prompt for a stable diffusion model with ControlNet to enhance the space with new furnishings and decor:
For bedrooms/living rooms/offices: Thoughtfully place desk and bed while preserving the room's current architectural features. Add contemporary pieces and soft accents.
For kitchens: Update with modern appliances and decor elements that complement the existing layout and surfaces. Introduce coordinating accessories.
For bathrooms: Refresh with new fixtures and accessories that work with the current wall and floor finishes. Add tasteful decor touches.
Focus on furniture placement and decor choices that enhance the room's function. Keep the structural elements unchanged.
Only return the prompt and keep it under 50 words. Do not include any other text like "Here is the prompt" or anything else.
""".strip()


@dataclass
class DesignConfig:
    input_dir: str
    output_dir: str
    prompt: str
    negative_prompt: str
    instruction: str = INSTRUCTION
    model_name: str = "stabilityai/stable-diffusion-3.5-large"
    instruct_model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    transform_type: Literal["canny", "blur", "depth"] = "canny"
    width: int = 1024
    height: int = 672
    controlnet_conditioning_scale: float = 0.8
    guidance_scale: float = 3.5
    num_inference_steps: int = 60
    generator_seed: int = 0
    max_sequence_length: int = 77
    num_images_per_prompt: int = 2


class Designer:
    def __init__(self, config: DesignConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.controlnet = SD3ControlNetModel.from_pretrained(
            f"{config.model_name}-controlnet-{config.transform_type}",
            torch_dtype=torch.float16,
        )
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            config.model_name, controlnet=self.controlnet, torch_dtype=torch.float16
        ).to(self.device)

        self.model = MllamaForConditionalGeneration.from_pretrained(
            config.instruct_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(config.instruct_model_name)

        if config.transform_type == "canny":
            self.pipe.image_processor = SD3CannyImageProcessor()

        self.generator = torch.Generator(device=self.device).manual_seed(
            self.config.generator_seed
        )

    def preprocess(self, image: Image.Image):
        if self.config.transform_type == "canny":
            image = apply_canny(image)
        elif self.config.transform_type == "blur":
            image = apply_blur(image)
        return image

    def design(self):
        image_paths = [
            filename
            for filename in os.listdir(self.config.input_dir)
            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
            )
        ]
        os.makedirs(self.config.output_dir, exist_ok=True)

        for image_path in tqdm(image_paths, desc="Processing images"):
            full_image_path = os.path.join(self.config.input_dir, image_path)
            image = Image.open(full_image_path)
            image = self.preprocess(image)

            generated_images = self.generate(image)
            for i, generated_image in enumerate(generated_images):
                base_name = os.path.splitext(image_path)[0]
                output_path = os.path.join(
                    self.config.output_dir, f"{base_name}_variant_{i}.png"
                )
                generated_image.save(output_path)

    def generate_prompt(self, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.config.instruction},
                ],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs, max_new_tokens=self.config.max_sequence_length
        )
        generated_text = self.processor.decode(output[0])
        return generated_text[len(input_text) :].strip().rstrip("<|eot_id|>")

    def generate(self, image: Image.Image) -> list[Image.Image]:
        # prompt = self.generate_prompt(image)
        return self.pipe(
            prompt=self.config.prompt,
            control_image=image,
            width=self.config.width,
            height=self.config.height,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            generator=self.generator,
            max_sequence_length=self.config.max_sequence_length,
            num_images_per_prompt=self.config.num_images_per_prompt,
            negative_prompt=self.config.negative_prompt,
        ).images
