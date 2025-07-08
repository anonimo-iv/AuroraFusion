import platform
import sys
import threading
import time
from os import PathLike
from pathlib import Path
from typing import NamedTuple, Optional


def device_widget(default="AUTO", exclude=None, added=None, description="Device:"):
    import openvino as ov
    import ipywidgets as widgets

    core = ov.Core()

    supported_devices = core.available_devices + ["AUTO"]
    exclude = exclude or []
    if exclude:
        for ex_device in exclude:
            if ex_device in supported_devices:
                supported_devices.remove(ex_device)

    added = added or []
    if added:
        for add_device in added:
            if add_device not in supported_devices:
                supported_devices.append(add_device)

    device = widgets.Dropdown(
        options=supported_devices,
        value=default,
        description=description,
        disabled=False,
    )
    return device


from pathlib import Path

supported_model_ids = [
    "tensorart/stable-diffusion-3.5-medium-turbo",
    "stabilityai/stable-diffusion-3.5-large-turbo",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-3-medium-diffusers",
]


def get_pipeline_options(default_value=(supported_model_ids[0], False)):
    import ipywidgets as widgets

    model_selector = widgets.Dropdown(options=supported_model_ids, value=default_value[0])

    load_t5 = widgets.Checkbox(
        value=default_value[1],
        description="Use t5 text encoder",
        disabled=False,
    )

    to_compress = widgets.Checkbox(
        value=True,
        description="Weight compression",
        disabled=False,
    )

    pt_pipeline_options = widgets.VBox([model_selector, load_t5, to_compress])
    return pt_pipeline_options, model_selector, load_t5, to_compress


def init_pipeline_without_t5(model_dir, device):
    import openvino_genai as ov_genai

    model_path = Path(model_dir)

    scheduler = ov_genai.Scheduler.from_config(model_path / "scheduler/scheduler_config.json")
    text_encoder = ov_genai.CLIPTextModelWithProjection(model_path / "text_encoder", device)
    text_encoder_2 = ov_genai.CLIPTextModelWithProjection(model_path / "text_encoder_2", device)
    transformer = ov_genai.SD3Transformer2DModel(model_path / "transformer", device)
    vae = ov_genai.AutoencoderKL(model_path / "vae_decoder", device=device)

    ov_pipe = ov_genai.Text2ImagePipeline.stable_diffusion_3(scheduler, text_encoder, text_encoder_2, transformer, vae)
    return ov_pipe

def init_pipeline(models_dict, device, use_flash_lora=False):
    import openvino_genai as ov_genai

    transformer = ov_genai.SD3Transformer2DModel(models_dict["transformer"], device)
    text_encoder = ov_genai.CLIPTextModelWithProjection(models_dict["text_encoder"], device)
    text_encoder_2 = ov_genai.CLIPTextModelWithProjection(models_dict["text_encoder_2"], device)
    vae_decoder = ov_genai.AutoencoderKL(models_dict["vae"], device=device)
    scheduler = ov_genai.Scheduler.from_config(models_dict["scheduler"] / "scheduler_config.json")

    if "text_encoder_3" in models_dict:
        text_encoder_3 = ov_genai.T5TextEncoder(models_dict["text_encoder_3"], device)
        ov_pipe = ov_genai.Text2ImagePipeline.stable_diffusion_3(
            scheduler,
            text_encoder,
            text_encoder_2,
            text_encoder_3,
            transformer,
            vae_decoder,
        )
    else:
        ov_pipe = ov_genai.Text2ImagePipeline.stable_diffusion_3(
            scheduler,
            text_encoder,
            text_encoder_2,
            transformer,
            vae_decoder,
        )

    return ov_pipe


# Model paths
TRANSFORMER_PATH = "/lus/flare/projects/hp-ptycho/binkma/models/sd3.5/transformer/"
VAE_DECODER_PATH = "/lus/flare/projects/hp-ptycho/binkma/models/sd3.5/vae_decoder/"
TEXT_ENCODER_PATH = "/lus/flare/projects/hp-ptycho/binkma/models/sd3.5/text_encoder/"
TEXT_ENCODER_2_PATH = "/lus/flare/projects/hp-ptycho/binkma/models/sd3.5/text_encoder_2/"
TEXT_ENCODER_3_PATH = "/lus/flare/projects/hp-ptycho/binkma/models/sd3.5/text_encoder_3/"
SCHEDULER_PATH = "/lus/flare/projects/hp-ptycho/binkma/models/sd3.5/scheduler/"

pt_pipeline_options, use_flash_lora, load_t5, _ = get_pipeline_options()

MODELS_BASE = Path("/lus/flare/projects/hp-ptycho/binkma/models/sd3.5")

models_dict = {
    "transformer": MODELS_BASE / "transformer",
    "vae": MODELS_BASE / "vae_decoder",
    "text_encoder": MODELS_BASE / "text_encoder",
    "text_encoder_2": MODELS_BASE / "text_encoder_2",
    "scheduler": MODELS_BASE / "scheduler",
}

if load_t5.value:
    models_dict["text_encoder_3"] = TEXT_ENCODER_3_PATH

device = device_widget()

# Initialize the pipeline
ov_pipe = init_pipeline(models_dict, device.value, use_flash_lora.value)

# FIXED: Generate image without torch.Generator and handle guidance_scale/negative_prompt compatibility
guidance_scale_value = 5 if not use_flash_lora.value else 0

# Prepare generation parameters
generation_params = {
    "prompt": "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
    "num_inference_steps": 28 if not use_flash_lora.value else 4,
    "guidance_scale": guidance_scale_value,
    "height": 512,
    "width": 512,
}

# Only include negative_prompt when guidance_scale > 1.0
if guidance_scale_value > 1.0:
    generation_params["negative_prompt"] = ""

# image = ov_pipe.generate(**generation_params).images[0]
result = ov_pipe.generate(**generation_params)

# Option 2: If you need reproducible results, set seed differently
# Some OpenVINO GenAI versions support 'seed' parameter directly:
# image = ov_pipe.generate(
#     prompt="A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
#     negative_prompt="",
#     num_inference_steps=28 if not use_flash_lora.value else 4,
#     guidance_scale=5 if not use_flash_lora.value else 0,
#     height=512,
#     width=512,
#     seed=141,  # Use integer seed instead of torch.Generator
# ).images[0]

# Save or display the image
# image.save("generated_image.png")
# print("Image generated successfully!")

import numpy as np
from PIL import Image

# Handle the tensor conversion
if hasattr(result, 'images'):
    # Some versions might still return an object with .images
    image = result.images[0]
elif isinstance(result, list):
    # If it returns a list of images
    image = result[0]
else:
    # Convert OpenVINO tensor to numpy array first
    image_array = np.array(result.data).reshape(result.shape)
    
    # Handle batch dimension if present
    if len(image_array.shape) == 4:
        # Remove batch dimension [batch, height, width, channels] -> [height, width, channels]
        image_array = image_array[0]
    
    # Ensure values are in 0-255 range
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    # Ensure proper channel ordering (should be H, W, C for PIL)
    if image_array.shape[0] == 3:  # If channels first (C, H, W)
        image_array = np.transpose(image_array, (1, 2, 0))  # Convert to (H, W, C)
    
    # Create PIL Image
    image = Image.fromarray(image_array)

# Option 2: If you need reproducible results, set seed differently
# Some OpenVINO GenAI versions support 'seed' parameter directly:
# image = ov_pipe.generate(
#     prompt="A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
#     negative_prompt="",
#     num_inference_steps=28 if not use_flash_lora.value else 4,
#     guidance_scale=5 if not use_flash_lora.value else 0,
#     height=512,
#     width=512,
#     seed=141,  # Use integer seed instead of torch.Generator
# ).images[0]

# Save or display the image
image.save("generated_image.png")
print("Image generated successfully!")