# from diffusers import AutoPipelineForText2Image
# import torch

# pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev').to('cpu')
# pipeline.load_lora_weights('Nishitbaria/Anime-style-flux-lora-Large', weight_name='lora.safetensors')
# image = pipeline('an anm Create a peaceful village scene with rolling green hills at sunset. Show a young woman in simple clothes painting on a canvas near a small cottage. Include warm lighting and soft colors to create a cozy atmosphere').images[0]
# image.save('image.jpg')

import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import os
os.environ["HF_TOKEN"] = "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"

# torch.cuda.set_per_process_memory_fraction(0.5, 0)

pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cpu")
n=5
image = load_image(
    f"/proj/zendnn/rjauhari/personal/img1/{n}.jpg"
)
prompt = "Pan across the gallery scene, showing various reactions of people viewing the artwork. Focus on Maya's gentle smile and the appreciation on viewers' faces. Add subtle gallery ambient sounds and soft background music."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=161,
    num_inference_steps=50,
).frames[0]
export_to_video(video, f"/proj/zendnn/rjauhari/personal/{n}.mp4", fps=24)
