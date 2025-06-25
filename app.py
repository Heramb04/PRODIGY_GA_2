import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables (e.g. HF_TOKEN)
load_dotenv()

# Optional: Set HF_TOKEN if using wandb
if "HF_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Choose device and dtype based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)

# Inference function
def generate_image(prompt, steps, guidance):
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
    return image

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Image Generator")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="e.g. 'a steampunk robot in a lush jungle'")
        steps = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="Inference Steps")
        guidance = gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="Guidance Scale")
    run_button = gr.Button("Generate")
    output = gr.Image(type="pil", label="Result")

    run_button.click(fn=generate_image, inputs=[prompt, steps, guidance], outputs=output)

# Launch app
if __name__ == "__main__":
    demo.launch()