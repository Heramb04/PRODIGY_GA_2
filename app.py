import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

def load_pipeline():
    # Auto-detect any available GPU backend or fallback to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float16 if device.type != "cpu" else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    ).to(device)

    return pipe

# Initialize pipeline once
pipe = load_pipeline()

def generate(prompt: str, steps: int, scale: float):
    """Run the pipeline and return a PIL image."""
    out = pipe(prompt, num_inference_steps=steps, guidance_scale=scale)
    return out.images[0]

# Build and launch Gradio UI
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=1, placeholder="Enter prompt…", label="Prompt"),
        gr.Slider(1, 100, value=50, step=1, label="Inference Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance Scale"),
    ],
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion Image Generator",
    description="Generates images based on your prompt!."
)

if __name__ == "__main__":
    demo.launch(share=True)
