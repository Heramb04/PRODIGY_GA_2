# Text-to-Image Generator with Stable Diffusion + Gradio

This project is a lightweight, fully local text-to-image generation app using **Stable Diffusion**, powered by **Gradio** for the front-end and hosted model weights from [Hugging Face](https://huggingface.co/). It’s designed for quick prototyping and customizable integration into larger AI pipelines.

## 🔧 Features

- Clean Gradio interface for prompt-based image generation  
- Loads custom Stable Diffusion weights from Hugging Face  
- Optional parameters for guidance scale, steps, and seed  
- Compatible across Windows, Linux, and WSL  
- Plug-and-play for AI workflows or creative apps  

### Running it locally:

A) bash
git clone https://github.com/Heramb04/PRODIGY_GA_2.git
cd (to the installation folder)
For creating a virtual environment:
python3 -m venv venv 
source venv\Scripts\activate
venv

Installing the required dependencies: 
pip install -r requirements.txt 
Then run the app by using:
python app.py

B) WINDOWS
git clone https://github.com/Heramb04/PRODIGY_GA_2.git
cd (to the installation folder)
For creating a virtual environment:
python -m venv venv 
venv\Scripts\activate
venv

Install the dependencies:
pip install -r requirements.txt

And run the app using
python app.py
