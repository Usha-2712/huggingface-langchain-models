# huggingface-langchain-models
creating ai-models and giving chosen prompting and setting up a Python environment, installing dependencies, configuring GPU usage, and running a transformer model with LangChain.

1. Create a Virtual Environment
Creating a virtual environment helps isolate dependencies and prevents conflicts with other Python projects.

For Windows (Command Prompt)
python -m venv langchain-env
langchain-env\Scripts\activate

2. Install Requirements
Once the virtual environment is activated, install the required dependencies.
pip install langchain transformers langchain-huggingface

3. If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch.

Run the following command (replacing cu126 with your CUDA version):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

To check which CUDA version you have installed, run:
nvcc --version

4. Check for GPU Availability
Run the following Python code to verify that your GPU is available:

import torch

#Check for GPU Availability
gpu_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU found"

print(f"GPU Available: {gpu_available}")
print(f"GPU Name: {device_name}")


Set Device in Pipeline
Once GPU availability is confirmed, specify the device in the transformer pipeline.

from transformers import pipeline

# Load the model and set device to GPU (device=0)
model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    device=0  # Use GPU (0 refers to the first GPU)
)

# Generate text
output = model("What is LangChain?")
print(output)
