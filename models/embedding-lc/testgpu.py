import torch

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")

# Optional: print the name of the GPU
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
