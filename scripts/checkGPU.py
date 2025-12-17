import torch

is_available = torch.cuda.is_available()
print(f"CUDA available: {is_available}")

if is_available:
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")

    current_device_name = torch.cuda.get_device_name(0)
    print(f"Current GPU name: {current_device_name}")