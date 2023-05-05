import torch
import tensorflow as tf


def is_gpu():
    print(f"Torch device: {torch.cuda.get_device_name()}")

