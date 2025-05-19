import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import os
from models.network_fbcnn import FBCNN as FBCNNModel
from typing import Tuple, Optional
import numpy as np
import sys

def check_environment():
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU for conversion")

# Configuration
n_channels = 3  # change to 1 for grayscale, 3 for color
nc = [64, 128, 256, 512]
nb = 4
model_path = 'model_zoo/fbcnn_color.pth'
output_path = 'fbcnn_color_mobile.ptl'  # Note: changed extension to .ptl

# Define a script-friendly wrapper
class FBCNNWrapper(torch.nn.Module):
    def __init__(self):
        super(FBCNNWrapper, self).__init__()
        self.model = FBCNNModel(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
        
    def forward(self, x: torch.Tensor, qf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, qf)

def convert_model():
    try:
        check_environment()
        # Instantiate and load weights
        device = torch.device('cpu')  # Use CPU for mobile conversion
        model = FBCNNWrapper()
        model.model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Create example inputs for tracing
        example_input = torch.randn(1, n_channels, 256, 256)
        example_qf = torch.tensor([[0.5]])  # Optional QF input

        # Script the model
        scripted_model = torch.jit.script(model)
        
        # Optimize for mobile
        optimized_model = optimize_for_mobile(scripted_model)
        
        # Save using lite interpreter
        optimized_model._save_for_lite_interpreter(output_path)
        print(f"Model saved to: {output_path}")

        loaded_model = torch.jit.load(output_path)
        test_output = loaded_model(example_input, example_qf)
        print("Model successfully loaded and tested!")
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    convert_model()
