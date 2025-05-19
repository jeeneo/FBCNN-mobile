import torch
import os
from models.network_fbcnn import FBCNN as FBCNNModel
from typing import Tuple, Optional

# Configuration
n_channels = 1 # change to 1 for grayscale, 3 for color
nc = [64, 128, 256, 512]
nb = 4
model_path = 'model_zoo/fbcnn_gray_double.pth'  # Change if needed
output_path = 'fbcnn_gray_double.pt'

# Define a script-friendly wrapper
class FBCNNWrapper(torch.nn.Module):
    def __init__(self):
        super(FBCNNWrapper, self).__init__()
        self.model = FBCNNModel(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')

    def forward(self, x: torch.Tensor, qf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, qf)

# Instantiate and load weights
device = torch.device('cpu')  # Use CPU for scripting
model = FBCNNWrapper()
model.model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Direct scripting without tracing
with torch.no_grad():
    scripted_model = torch.jit.script(model)

# Save for mobile
scripted_model.save(output_path)
print(f"TorchScript model saved to: {output_path}")
