import torch
import cv2
import numpy as np
import time
from utils import utils_image as util

def load_mobile_model(model_path='fbcnn_mobile.pt'):
    """Load TorchScript model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path)
    model = model.to(device)
    model.eval()
    return model, device

def process_image(model, image_path, strength=0.5, device='cpu'):
    """Process a single image with the mobile model."""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR uint8
    
    # Convert to tensor
    tensor = util.uint2tensor4(img).to(device)
    
    # Prepare strength input
    qf = torch.tensor([[strength]], device=device)
    
    # Inference
    start = time.time()
    with torch.no_grad():
        out_tensor, pred_qf = model(tensor, qf)
    elapsed = (time.time() - start) * 1000  # ms
    
    # Convert back to image
    out_np = util.single2uint(util.tensor2single(out_tensor))
    
    return out_np, elapsed, pred_qf.item()

def main():
    # Load model
    model, device = load_mobile_model()
    print(f"Model loaded on {device}")
    
    # Test parameters
    image_path = 'input.jpg'  # Change this to your test image
    strength = 0.3  # Change this to your desired strength (0.0 to 1.0)
    
    # Process image
    result, time_ms, pred_qf = process_image(model, image_path, strength, device)
    
    # Save result
    cv2.imwrite('result.png', result)
    print(f"Processing time: {time_ms:.1f}ms")
    print(f"Predicted quality factor: {pred_qf:.3f}")

if __name__ == '__main__':
    main()
