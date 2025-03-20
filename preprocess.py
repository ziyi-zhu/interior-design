import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


def apply_canny(img):
    """Apply Canny edge detection to an image"""
    img_tensor = F.to_tensor(img)
    img_np = (img_tensor.numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_canny = cv2.Canny(img_gray, 100, 200)
    # Convert back to RGB for consistency
    img_canny_rgb = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_canny_rgb)


def apply_blur(img):
    """Apply Gaussian blur to an image"""
    gaussian_blur = transforms.GaussianBlur(kernel_size=51)
    return gaussian_blur(img)


def get_depth_model(checkpoint_path):
    """Initialize and return the DepthFM model"""
    try:
        from depthfm.dfm import DepthFM
    except ImportError:
        raise ImportError(
            "Please install depthfm from https://github.com/CompVis/depth-fm"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for depth estimation")

    device = torch.device("cuda")
    model = DepthFM(ckpt_path=checkpoint_path).to(device)
    model.eval()
    return model, device


def apply_depth(img, model, device):
    """Apply depth estimation to an image using DepthFM"""
    img_tensor = F.to_tensor(img)
    c, h, w = img_tensor.shape
    # Add batch dimension before interpolating
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = torch.nn.functional.interpolate(
        img_tensor, (512, 512), mode="bilinear", align_corners=False
    )
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        depth = model(img_tensor, num_steps=2, ensemble_size=4)

    # Ensure depth has batch dimension before interpolating back
    depth = torch.nn.functional.interpolate(
        depth, (h, w), mode="bilinear", align_corners=False
    )
    depth = depth.squeeze(0)  # Remove batch dimension

    # Convert depth map to image
    depth_np = depth.squeeze(0).cpu().numpy()
    depth_np = (depth_np * 255).astype(np.uint8)
    return Image.fromarray(depth_np, mode="L")
