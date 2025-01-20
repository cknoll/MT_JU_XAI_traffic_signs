### Standard libraries
import argparse
import os
import random

### 3rd party libraries
import numpy as np
import cv2
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

## PyTorch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

## Local libraries
from utils import setup_environment, prepare_categories_and_images, create_output_directories, save_xai_outputs, load_checkpoint, normalize_image, get_rgb_heatmap
from ATSDS import ATSDS
from model import get_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Integrated Gradients Visualization Pipeline")

    # Configuration variables
    parser.add_argument('--model_name', type=str, default="simple_cnn", help="Name of the model.")
    parser.add_argument('--model_checkpoint', type=str, default="model/simple_cnn_1_1.tar", help="Path to the model checkpoint.")
    parser.add_argument('--dataset_path', type=str, default="data", help="Path to the dataset.")
    parser.add_argument('--dataset_type', type=str, default="atsds_large", help="Type of the dataset.")
    parser.add_argument('--dataset_split', type=str, default="test", help="Dataset split (e.g., 'train', 'test').")
    parser.add_argument('--images_path', type=str, default="data/atsds_large/test", help="Path to the images.")
    parser.add_argument('--output_path', type=str, default="data/auswertung/", help="Path to save outputs.")
    parser.add_argument('--random_seed', type=int, default=1414, help="Random seed for reproducibility.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for data loader.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of images for IG explanation.")

    return parser.parse_args()


def compute_ig_masks(model, device, categories, imagedict, label_idx_dict, output_path, images_path):
    """Generate Integrated Gradients visualizations for each image in the dataset."""
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    for category in categories:
        images = imagedict[category]
        for image_name in images:
            image_path = os.path.join(images_path, category, image_name)
            with open(image_path, 'rb') as f:
                with Image.open(f) as current_image:
                    current_image_tensor = TRANSFORM_TEST(current_image).unsqueeze(0).to(device)
                    current_image_tensor.requires_grad = True

                    # Set baseline as a tensor of zeros
                    baseline = torch.zeros_like(current_image_tensor)

                    # Perform Integrated Gradients computation
                    attr_ig = ig.attribute(current_image_tensor, baselines=baseline, target=label_idx_dict[category])
                    attr_ig = attr_ig.squeeze().detach().cpu().numpy()

                    # Convert to visualization format
                    ig_mask = np.sum(attr_ig, axis=0)  # Aggregate across channels
                    ig_mask = normalize_image(ig_mask)

                    # Overlay IG mask on original image
                    overlay_image = np.array(current_image).astype(np.float32) / 255.0
                    mask_on_image_result = mask_on_image_ig(ig_mask, overlay_image, alpha=0.3)

                    # Create output directories if they do not exist
                    mask_output_dir = os.path.join(output_path, category, 'mask')
                    overlay_output_dir = os.path.join(output_path, category, 'mask_on_image')
                    os.makedirs(mask_output_dir, exist_ok=True)
                    os.makedirs(overlay_output_dir, exist_ok=True)

                    # Save IG mask and overlay image
                    mask_output_path = os.path.join(mask_output_dir, image_name.replace('.jpg', '.npy'))
                    overlay_output_path = os.path.join(overlay_output_dir, image_name)
                    np.save(mask_output_path, ig_mask)
                    Image.fromarray((mask_on_image_result * 255).astype(np.uint8)).save(overlay_output_path, "PNG")


def mask_on_image_ig(mask, img, alpha=0.5):
    # Ensure the mask and image have the same dimensions
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Generate heatmap from the mask
    heatmap = get_rgb_heatmap(mask)

    # Squeeze image if it has extra dimensions
    if len(img.shape) == 4 and img.shape[0] == 1:  # Batch size of 1
        img = img.squeeze()

    # Normalize the image to [0, 1] if it's not already
    if img.max() > 1:
        img = img.astype(np.float32) / 255

    # Blend the heatmap and image
    cam_on_img = (1 - alpha) * img + alpha * heatmap
    return np.copy(cam_on_img)

def main():
    # Parse command-line arguments
    args = parse_args()

    # Transforms
    global TRANSFORM_TEST
    TRANSFORM_TEST = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Setup environment
    device = setup_environment(args.random_seed)

    # Load dataset and dataloader
    testset = ATSDS(root=args.dataset_path, split=args.dataset_split, dataset_type=args.dataset_type, transform=TRANSFORM_TEST)

    # Load model
    model = get_model(args.model_name, n_classes=testset.get_num_classes()).to(device)
    model.eval()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Load checkpoint
    epoch, trainstats = load_checkpoint(args.model_checkpoint, model, optimizer, scheduler, device)
    print(f"Model checkpoint loaded. Epoch: {epoch}")

    # Prepare categories and images
    categories, label_idx_dict, imagedict = prepare_categories_and_images(args.images_path)

    # Ensure output directories exist
    output_path = os.path.join(args.output_path, args.model_name, "ig", "test/")
    create_output_directories(output_path, categories)

    # Generate Integrated Gradients visualizations
    compute_ig_masks(model, device, categories, imagedict, label_idx_dict, output_path, args.images_path)


if __name__ == "__main__":
    main()
