## Standard libraries
import os

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CUDA_LAUNCH_BLOCKING=1

## PyTorch
import torch
from torchvision import transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# our own modules

# Import Gradcam - Check the gradcam.py for the code.
from ATSDS import ATSDS
from gradcam import get_gradcam
from model import get_model, load_model
from utils import  get_default_arg_parser, prepare_categories_and_images, setup_environment, read_conf_from_dotenv, save_xai_outputs, create_output_directories

try:
    # some optional debugging helpers
    from ipydex import IPS, activate_ips_on_exception
    activate_ips_on_exception()
except ImportError:
    pass

transform_test = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def normalize_image(img):
    return np.nan_to_num((img-img.min())/(img.max()-img.min()))

def get_input_tensors(image):
    return transform_test(image).unsqueeze(0)

def generate_gradcam_visualizations(model: torch.nn.Module, device: torch.device, categories: list[str],
                                     imagedict: dict[str, list[str]], label_idx_dict: dict[str, int],
                                     output_path: str, images_path: str, target_layer: torch.nn.Module) -> None:
    """
    Generate Grad-CAM visualizations for each image in the dataset and save them.

    Args:
        model (torch.nn.Module): The model to be used for generating Grad-CAM visualizations.
        device (torch.device): The device to run the model on (GPU or CPU).
        categories (list): List of categories in the dataset.
        imagedict (dict): A dictionary of image filenames for each category.
        label_idx_dict (dict): A dictionary mapping category names to indices.
        output_path (str): Path where Grad-CAM results will be saved.
        images_path (str): Path to the dataset images.
        target_layer (torch.nn.Module): The target layer for Grad-CAM.
    """
    for category in categories:
        model.eval()
        images = imagedict[category]
        for image_name in images:
            with Image.open(os.path.join(images_path, category, image_name)) as img:
                image_tensor = transform_test(img).unsqueeze(0).to(device)
                shape = img.size[::-1]  # PIL uses (width, height)

                mask, _ = get_gradcam(model, target_layer, image_tensor, label_idx_dict[category], shape)
                save_xai_outputs(mask, np.array(img), category, image_name, output_path)

def main():

    parser = get_default_arg_parser()
    args = parser.parse_args()

    # Changable Parameters
    model_name = "_".join(args.model_full_name.split("_")[:-2])
    model_cpt = args.model_full_name + ".tar"
   
    dataset_type = args.dataset_type
    dataset_split = args.dataset_split
    random_seed = args.random_seed

    conf = read_conf_from_dotenv()

    if args.data_base_path is None:
        # from .env
        BASE_DIR = conf.BASE_DIR
    else:
        BASE_DIR = args.data_base_path

    if args.model_cp_base_path is None:
        # from .env
        CHECKPOINT_PATH = conf.MODEL_DIR
    else:
        CHECKPOINT_PATH = args.model_cp_base_path

    IMAGES_PATH = os.path.join(BASE_DIR, dataset_type, dataset_split)
    output_path = os.path.join(BASE_DIR, "XAI_results", model_name, "gradcam", dataset_split)

    # Setup environment
    device = setup_environment(random_seed)

    testset = ATSDS(root=BASE_DIR, split=dataset_split, dataset_type=dataset_type, transform=transform_test)

    model = get_model(model_name, n_classes = testset.get_num_classes())
    model = model.to(device)
    model.eval()
    loss_criterion = nn.CrossEntropyLoss()

    loss_criterion = loss_criterion.to(device)
    optimizer = optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,200)

    epoch,trainstats = load_model(model, optimizer, scheduler, os.path.join(CHECKPOINT_PATH, model_cpt), device)
    print(f"Model checkpoint loaded. Epoch: {epoch}")
    
    ##############

    # print(model) # confirm gradcam layer if necessary
    if model_name == "simple_cnn":
        GRADCAM_TARGET_LAYER = model.conv3 # Simple CNN
    elif model_name == "resnet50":
        GRADCAM_TARGET_LAYER = model.layer4[-1].conv3
    elif model_name == "convnext_tiny":
        GRADCAM_TARGET_LAYER = model.features[-1][-1].block[0]
    elif model_name == "vgg16":
        GRADCAM_TARGET_LAYER = model.features[-3]

    print(GRADCAM_TARGET_LAYER)

    # Prepare categories and images
    categories, label_idx_dict, imagedict = prepare_categories_and_images(IMAGES_PATH)

    # Ensure output directories exist
    create_output_directories(output_path, categories)

    # Run Grad-CAM visualization
    generate_gradcam_visualizations(
        model, device, categories, imagedict, label_idx_dict, 
        output_path, IMAGES_PATH, GRADCAM_TARGET_LAYER
    )

if __name__ == "__main__":
    main()