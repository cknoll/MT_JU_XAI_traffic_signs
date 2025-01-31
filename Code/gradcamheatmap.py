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
from utils import  get_default_arg_parser, mask_on_image, prepare_categories_and_images, setup_environment, read_conf_from_dotenv

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
        # if argument is not passed: use hardcoded default
        BASE_DIR = conf.BASE_DIR
    else:
        BASE_DIR = args.data_base_path

    if args.model_cp_base_path is None:
        # if argument is not passed: use hardcoded default  
        CHECKPOINT_PATH = conf.MODEL_DIR
    else:
        CHECKPOINT_PATH = args.model_cp_base_path

    IMAGES_PATH = os.path.join(BASE_DIR, dataset_type, dataset_split)
    output_path = os.path.join(BASE_DIR, "XAI_results", model_name, "gradcam/", dataset_split)

    # This creates the needed folders inside. 
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Setup environment
    device = setup_environment(random_seed)

    testset = ATSDS(root=BASE_DIR, split=dataset_split, dataset_type=dataset_type, transform=transform_test)

    #################

    model = get_model(model_name, n_classes = testset.get_num_classes())
    model = model.to(device)
    model.eval()
    loss_criterion = nn.CrossEntropyLoss()

    loss_criterion = loss_criterion.to(device)
    optimizer = optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,200)

    epoch,trainstats = load_model(model, optimizer, scheduler, CHECKPOINT_PATH + model_cpt, device)
    print(f"Model checkpoint loaded. Epoch: {epoch}")
    
    # Prepare categories and images
    CATEGORIES, label_idx_dict, imagedict = prepare_categories_and_images(IMAGES_PATH)

    output_types = ["mask","mask_on_image"]
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

    # Define what percentage cutouts you want.
    # If you are only interested in the masks you can just make this an empty list []

    class_to_dataset_class_dict = {}
    for cat in CATEGORIES:
        class_to_dataset_class_dict[cat] = cat

        for outputs in output_types:
            os.makedirs(os.path.join(output_path, cat, outputs), exist_ok=True)

    for cat in class_to_dataset_class_dict:
        model.eval()
        images = imagedict[cat]
        for imagename in images:
            fpath = os.path.abspath(os.path.join(IMAGES_PATH, cat, imagename))
            with open(fpath, 'rb') as f:
                with Image.open(f) as current_image:
                    current_image_tensor = get_input_tensors(current_image)
                    #print(imagename, class_to_dataset_class_dict[image_class])
                # These values are only used for the example pictures, the pipeline values are below them.
                current_image_tensor = current_image_tensor.cuda()
                shape = (np.array(current_image).shape[0],np.array(current_image).shape[1])
                original_mask, _ = get_gradcam(model,GRADCAM_TARGET_LAYER,current_image_tensor,label_idx_dict[class_to_dataset_class_dict[cat]],shape)
                image = np.copy(current_image)
                mask = np.copy(original_mask)
                np.save(os.path.join(output_path, cat, "mask", imagename), mask)
                overlay_image = (mask_on_image(normalize_image(mask),normalize_image(image),alpha=0.3)*255).astype(np.uint8)
                save_overlay_image = Image.fromarray(overlay_image)
                save_overlay_image.save(os.path.join(output_path, cat, "mask_on_image", imagename), "PNG")

if __name__ == "__main__":
    main()