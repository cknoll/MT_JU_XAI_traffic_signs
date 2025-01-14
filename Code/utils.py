import argparse
import os
from PIL import Image
from types import SimpleNamespace  # used as flexible Container Class

import numpy as np
import cv2

from dotenv import load_dotenv


# Returns the Image with the Mask as overlay.
def mask_on_image(mask,img,alpha=0.5):
    heatmap = get_rgb_heatmap(mask)
    img = img.squeeze()
    cam_on_img = (1-alpha)*img + alpha*heatmap
    return np.copy(cam_on_img)

def get_rgb_heatmap(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    return np.copy(heatmap)

#get a cutout based on a cutoff value
def get_cutoff_area(mask,img,cutoff = 0.5):
    for i in range(3):
        img[:,:,i] = np.where(mask>cutoff,img[:,:,i],0)
    return np.copy(img)

#get a cutout based on a percentage value.
def get_percentage_of_image(image,mask,percentage, fill_value = 0.0):
    masked_image = np.zeros_like(image)
    n = mask.size
    sortedvalues = np.sort(mask.flatten('K'))[::-1]

    index = int(n/100*percentage)
    index_2 = n//100*percentage
    cutoff = sortedvalues[index]
    for i in range(3):
        masked_image[:,:,i] = np.where(mask-cutoff>0.0,image[:,:,i],fill_value)
    return masked_image


def get_percentage_of_image_1d(image,mask,percentage, fill_value = 0.0):
    image = normalize_image(image)
    mask = normalize_image(mask)
    masked_image = np.zeros_like(image)
    n = mask.size
    sortedvalues = np.sort(mask.flatten('K'))[::-1]

    index = int(n/100*percentage)
    index_2 = n//100*percentage
    cutoff = sortedvalues[index]
    for i in range(3):
        masked_image = np.where(mask-cutoff>0.0,image,fill_value)
    return masked_image

def normalize_image(img):
    return np.nan_to_num((img-img.min())/(img.max()-img.min()), nan=0.0, posinf=0.0,neginf=0.0)

def get_input_tensors(image):
    return transform_test(image).unsqueeze(0)


def get_contained_part(mask1,mask2):
    mask1,mask2 = normalize_image(mask1),normalize_image(mask2)
    return np.array((mask1 == 1.0) & (mask2 == 1.0))

def get_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # for every argument we also have a short form
    parser.add_argument(
        "--model_full_name", "-n", type=str, required=True, help="Full model name (e.g., simple_cnn_1_1)"
    )
    parser.add_argument("--model_cp_base_path", "-cp", type=str, help="directory of model checkpoints", default=None)
    parser.add_argument("--data_base_path", "-d", type=str, help="data path", default=None)
    return parser

def generate_adversarial_examples(
    adv_folder,
    pct_range,
    categories,
    imagedict,
    img_path,
    background_dir,
    xai_dir,
    mask_condition
):
    """
    Generate adversarial examples based on the given mask condition.

    """
    for pct in pct_range:
        print(f"Processing percentage: {pct}%")
        for category in categories:
            output_dir = os.path.join(adv_folder, str(pct), "test", category)
            os.makedirs(output_dir, exist_ok=True)

            images = imagedict[category]
            for imagename in images:
                # Load original image and background
                current_img = normalize_image(np.array(Image.open(os.path.join(img_path, category, imagename))))
                current_background = normalize_image(np.array(Image.open(os.path.join(background_dir, category, imagename))))

                # Load and process XAI mask
                xai_mask = np.load(os.path.join(xai_dir, category, "mask", f"{imagename}.npy"))
                adv_mask = normalize_image(get_percentage_of_image(np.ones_like(current_img), xai_mask, pct / 10))

                # Create adversarial example using the mask condition
                adv_example = np.where(mask_condition(adv_mask), current_img, current_background)
                adv_example_save = Image.fromarray((adv_example * 255).astype('uint8'))

                # Save adversarial example
                adv_example_save.save(os.path.join(output_dir, imagename))


def create_image_dict(DATASET, DATASET_SPLIT):
    IMAGES_PATH = os.path.join('data', DATASET, DATASET_SPLIT)

    # Define our Categories
    CATEGORIES = sorted(os.listdir(IMAGES_PATH))

    imagedict = {}
    for cat in CATEGORIES:
        imagedict[cat] = []
        imagelist = os.listdir(os.path.join(IMAGES_PATH, cat))
        for im in imagelist:
            imagedict[cat].append(im)

    return CATEGORIES, imagedict


def read_conf_from_dotenv() -> SimpleNamespace:
    assert os.path.isfile(".env")
    load_dotenv()

    conf = SimpleNamespace()
    conf.BASE_DIR = os.getenv("BASE_DIR")

    assert conf.BASE_DIR is not None
    return conf


def get_dir_path(*parts, check_exists=True):
    path = os.path.join(*parts)

    if check_exists and not os.path.isdir(path):
        msg = f"Path {path} unexpectedly is not a directory"
        raise FileNotFoundError(msg)
    return path
