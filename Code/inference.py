import os
import shutil
import argparse
import glob
import json
import tqdm

# shortcut
pjoin = os.path.join

# Personal debug module (`pip install ipydex`)
from ipydex import IPS

from PIL import Image
import torch
from torchvision import transforms
from model import get_model

# Transformation for inference images
transform_inference = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Function to load the model
def load_model(filepath, model, device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)  # Load to CPU or GPU
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["epoch"], checkpoint["trainstats"]


def to_list(tensor):
    res = tensor.cpu().squeeze().tolist()
    if isinstance(res, list):
        res2 = [round(elt, 3) for elt in res]
    else:
        res2 = res
    return res2


# Function to predict class for an image
def predict_image(model, image_path, class_names, full_res=False):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_inference(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    if full_res:
        return {
            "outputs": to_list(outputs),
            "predicted": to_list(predicted),
            "class": class_names[predicted.item()],
        }
    else:
        return class_names[predicted.item()]


def get_image_paths(base_path, sub_path="test"):
    """
    return a list of absolute paths of all images e.g. from train/ or test/
    """

    host_path = pjoin(base_path, sub_path)
    class_dirs = glob.glob(pjoin(host_path, "*"))

    all_paths = []

    for class_dir in class_dirs:
        image_paths_for_class = glob.glob(pjoin(class_dir, "*"))
        image_paths_for_class.sort()
        all_paths.extend(image_paths_for_class)

    return all_paths


# Function to organize images into class folders
def organize_images(model, input_folder, output_folder, class_names):
    # Clear the output folder at the beginning
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Remove all contents in the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Create class folders
    for class_name in class_names:
        class_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(input_folder, filename)
            predicted_class = predict_image(model, file_path, class_names)
            dest_folder = os.path.join(output_folder, predicted_class)
            shutil.copy(file_path, os.path.join(dest_folder, filename))
            print(f"Copied {filename} to {dest_folder}")


def classify_with_json_result(model, data_base_path, class_names):

    all_img_paths = get_image_paths(data_base_path)
    result_dict = {}
    path_start_idx = len(data_base_path) + 1

    if not "horse" in data_base_path:
        all_img_paths = tqdm.tqdm(all_img_paths)

    for image_path in all_img_paths:
        res = predict_image(model, image_path, class_names, full_res=True)

        # short_path = "test/00001/000000.png"
        short_path = image_path[path_start_idx:]
        train_test_dir, class_dir, fname = short_path.split(os.path.sep)
        boolean_result = (class_dir == res["class"])
        res["boolean_result"] = boolean_result
        result_dict[short_path] = res

    json_fpath = "results.json"
    with open(json_fpath, "w") as fp:
        json.dump(result_dict, fp, indent=2)

    print(f"file written: {json_fpath}")


def main(model_full_name, data_base_path=None, mode="copy"):

    if data_base_path is None:
        # Hardcoded path for HPC
        data_base_path = "/data/horse/ws/knoll-traffic_sign_reproduction/atsds_large"

    # Derive model path and model name
    filepath = "model/" + model_full_name + ".tar"
    model_name = "_".join(model_full_name.split("_")[:-2])  # Extract model_name

    # Define class names
    class_names = [f"{i:05d}" for i in range(1, 20)]  # e.g., 00001 to 00019

    # Initialize the model
    model = get_model(model_name=model_name, n_classes=len(class_names)).to(device)

    # Load model weights
    model, epoch, train_stats = load_model(filepath, model, device)
    model.eval()  # Set model to evaluation mode
    print(f"Loaded model: {model_name} | Epoch: {epoch}")

    if mode == "json":
        classify_with_json_result(model, data_base_path, class_names)
    else:
        # this is the original mode
        # mode == "copy"

        input_folder = os.path.join(data_base_path, "inference/images_to_classify")
        output_folder = os.path.join(data_base_path, "inference/classified_images")

        # Organize images into class folders
        organize_images(model, input_folder, output_folder, class_names)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # for every argument we also have a short form
    parser.add_argument(
        "--model_full_name", "-n", type=str, required=True, help="Full model name (e.g., simple_cnn_1_1)"
    )
    parser.add_argument("--data_base_path", "-d", type=str, help="data path", default=None)
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help=(
            "mode: 'copy' (default) or 'json'. Mode 'json' means a) the files are read from "
            "their original class-subdirs and b) the result is `results.json` and not a directory full of files"
        ),
        default="copy",
    )
    args = parser.parse_args()

    # Check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run the main function
    main(model_full_name=args.model_full_name, data_base_path=args.data_base_path, mode=args.mode)
