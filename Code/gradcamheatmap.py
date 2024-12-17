
from ATSDS import ATSDS
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as transforms
## Standard libraries
import os
import json
import math
import random
import numpy as np 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CUDA_LAUNCH_BLOCKING=1

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import argparse
import cv2

import matplotlib.image
from PIL import Image

# Import Gradcam - Check the gradcam.py for the code.
from gradcam import get_gradcam

parser = argparse.ArgumentParser()
parser.add_argument("--model_full_name", type=str, required=True, help="Full model name (e.g., simple_cnn_1_1)")
args = parser.parse_args()


# Changable Parameters
# model_name = "simple_cnn"
# model_cpt = model_name + "_1_1.tar"
model_name = "_".join(args.model_full_name.split("_")[:-2])
model_cpt = args.model_full_name + ".tar"

# DATASET_PATH = "data"
DATASET_PATH = "/data/horse/ws/knoll-traffic_sign_reproduction"
dataset_type = "atsds_large"
dataset_split = "test"

CHECKPOINT_PATH = "model/"

transform_test_crop = transforms.Compose(
    [transforms.Resize(256),
    transforms.CenterCrop(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


transform_test = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_model(model,optimizer,scheduler,filepath):
    cpt = torch.load(filepath,map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(cpt['model'])
    optimizer.load_state_dict(cpt['optimizer'])
    scheduler.load_state_dict(cpt['scheduler'])
    return cpt['epoch'], cpt['trainstats']
    
RANDOM_SEED = 1414

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Used for reproducability to fix randomness in some GPU calculations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

testset = ATSDS(root=DATASET_PATH, split=dataset_split, dataset_type=dataset_type, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = True, num_workers = 2)
import torchvision
print(torchvision.__file__)

from model import get_model

model = get_model(model_name, n_classes = testset.get_num_classes())
model = model.to(device)
model.eval()
loss_criterion = nn.CrossEntropyLoss()

loss_criterion = loss_criterion.to(device)
optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,200)
running_loss = 0
total = 0
correct = 0
save_osc = 1
epoch = 0
trainloss = []
trainacc = []

epoch,trainstats = load_model(model, optimizer, scheduler, "model/" + model_cpt)
train_loss = trainstats[0]
test_loss = trainstats[1]
detailed_stats = trainstats[2]

plt.plot(np.divide(train_loss,450), label = "train")
plt.plot(test_loss, label = "test")
plt.legend()

# IMAGES_PATH = 'data/' + dataset_type + '/' + dataset_split + '/'
IMAGES_PATH = '/data/horse/ws/knoll-traffic_sign_reproduction/' + dataset_type + '/' + dataset_split + '/'

# Define our Categories
CATEGORIES = sorted(os.listdir(IMAGES_PATH))
class_to_dataset_class_dict = {}
for cat in CATEGORIES:
    class_to_dataset_class_dict[cat] = cat
label_idx_dict = {}
for count,cat in enumerate(CATEGORIES):
    label_idx_dict[cat] = count

imagedict = {}
for cat in CATEGORIES:
    imagedict[cat] = []
    imagelist = os.listdir(IMAGES_PATH + cat + "/")
    for im in imagelist:
        imagedict[cat].append(im)            
output_types = ["mask","mask_on_image"]     




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
def get_percentage_of_image(image,mask,percentage):
    masked_image = np.zeros_like(image)
    n = mask.size
    sortedvalues = np.sort(mask.flatten('K'))[::-1]
    
    index = int(n/100*percentage)
    index_2 = n//100*percentage
    cutoff = sortedvalues[index]
    for i in range(3):
        masked_image[:,:,i] = np.where(mask-cutoff>0,image[:,:,i],0)
    return masked_image

def normalize_image(img):
    return np.nan_to_num((img-img.min())/(img.max()-img.min()))

def get_input_tensors(image):
    return transform_test(image).unsqueeze(0)


print(model) # confirm gradcam layer if necessary
if model_name == "simple_cnn":
    GRADCAM_TARGET_LAYER = model.conv3 # Simple CNN
elif model_name == "resnet50":
    GRADCAM_TARGET_LAYER = model.layer4[-1].conv3
elif model_name == "convnext":
    GRADCAM_TARGET_LAYER = model.features[-1][-1].block[0]
elif model_name == "vgg16":
    GRADCAM_TARGET_LAYER = model.features[-3]
    
# GRADCAM_TARGET_LAYER = model.features[-1][-1].block[0] # convnext
#GRADCAM_TARGET_LAYER = model.features[-3] # VGG
#GRADCAM_TARGET_LAYER = model.layer4[-1].conv3 # resnet50
print(GRADCAM_TARGET_LAYER)

# Make sure the Output_path exists.
# Define what percentage cutouts you want. 
# If you are only interested in the masks you can just make this an empty list []


# IMAGES_PATH = 'data/' + dataset_type + '/' + dataset_split + '/'
IMAGES_SUFFIX = '.png'
# output_path = 'data/auswertung/' + model_name + "/" + "gradcam/" + dataset_split + "/"
output_path = '/data/horse/ws/knoll-traffic_sign_reproduction/auswertung/' + model_name + "/" + "gradcam/" + dataset_split + "/"


# This creates the needed folders inside. As mentioned above the Folder defined in output_path has to already exist


if not os.path.isdir(output_path):
    os.makedirs(output_path)

for cat in CATEGORIES:
    for outputs in output_types:
        if not os.path.isdir(output_path + cat + "/" + outputs):
            os.makedirs(output_path + cat + "/" + outputs)


for cat in class_to_dataset_class_dict:
    model.eval()
    images = imagedict[cat]
    for imagename in images:
        with open(os.path.abspath(IMAGES_PATH + cat + "/" + imagename), 'rb') as f:
            with Image.open(f) as current_image:
                current_image_tensor = get_input_tensors(current_image)
                #print(imagename, class_to_dataset_class_dict[image_class])
            # These values are only used for the example pictures, the pipeline values are below them.
            current_image_tensor = current_image_tensor.cuda()
            shape = (np.array(current_image).shape[0],np.array(current_image).shape[1])
            original_mask, _ = get_gradcam(model,GRADCAM_TARGET_LAYER,current_image_tensor,label_idx_dict[class_to_dataset_class_dict[cat]],shape)
            image = np.copy(current_image)
            mask = np.copy(original_mask)
            np.save(output_path + cat +  "/mask/" + imagename, mask)
            overlay_image = (mask_on_image(normalize_image(mask),normalize_image(image),alpha=0.3)*255).astype(np.uint8)
            save_overlay_image = Image.fromarray(overlay_image)
            save_overlay_image.save(output_path + cat + "/mask_on_image/" + imagename, "PNG")