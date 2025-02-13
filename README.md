# Background
This repo contains code based on the master thesis "Evaluation of XAI-Algorithms Using Neural Networks" by J. Ullrich.


## Usage


### General Notes on Paths

Many scripts and notebooks in this repo depend on paths. To ensure that the code runs on different machines (local development machines, HPC, etc) we use a `.env` file. This file is machine-specific and is expected to define the necessary paths in environment variables.

Example:

```.env
BASE_DIR="/home/username/axaiev/data
```

This file is evaluated by `utils.read_paths_from_dotenv()`. Note: The package `opencv-python` has to be installed (see `requirements.txt`)


The expected path structure is as follows:

```
<BASE_DIR>                      specified in .env file
├── atsds_large/
│   ├── test/
│   │   ├── 0001/               class directory
│   │   │   ├── 000000.png      individual image of this class
│   │   │   └── ...             more images
│   │   └── ...                 more classes
│   └── train/
│       └── <class dirs with image files>
│
├── atsds_large_background/...  background images with same structure
│                               as in atsds_large (test/..., train/...)
│
├── atsds_large_mask/...        corresponding mask images with same structure
│                               as in atsds_large (test/..., train/...)
├── model_checkpoints/
│   ├── convnext_tiny_1_1.tar
│   ├── resnet50_1_1.tar
│   ├── simple_cnn_1_1.tar
│   └── vgg16_1_1.tar
│
├── XAI_evaluation
│   ├── simple_cnn/gradcam/test/    same structure as `XAI_results`
│   │   ├── revelation
│   │   └── occlusion
│   └── ...                     other XAI methods and models
│
└── XAI_results
    ├── simple_cnn/             cnn model directory
    │   ├── gradcam/            xai method
    │   │   ├── test/           split fraction (train/test)
    │   │   │   ├── mask/
    │   │   │   │   ├── 000000.png.npy
    │   │   │   │   └── ...
    │   │   │   ├── mask_on_image/
    │   │   │   │   ├── 000000.png
    │   │   │   │   └── ...
    │   …   …   …
    ├── vgg16/...
    ├── resnet50/..
    ├── convnext_tiny/..
```


### Unittests

To ensure the environment is setup as expected run `pytest -s Code/tests.py`.

### `inference.py`

### Examples

#### Inference

Local usage:
- `python inference.py py3 --model_full_name simple_cnn_1_1 -d ./`

Usage with mounted group drive and json-mode (see `--help`) :
- `python inference.py --model_full_name simple_cnn_1_1 -cp /home/ck/mnt/XAI-DIA-gl/Julian/Dataset_Masterarbeit/model_checkpoints -d /home/ck/mnt/XAI-DIA-gl/Julian/Dataset_Masterarbeit/atsds_large -m json`

Output:
The output images are created within the class folders in the path 'inference/classified_images'


#### Creating gradcam heatmaps

- `python gradcamheatmap.py --model_full_name simple_cnn_1_1`

-  the data path  (as parent folder of "atsds_large") and model path can be specified in the .env file(BASE_DIR=data
MODEL_DIR=model).

Output:
The masks (npy files) and images with masks are created as out put in '$data_base_path/auswertung/$MODEL_FULL_NAME/gradcam/test' folder.

#### Creating Prism heatmaps

- `python PRISM_pipeline.py --model_full_name simple_cnn_1_1` 

-  the data path  (as parent folder of "atsds_large") and model path can be specified in the .env file(BASE_DIR=data
MODEL_DIR=model).

Output:
The masks (npy files) and images with masks are created as out put in '$data_base_path/XAI_results/$MODEL_FULL_NAME/prism/test' folder.

#### Creating Lime heatmaps

- `python lime_pipeline.py --model_full_name simple_cnn_1_1` 

-  the data path  (as parent folder of "atsds_large") and model path can be specified in the .env file(BASE_DIR=data
MODEL_DIR=model).

Output:
The masks (npy files) and images with masks are created as out put in '$data_base_path/XAI_results/$MODEL_FULL_NAME/lime/test' folder.

#### Creating Integrated Gradients heatmaps

- `python int_g_pipeline.py --model_full_name simple_cnn_1_1` 

-  the data path  (as parent folder of "atsds_large") and model path can be specified in the .env file(BASE_DIR=data
MODEL_DIR=model).


Output:
The masks (npy files) and images with masks are created as out put in '$data_base_path/XAI_results/$MODEL_FULL_NAME/ig/test' folder.

#### Creating Xrai heatmaps

- `python Xrai_pipeline.py --model_full_name simple_cnn_1_1` 

-  the data path  (as parent folder of "atsds_large") and model path can be specified in the .env file(BASE_DIR=data
MODEL_DIR=model).

Output:
The masks (npy files) and images with masks are created as out put in '$data_base_path/XAI_results/$MODEL_FULL_NAME/xrai/test' folder.

#### Occlusion

Steps to follow:
- First, produce gradcam or other XAI method masks by running the respective scripts(example gradcamheatmap.py).
- Next, copy 'atsds_large_background' and 'atsds_large_mask' folders to the base data path.
- Run the first four cells of Create_Occlusion_Dataset.ipynb.
    -  In the second cell change the MODEL_TYPE, XAI_TYPE, BASE_DIR, XAI_DIR, ADV_FOLDER accordingly.
- Now the occlusion dataset is created in the ADV_FOLDER.
- Run the first nine cells of Auswertung_Occlusion.ipynb.
    -  In the second cell, change the MODEL_TYPE, XAI_NAME, CHECKPOINT_PATH accordingly.
    -  In fourth cell, change the list of xai_methods for plot accordingly.
- Now the plot for the selected xai methods for the selected model is displayed.

#### Revelation

Steps to follow:
- First, produce gradcam or other XAI method masks by running the respective scripts(example gradcamheatmap.py).
- Next, copy 'atsds_large_background' and 'atsds_large_mask' folders to the base data path.
- Run the first four cells of Create_Revelation_Dataset.ipynb.
    -  In the second cell change the MODEL_TYPE, XAI_TYPE, BASE_DIR, XAI_DIR, ADV_FOLDER accordingly.
- Now the Revelation dataset is created in the ADV_FOLDER.
- Run the first nine cells of Auswertung_Occlusion.ipynb.
    -  In the second cell, change the MODEL_TYPE, XAI_NAME, CHECKPOINT_PATH accordingly.
    -  In fourth cell, change the list of xai_methods for plot accordingly.
- Now the plot for the selected xai methods for the selected model is displayed.
