# Background
This repo contains code based on the master thesis "Evaluation of XAI-Algorithms Using Neural Networks" by J. Ullrich.


## Usage

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

- `MODEL_FULL_NAME="simple_cnn_1_1"`
- `python gradcamheatmap.py --model_full_name $MODEL_FULL_NAME`

-  the data path can be specified (as parent folder of "atsds_large")
- `python gradcamheatmap.py --model_full_name simple_cnn_1_1 --data_base_path /home/ck/mnt/XAI-DIA-gl/Julian/Dataset_Masterarbeit`

Output:
The masks(npy files) and images with masks are created as out put in '$data_base_path/auswertung/$MODEL_FULL_NAME/gradcam/test' folder.

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

