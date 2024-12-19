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


#### Creating gradcam heatmaps

- `MODEL_FULL_NAME="simple_cnn_1_1"`
- `python gradcamheatmap.py --model_full_name $MODEL_FULL_NAME`

-  the data path can be specified (as parent folder of "atsds_large")
- `python gradcamheatmap.py --model_full_name simple_cnn_1_1 --data_base_path /home/ck/mnt/XAI-DIA-gl/Julian/Dataset_Masterarbeit`