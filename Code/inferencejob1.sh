#!/bin/bash

#SBATCH --job-name=inferencejob1         # Job name
#SBATCH --output=inferencejob1.log      # Output log file
#SBATCH --error=inferencejob1.log        # Error log file
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH --mem=16G                            # Memory
#SBATCH --time=04:00:00                      # Maximum run time (e.g., 2 hours)
#SBATCH --partition=alpha                    # Partition to run on
#SBATCH --cpus-per-gpu=4

# Load necessary modules
module purge
module load release/24.04 GCCcore/12.2.0 Python/3.10.8 CUDA/11.7

# Activate your virtual environment
source ~/venv1/bin/activate

# Define the model arguments
MODEL_FULL_NAME="simple_cnn_1_1"


# Run the Python script 
python inference_testimages_copy.py
python inference.py --model_full_name $MODEL_FULL_NAME
