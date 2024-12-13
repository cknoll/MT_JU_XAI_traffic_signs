#!/bin/bash

#SBATCH --job-name=gradcamjob1         # Job name
#SBATCH --output=gradcamjob1.log      # Output log file
#SBATCH --error=gradcamjob1_error.log        # Error log file
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH --time=06:00:00                      # Maximum run time (e.g., 2 hours)
#SBATCH --partition=alpha                    # Partition to run on
#SBATCH --ntasks=1


# Load necessary modules
module purge
module load release/24.04 GCCcore/12.2.0 Python/3.10.8 CUDA/11.7

# Activate your virtual environment
source ~/venv1/bin/activate

# Define the model arguments
MODEL_FULL_NAME="simple_cnn_1_1"


# Run the Python script 
python gradcamheatmap.py --model_full_name $MODEL_FULL_NAME