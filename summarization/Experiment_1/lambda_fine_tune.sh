#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning # Job name
#SBATCH --nodes=1# Use one node
#SBATCH --account=robert
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:1                   # Request one GPU
#SBATCH --output=output.log     # Standard output and error log

# Load any modules and set up your environment
#module load python/3.8
#module load cuda/10.1                 # Load CUDA module, if required

# run script via Makefile
pip uninstall transformers
pip install transformers==4.28.0
pip install torch 
pip install datasets
pip install evaluate
pip install numpy
pip install huggingface
pip install regex --upgrade
pip install huggingface_hub
git config user.email "mllm.dev.222@gmail.com"
git config user.name "mllm-dev"
echo $(pwd)
#python -m torch.distributed.launch --nproc_per_node=1 main.py
python main.py

