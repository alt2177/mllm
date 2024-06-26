#!/bin/bash

## Resource Requests
#SBATCH --job-name=mllm_fine_tuning # Job name
#SBATCH --nodes=1# Use one node
#SBATCH --account=austin_tao
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:8                   # Request one GPU
#SBATCH --output=output.log     # Standard output and error log

# run script via Makefile
#pip uninstall transformers
#pip install transformers==4.28.0
#pip install torch 
#pip install datasets
#pip install evaluate
#pip install numpy
#pip install huggingface
#pip install regex --upgrade
#pip install huggingface_hub
#git config user.email "mllm.dev.222@gmail.com"
#git config user.name "mllm-dev"
#SBATCH --mail-type=BEGIN,END,FAIL           # Send email on job END and FAIL
#SBATCH --mail-user=austin.tao@berkeley.edu 

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
python main.py

