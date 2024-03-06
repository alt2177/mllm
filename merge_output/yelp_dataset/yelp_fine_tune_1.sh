#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning_yelp # Job name
#SBATCH --nodes=1# Use one node
#SBATCH --account=phudish_p
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:6                   # Request one GPU
#SBATCH --output=output_1.log     # Standard output and error log
#SBATCH --mail-type=START,FAIL,END           # Send email on job END and FAIL
#SBATCH --mail-user=phudish_p@berkeley.edu 

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

#python -m torch.distributed.launch --nproc_per_node=6 main.py
python main.py
