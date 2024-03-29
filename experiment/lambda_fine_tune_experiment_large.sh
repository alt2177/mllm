#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning # Job name
#SBATCH --nodes=1# Use one node
#SBATCH --account=sean_mcavoy
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:6                   # Request one GPU
#SBATCH --output=output_large.log     # Standard output and error log
#SBATCH --mail-type=END,FAIL           # Send email on job END and FAIL
#SBATCH --mail-user=sean_mcavoy@berkeley.edu 

# run script via Makefile
#pip uninstall transformers
#pip install transformers==4.28.0
#pip install accelerate -U


# Needed for main_experiment:
#pip install transformers
#pip install torch 
#pip install datasets
#pip install evaluate
#pip install numpy
#pip install huggingface
#pip install regex --upgrade
#pip install huggingface_hub
git config user.email "mllm.dev.222@gmail.com"
git config user.name "mllm-dev"
#git remote set-url origin https://huggingface.co/mllm-dev/yelp_finetuned_6gpu_full
python main_experiment_large.py
