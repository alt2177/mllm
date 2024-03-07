#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning # Job name
#SBATCH --nodes=2# Use one node
#SBATCH --account=robert_thompson
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:8                   # Request one GPU
#SBATCH --output=output.log     # Standard output and error log
#SBATCH --mail-type=END,FAIL           # Send email on job END and FAIL
#SBATCH --mail-user=robert_thompson@berkeley.edu 
<<<<<<< HEAD

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
<<<<<<< HEAD
#git config user.email "mllm.dev.222@gmail.com"
#git config user.name "mllm-dev"
=======

# run script via Makefile
<<<<<<< HEAD
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

python -m torch.distributed.launch --nproc_per_node=4 main.py
#python main.py
>>>>>>> 817d4eeb6 (starting to parallelize with torch)
=======
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
>>>>>>> 4f63ef5a3 (main is working well with pushing)

#python -m torch.distributed.launch --nproc_per_node=4 main.py
=======
git config user.email "mllm.dev.222@gmail.com"
git config user.name "mllm-dev"
#git remote set-url origin https://huggingface.co/mllm-dev/yelp_finetuned_6gpu_full
>>>>>>> 3ff4795f6 (add comments)
python main.py
