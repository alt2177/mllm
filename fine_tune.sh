#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning # Job name
#SBATCH --nodes=1                      # Use one node
#SBATCH --partition=jsteinhardt        # Specify the partition to run on
#SBATCH --nodelist=saruman
#SBATCH --gres=gpu:1                   # Request one GPU
<<<<<<< HEAD
<<<<<<< HEAD
#SBATCH --output=output.log     # Standard output and error log
<<<<<<< HEAD
#SBATCH --mail-type=BEGIN,END,FAIL           # Send email on job END and FAIL
#SBATCH --mail-user=mllm.dev.222@gmail.com
=======
#SBATCH --mail-type=END,FAIL           # Send email on job END and FAIL
#SBATCH --mail-user=robert_thompson@berkeley.edu
=======
#SBATCH --output=my_gpu_job_%j.log     # Standard output and error log
#SBATCH --mail-type=BEGIN, END, FAIL           # Send email on job END and FAIL
#SBATCH --mail-user= austin.tao@berkeley.edu, phudish_p@berkeley.edu, sean_mcavoy@berkeley.edu, robert_thompson@berkeley.edu 
>>>>>>> 82f519b2f (created sbatch file)
<<<<<<< HEAD
>>>>>>> 7888da94f (created sbatch file)
=======
=======
#SBATCH --output=output.log     # Standard output and error log
#SBATCH --mail-type=END,FAIL           # Send email on job END and FAIL
#SBATCH --mail-user=austin.tao@berkeley.edu,phudish_p@berkeley.edu,sean_mcavoy@berkeley.edu,robert_thompson@berkeley.edu 
>>>>>>> 9e63d322e (debugging fine_tune.sh)
>>>>>>> cbe69ee22 (debugging fine_tune.sh)

# Load any modules and set up your environment
module load python/3.8
module load cuda/10.1                 # Load CUDA module, if required

# run script via Makefile
<<<<<<< HEAD
<<<<<<< HEAD
pip uninstall transformers
pip install transformers==4.28.0
pip install torch
#pip install datasets
#pip install evaluate
#pip install numpy
pip install huggingface
pip install regex --upgrade
pip install huggingface_hub
git config user.email "mllm.dev.222@gmail.com"
git config user.name "mllm-dev"
echo $(pwd)
#python -m torch.distributed.launch --nproc_per_node=1 main.py
<<<<<<< HEAD
python3 ./main.py
=======
python main.py
=======
make run
>>>>>>> 82f519b2f (created sbatch file)
<<<<<<< HEAD
>>>>>>> 7888da94f (created sbatch file)
=======
=======
pip uninstall transformers
pip install transformers==4.28.0
pip install torch 
pip install datasets
pip install evaluate
pip install numpy
pip install huggingface
echo $(pwd)
python3 ./main.py
#make run
>>>>>>> 9e63d322e (debugging fine_tune.sh)
>>>>>>> cbe69ee22 (debugging fine_tune.sh)
