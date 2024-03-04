#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning # Job name
#SBATCH --nodes=1                      # Use one node
#SBATCH --partition=jsteinhardt        # Specify the partition to run on
#SBATCH --nodelist=saruman
#SBATCH --gres=gpu:1                   # Request one GPU
#SBATCH --output=my_gpu_job_%j.log     # Standard output and error log
#SBATCH --mail-type=BEGIN, END, FAIL           # Send email on job END and FAIL
#SBATCH --mail-user= austin.tao@berkeley.edu, phudish_p@berkeley.edu, sean_mcavoy@berkeley.edu, robert_thompson@berkeley.edu 

# Load any modules and set up your environment
module load python/3.8
module load cuda/10.1                 # Load CUDA module, if required

# run script via Makefile
make run
