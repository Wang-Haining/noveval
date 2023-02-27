#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=hw56@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=0-0:30:00
#SBATCH --mem=64gb
#SBATCH --partition gpu-debug
#SBATCH --mail-type=ALL
#SBATCH --job-name=gpt2
#SBATCH --output=/N/scratch/hw56/ppl_nanoGPT/output.log
#SBATCH --error=/N/scratch/hw56/ppl_nanoGPT/error.log

######  Module commands #####
module load python/3.10.5
module load cudatoolkit/11.7

######  Job commands go below this line #####
# load venv
source .ppl/bin/activate

# python scripts
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_wikipedia_en.py
