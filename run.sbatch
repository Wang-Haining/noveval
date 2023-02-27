#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=hw56@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=2-0:0:00
#SBATCH --mem=32gb
#SBATCH --partition gpu
#SBATCH --mail-type=ALL
#SBATCH --job-name=br200_gpt2
#SBATCH --output=/N/scratch/hw56/ppl_nanoGPT/torchrun-$(date +%s).err
#SBATCH --error=/N/scratch/hw56/ppl_nanoGPT/torchrun-$(date +%s).out

######  Module commands #####
module load python/3.10.5
module load cudatoolkit/11.7 

######  Job commands go below this line #####
# load venv
source .ppl/bin/activate

# python scripts
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_wikipedia_en.py
