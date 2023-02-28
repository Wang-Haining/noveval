#!/bin/bash
##
## gpuexample.sbatch submit a job using a GPU
##
## Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
## All other lines are read by the shell.
##
#SBATCH --account=priority-jasonclark	# priority account to use
#SBATCH --job-name=nanogpt2           # job name
#SBATCH --partition=gpupriority         # queue partition to run the job in
#SBATCH --nodes=1                       # number of nodes to allocate
#SBATCH --ntasks-per-node=1             # number of descrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=8		# number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --gpus-per-task=2		# number of GPUs to allocate - all GPUs are currently A40 model
#SBATCH --mem=64000                     # 2000 MB of Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=14-00:00:00               # Maximum job run time
#SBATCH --output=/home/g91p721/ppl_nanoGPT/torchrun-$(date+%s).out 	# standard output file (%j = jobid)
#SBATCH --error=/home/g91p721/ppl_nanoGPT/torchrun-$(date+%s).err 	# standard error file
#SBATCH --mail-user=hw56@iu.edu	# email address to recieve job updates
#SBATCH --mail-type=ALL			# conditions to recieve emial notifications for job
## Run 'man sbatch' for more information on the options above.
module load Python/3.10.4-GCCcore-11.3.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate .ppl
torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2_wikipedia_en.py
