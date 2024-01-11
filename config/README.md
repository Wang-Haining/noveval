## Config



### Note
The original run was on Big Red 200. For a rigorous reproduction, please consider the alignments with BR200's Slurm
modules  
```bash
# on Big Red 200, with `module list`
module load ~/.module/default

#Currently Loaded Modules:
#  1) craype-x86-rome          5) gcc/11.2.0          9) cray-libsci/21.08.1.2  13) cudatoolkit/11.7
#  2) libfabric/1.11.0.3.71    6) craype/2.7.14      10) PrgEnv-gnu/8.3.2       14) python/3.10.5
#  3) craype-network-ofi       7) cray-dsmml/0.2.2   11) xalt/2.10.34           
#  4) perftools-base/21.12.0   8) cray-mpich/8.1.14  12) git/2.34
```

The sbatch files are located in `run` folder.

The GPT-2 was trained in a transition period from Pytorch 1.0 to 2.0, some issues are reported on the introduction of 
flash attention. We did encounter the problem and followed the suggestion of setting AdamW's eps to 1e-5.
See discussion https://github.com/karpathy/nanoGPT/issues/137#issuecomment-1463268225. 