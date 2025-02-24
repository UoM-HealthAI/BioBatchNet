# pothole_submit.sh
#!/bin/bash --login
#$ -cwd
#$ -l v100           # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)
#$ -N eva_domand       # job name
#$ -pe smp.pe 8      # 4 CPU cores available to the host code
                     # Can use up to 12 CPUs with an A100 GPU.

# This will upload your current environment to the working node                      
#$ -V                # load current environment

#$ -o ~/scratch/code/haiping/BioBatchNet_project/csf_log/run_imc_log.txt # output directory
#$ -e ~/scratch/code/haiping/BioBatchNet_project/csf_log/run_imc_error.txt   # error directory 

# load extra environments if needed (but if -V, normally don't have to)
# module load libs/nvidia-hpc-sdk/23.9
module load apps/binapps/anaconda3/2022.10
source activate BioBatchNet

# Run your project
python ../BioBatchNet/imc.py -c config_imc.yaml
