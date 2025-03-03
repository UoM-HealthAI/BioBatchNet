#!/bin/bash --login
#$ -cwd
#$ -pe smp.pe 8  # 请求 8 核 CPU
#$ -l h_vmem=32G  # 请求 32GB 内存
#$ -l v100 
#$ -j y  # 结合标准输出和错误输出

#$ -o ~/scratch/code/haiping/BioBatchNet_project/csf_log/run_baseline_log.txt # output directory
#$ -e ~/scratch/code/haiping/BioBatchNet_project/csf_log/run_baseline_error.txt   # error directory 


module load libs/cuda

SIF_PATH="/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/BioBatchNet-csf3ready.sif"
SCRIPT_PATH="run_baseline.py"
CONDA_ENV="BioBatchNet"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
   export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
   NVIDIAFLAG=--nv
fi

echo "Starting Singularity GPU job on $(hostname)"
echo "Running with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 使用容器内的conda
singularity exec $NVIDIAFLAG --bind /scratch,/mnt $SIF_PATH bash -c "
    export CONDA_PREFIX='/opt/miniconda'
    export PATH=\"\$CONDA_PREFIX/bin:\$PATH\"
    source \$CONDA_PREFIX/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    cd /scratch/w29632hl/code/haiping/BioBatchNet_project/Baseline  
    python $SCRIPT_PATH
"

echo "Job finished"
