#!/bin/bash --login
#$ -cwd
#$ -pe smp.pe 8  # 请求 8 核 CPU
#$ -l h_vmem=32G  # 请求 32GB 内存
#$ -l v100 
#$ -j y  # 结合标准输出和错误输出
#$ -N run_pan_mac
#$ -o ~/scratch/code/haiping/BioBatchNet_project/csf_log/run_baseline/run_baseline_log_gene1.txt # output directory
#$ -e ~/scratch/code/haiping/BioBatchNet_project/csf_log/run_baseline/run_baseline_error_gene1.txt    # error directory 

module load libs/cuda

SIF_PATH="/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/BioBatchNetV2.sif"
SCRIPT_PATH="run_baseline.py"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
   export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
   NVIDIAFLAG=--nv
fi

echo "Starting Singularity GPU job on $(hostname)"
echo "Running with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

singularity exec --cleanenv $NVIDIAFLAG --bind /scratch,/mnt $SIF_PATH bash -c "

    # 或者可以设置 CONDA_ENVS_PATH 指向容器内环境路径
    export CONDA_ENVS_PATH=/opt/miniconda/envs

    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate scvi
    
    echo '=== Container environment paths (after activating scvi): ==='
    which python
    which conda
    python --version
    conda --version
    conda env list
    echo '=================================='
    
    cd /scratch/w29632hl/code/haiping/BioBatchNet_project/Baseline  
    python $SCRIPT_PATH
"

