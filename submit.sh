#! /bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=/data/houbb/.logs/biowulf/retinanet_objectdet_3.err
#SBATCH --output=/data/houbb/.logs/biowulf/retinanet_objectdet_3.out
#SBATCH --time=10-00:00:00

module load CUDA/11.8 cuDNN/8.9.2/CUDA-11
source /data/houbb/_venv/python39/bin/activate

export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/CUDA/11.8.0/'
export WANDB_API_KEY='fddfd0cc8be3b16c26971c6536b4cfce1c8e6d17'

python main.py
