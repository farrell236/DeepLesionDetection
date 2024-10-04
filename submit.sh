#! /bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=/data/houbb/.logs/biowulf/FastRCNNPredictor.err
#SBATCH --output=/data/houbb/.logs/biowulf/FastRCNNPredictor.out
#SBATCH --time=10-00:00:00

source /data/houbb/_venv/python39/bin/activate
python main.py
