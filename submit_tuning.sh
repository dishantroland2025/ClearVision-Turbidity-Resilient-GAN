#!/bin/bash
#SBATCH --job-name=CV_Tune_100
#SBATCH --output=logs/tune_%j.out
#SBATCH --error=logs/tune_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1          # Request 1 GPU (or change if allowed more)
#SBATCH --cpus-per-task=4     # 4 CPU cores
#SBATCH --mem=32G             # 32GB RAM
#SBATCH --time=48:00:00       # Request 48 hours to cover all 100 trials

# 1. Load Environment
# Update this line to match your cluster's module system
module load cuda/11.7
source activate my_env

# 2. Prepare Directories
mkdir -p logs

# 3. Run
echo "Starting 100-Trial Anchored Tuning on $(hostname)"
python tune_ClearVision.py