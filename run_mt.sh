#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --job-name=mt
#SBATCH --output=mt.out
#SBATCH --account=rrg-mageed
#SBATCH --mail-user=zcy94@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
module load cuda cudnn
source ~/bert_multitask/bin/activate

python3 mt_org.py
