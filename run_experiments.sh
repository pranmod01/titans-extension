#!/bin/bash
#SBATCH --job-name=titans
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

source /insomnia001/home/pm3361/titans-extension/venv/bin/activate

python -m multi_signal_titans.experiments --experiment 1
# python -m multi_signal_titans.experiments --experiment 2
# python -m multi_signal_titans.experiments --experiment 3
# python -m multi_signal_titans.experiments --experiment 4
