#!/bin/bash
#SBATCH --job-name=titans_synthetic
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/synthetic_%j.out
#SBATCH --error=logs/synthetic_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

set -e
mkdir -p logs

source /insomnia001/home/pm3361/titans-extension/venv/bin/activate

cd /insomnia001/home/pm3361/titans-extension

# Generate datasets if they don't exist yet
if [ ! -f synthetic_tasks/data/knowledge_update_train.json ]; then
    echo "Generating synthetic datasets..."
    python synthetic_tasks/generate_tasks.py --seed 42
fi

# Full evaluation: all 3 tasks × 3 variants
# Checkpoints are saved so you can resume if the job is preempted.
python synthetic_tasks/eval_synthetic.py \
    --max_steps 2000 \
    --batch_size 4 \
    --seq_len 1024 \
    --log_every 100 \
    --eval_every 250 \
    --seed 42 \
    --checkpoint_dir ./synthetic_tasks/checkpoints

echo "Done. Results in synthetic_tasks/results/"
