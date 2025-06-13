#!/bin/bash
#SBATCH --job-name=marloes_ablation
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ablation_%A.out
#SBATCH --error=logs/ablation_%A.err

module purge
module load Python/3.11.3-GCCcore-12.3.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Will run withGRU and noGRU, for seeds 10-19
poetry run python experiments/dyna_run_experiment.py \
    --experiment ablation \
    --config configs/dyna_config.yaml \
    --parallel \
    --workers $SLURM_CPUS_PER_TASK \
