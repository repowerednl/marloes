#!/bin/bash
#SBATCH --job-name=marloes_main
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/main_%A.out
#SBATCH --error=logs/main_%A.err

module purge
module load Python/3.11.3-GCCcore-12.3.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Runs all methods, both data scenarios, all seeds in parallel
poetry run python experiments/dyna_run_experiment.py \
    --experiment main \
    --config configs/dyna_config.yaml \
    --parallel \
    --workers $SLURM_CPUS_PER_TASK \
