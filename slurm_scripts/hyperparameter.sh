#!/bin/bash
#SBATCH --job-name=marloes_hpsearch
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/hpsearch_%A.out
#SBATCH --error=logs/hpsearch_%A.err

module purge
module load Python/3.11.3-GCCcore-12.3.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TRIALS=200

poetry run python run_experiment.py \
    --experiment hyperparam_search \
    --config configs/dyna_config.yaml \
    --parallel \
    --workers $SLURM_CPUS_PER_TASK \
    --trials $TRIALS \
