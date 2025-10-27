#!/bin/bash
#SBATCH --job-name=BM_Shadish_ARR
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-11:30:00
#SBATCH --mem=24G
#SBATCH --array=1-40
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

# avoid BLAS oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load Python
source ~/venvs/BMproject/bin/activate
cd "$SLURM_SUBMIT_DIR"

OUT_ROOT="$SLURM_SUBMIT_DIR/results_Shadish_array"
RUN_NAME="Shadish_task-${SLURM_ARRAY_TASK_ID}"

srun python -u -m bmproject.mc_runner_Shadish \
  --out-root "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --resume false \
  --N_runs 50 \
  --N_chunk 3 \
  --N_workers "$SLURM_CPUS_PER_TASK" \
  --B_boot 2000 \
  --alphas "0.10,0.05" \
  --salt "$SLURM_ARRAY_TASK_ID" \
  --input_feather_path "$SLURM_SUBMIT_DIR/data/shadish_generated.feather"

echo "[OK] task ${SLURM_ARRAY_TASK_ID} -> ${OUT_ROOT}/${RUN_NAME}/out_shadish.feather"
