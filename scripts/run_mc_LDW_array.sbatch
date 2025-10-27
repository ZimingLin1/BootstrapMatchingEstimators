#!/bin/bash
#SBATCH --job-name=BM_LDW_ARR
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

# 防止 BLAS 抢核
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load Python
source ~/venvs/BMproject/bin/activate
cd "$SLURM_SUBMIT_DIR"

N_RUNS_PIECE=50
# 固定到提交目录，避免写到 SCRATCH 或双层 BMproject
OUT_ROOT="$SLURM_SUBMIT_DIR/results_LDW_array"
RUN_NAME="LDW_task-${SLURM_ARRAY_TASK_ID}"

srun python -u -m bmproject.mc_runner_LDW \
  --out-root "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --resume false \
  --N_runs "$N_RUNS_PIECE" \
  --N_chunk 3 \
  --N_workers "$SLURM_CPUS_PER_TASK" \
  --B_boot 2000 \
  --alphas "0.10,0.05" \
  --salt "$SLURM_ARRAY_TASK_ID" \
  --input_feather_path "$SLURM_SUBMIT_DIR/data/exp_generated.feather"

echo "[OK] task ${SLURM_ARRAY_TASK_ID} -> ${OUT_ROOT}/${RUN_NAME}/out.feather"
