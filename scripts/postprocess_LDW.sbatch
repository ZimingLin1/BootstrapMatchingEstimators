#!/bin/bash
#SBATCH --job-name=BM_LDW_POST
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs result

module load Python
source ~/venvs/BMproject/bin/activate
cd "$SLURM_SUBMIT_DIR"

RUNFILE="result/out_LDW.feather"
DATAFILE="data/exp_generated.feather"
CSV_OUT="result/LDW_res.csv"
TEX_OUT="result/Coverage_LDW.tex"

python -u -m bmproject.postprocess_LDW \
  --runfile "$RUNFILE" \
  --datafile "$DATAFILE" \
  --outfile_csv "$CSV_OUT" \
  --outfile_tex "$TEX_OUT" \
  --alphas "0.10,0.05" \
  --scale_k 1000

echo "[OK] Wrote $CSV_OUT and $TEX_OUT"
