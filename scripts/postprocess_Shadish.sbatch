#!/bin/bash
#SBATCH --job-name=BM_Shadish_POST
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

RUNFILE="result/out_Shadish.feather"
DATAFILE="data/shadish_generated.feather"
CSV_OUT="result/Shadish_res.csv"
TEX_OUT="result/Coverage_Shadish.tex"

# # 如果 postprocess_Shadish.py 放在项目根目录（常见）
# python -u postprocess_Shadish.py \
#   --runfile "$RUNFILE" \
#   --datafile "$DATAFILE" \
#   --outfile_csv "$CSV_OUT" \
#   --outfile_tex "$TEX_OUT" \
#   --alphas "0.10,0.05" \
#   --scale_k 1000

# 若在包内（例如 bmproject/postprocess_Shadish.py），改用下面这段：
python -u -m bmproject.postprocess_Shadish \
  --runfile "$RUNFILE" \
  --datafile "$DATAFILE" \
  --outfile_csv "$CSV_OUT" \
  --outfile_tex "$TEX_OUT" \
  --alphas "0.10,0.05" \
  --scale_k 1000

echo "[OK] Wrote $CSV_OUT and $TEX_OUT"
