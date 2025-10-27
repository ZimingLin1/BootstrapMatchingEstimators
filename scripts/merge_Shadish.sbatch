#!/bin/bash
#SBATCH --job-name=BM_Shadish_MERGE
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs result

module load Python
source ~/venvs/BMproject/bin/activate
cd "$SLURM_SUBMIT_DIR"

OUT_ROOT="$SLURM_SUBMIT_DIR/results_Shadish_array"
OUT_FINAL="$SLURM_SUBMIT_DIR/result/out_Shadish.feather"

export OUT_ROOT OUT_FINAL

python - <<'PY'
import os, glob, pandas as pd
root = os.environ["OUT_ROOT"]
out  = os.environ["OUT_FINAL"]
files = sorted(glob.glob(os.path.join(root, "Shadish_task-*/out_shadish.feather")))
if not files:
    raise SystemExit(f"No per-task out_shadish.feather files found under {root}")
dfs = []
for f in files:
    try:
        dfs.append(pd.read_feather(f))
    except Exception as e:
        print(f"[WARN] skip {f}: {e}")
if not dfs:
    raise SystemExit("All per-task files failed to read.")
df = pd.concat(dfs, ignore_index=True)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
df.reset_index(drop=True).to_feather(out)
print(f"[OK] merged {len(files)} files into {out} ; rows={len(df)} ; cols={list(df.columns)}")
PY
