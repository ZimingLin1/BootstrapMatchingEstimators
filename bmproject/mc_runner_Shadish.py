#!/usr/bin/env python3
"""
Shadish Monte Carlo runner (Python)

- Input dataset (default): data/shadish_generated.feather
- Treatment indicator: vm (1 = treated)
- Outcome: mathall
- Covariates: vocabpre, mathpre, numbmath, age, momdegr, daddegr, hsgpaar, cauc, male
- Fixed treated:control ratio = 79:131
- Model complexity: M = floor(c * N ** (1/3)), c in --M_poly_list (default: 0.5,1,2,5)
- Features: percentile bootstrap CIs, parallel execution, resume, Feather output

Requirements:
    pip install pandas pyarrow numpy tqdm

Assumes an estimators.py in the same directory exposing:
    def BCM(X, Y, W, M) -> dict  # returns {'est','se','AIse'}
"""


import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from pathlib import Path
import time

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# ---------------------------------------------
# CLI
# ---------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo runner for BCM estimator (Shadish dataset)")
    p.add_argument("--input_feather_path", type=str, default="data/shadish_generated.feather",
                   help="Path to dataset containing treated + controls")
    p.add_argument("--output_feather_path", type=str, default="result/out_shadish.feather",
                   help="Output filename (Feather)")
    p.add_argument("--resume", type=lambda x: str(x).lower() in {"1","true","t","yes","y"}, default=True,
                   help="Resume from previous intermediate results?")
    p.add_argument("--N_runs", type=int, default=2000,
                   help="Number of runs per estimator for each (N, M)")
    p.add_argument("--N_chunk", type=int, default=200,
                   help="After N_chunk runs per worker, intermediate results will be saved")
    p.add_argument("--N_workers", type=int, default=10,
                   help="If > 1, runs will be parallelized using concurrent futures")
    p.add_argument("--B_boot", type=int, default=1000,
                   help="Number of bootstrap resamples per iteration")
    p.add_argument("--alphas", type=str, default="0.10,0.05",
                   help='Comma-separated alpha list. Example: "0.10,0.05" for 90% & 95%.')
    p.add_argument("--out-root", type=str, default=None,
                   help="If set, write to <out-root>/<run-name>/out_shadish.feather")
    p.add_argument("--run-name", type=str, default=None,
                   help="Subfolder name for this run (defaults to job-$SLURM_JOB_ID or a timestamp)")
    p.add_argument("--salt", type=int, default=0,
                   help="Extra random salt to de-correlate RNG streams across Slurm array tasks")
    # NEW: multipliers for M
    p.add_argument("--M_poly_list", type=str, default="0.5,1,2,5",
                   help="Comma-separated multipliers for M. Uses M=floor(c * N^(1/3)) for each c.")
    return p.parse_args()

# ---------------------------------------------
# Helpers
# ---------------------------------------------

def parse_alpha_list(alpha_str: str) -> List[float]:
    alphas = []
    for s in alpha_str.split(","):
        s = s.strip()
        if s:
            alphas.append(float(s))
    return sorted(set(alphas))

def alpha_to_tag(a: float) -> str:
    lvl = int(round((1.0 - a) * 100))
    return f"p{lvl:d}"

# ---------------------------------------------
# Estimator registry
# ---------------------------------------------

try:
    from .estimators import BCM  # package-style
except Exception:
    try:
        from estimators import BCM  # script-style
    except Exception as e:
        BCM = None
        warnings.warn(
            "Could not import BCM from estimators.py. Provide estimators.py with BCM(X,Y,W,M).\n"
            f"Import error: {e}"
        )

def BCM_ps(M: int, X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> Dict[str, float]:
    """
    Wrapper allowing BCM to return a dict, an object with attributes,
    or a (est, se, AIse) tuple-like.
    """
    if BCM is None:
        raise RuntimeError("estimators.BCM not found. Create estimators.py with BCM(X,Y,W,M).")
    out = BCM(X, Y, W, M)

    if isinstance(out, dict):
        return {"est": float(out["est"]), "se": float(out["se"]), "AIse": float(out["AIse"])}
    if hasattr(out, "est") and hasattr(out, "se") and hasattr(out, "AIse"):
        return {"est": float(out.est), "se": float(out.se), "AIse": float(out.AIse)}
    if isinstance(out, (tuple, list, np.ndarray)) and len(out) >= 3:
        est, se, AIse = out[:3]
        return {"est": float(est), "se": float(se), "AIse": float(AIse)}
    raise TypeError("Unsupported BCM return type.")

ESTIMATORS = {"BCM_ps": BCM_ps}

# Shadish columns
COVARIATES = [
    "vocabpre", "mathpre", "numbmath", "age",
    "momdegr", "daddegr", "hsgpaar", "cauc", "male",
]
TREAT_COL = "vm"
OUTCOME_COL = "mathall"

# ---------------------------------------------
# Data sampling
# ---------------------------------------------

@dataclass
class Sampled:
    X: np.ndarray
    Y: np.ndarray
    W: np.ndarray

def get_sample(df: pd.DataFrame, covariates: List[str], N_treated: int, N_control: int, rng: np.random.Generator) -> Sampled:
    tmask = df[TREAT_COL] == 1
    treated_df = df.loc[tmask]
    control_df = df.loc[~tmask]

    if N_treated > len(treated_df) or N_control > len(control_df):
        raise ValueError("Requested sample sizes exceed available treated/control counts.")

    treated_idx = rng.choice(treated_df.index.values, size=N_treated, replace=False)
    control_idx = rng.choice(control_df.index.values, size=N_control, replace=False)
    s = df.loc[np.concatenate([treated_idx, control_idx])]

    X = s[covariates].to_numpy(copy=False)
    Y = s[OUTCOME_COL].to_numpy(copy=False)
    W = s[TREAT_COL].to_numpy(copy=False)
    return Sampled(X=X, Y=Y, W=W)

# ---------------------------------------------
# Bootstrap (percentile CI)
# ---------------------------------------------


def bootstrap_percentile_CIs(
    X, Y, W, M, B, alphas, rng, est_fun, hat_est: float | None = None
):
    """
    Stratified bootstrap (by treatment arm):
      - resample treated and control *separately* with replacement
      - preserve group sizes within each bootstrap replicate
    """
    if hat_est is None:
        base = est_fun(M, X, Y, W)
        hat = float(base["est"])
    else:
        hat = float(hat_est)

    # indices by arm
    W = np.asarray(W)
    idx1 = np.flatnonzero(W == 1)
    idx0 = np.flatnonzero(W == 0)
    n1, n0 = len(idx1), len(idx0)

    # child RNG for reproducibility per-iteration
    child = np.random.default_rng(rng.integers(0, 2**63 - 1))

    tau_stars = np.empty(B, dtype=float)
    for b in range(B):
        # resample within arm (with replacement)
        jb1 = child.choice(idx1, size=n1, replace=True)
        jb0 = child.choice(idx0, size=n0, replace=True)
        jb = np.concatenate([jb1, jb0])

        out_b = est_fun(M, X[jb], Y[jb], W[jb])
        tau_stars[b] = float(out_b["est"])

    ci_map = {}
    for a in alphas:
        try:
            lo = float(np.quantile(tau_stars, a/2.0, method="linear"))
            hi = float(np.quantile(tau_stars, 1.0 - a/2.0, method="linear"))
        except TypeError:  # NumPy < 1.22 fallback
            lo = float(np.quantile(tau_stars, a/2.0, interpolation="linear"))
            hi = float(np.quantile(tau_stars, 1.0 - a/2.0, interpolation="linear"))
        ci_map[a] = (lo, hi)
    return hat, ci_map


# ---------------------------------------------
# Single iteration
# ---------------------------------------------

def one_iteration(df: pd.DataFrame, N_treated: int, N_control: int, M: int, method: str, seed: int,
                  B_boot: int, alphas: List[float]):
    rng = np.random.default_rng(seed)
    sample = get_sample(df, COVARIATES, N_treated, N_control, rng)
    fn = ESTIMATORS[method]

    res = fn(M, sample.X, sample.Y, sample.W)
    est = float(res["est"]); se = float(res["se"]); AIse = float(res["AIse"])

    _, ci_map = bootstrap_percentile_CIs(
        sample.X, sample.Y, sample.W, M, B_boot, alphas, rng, est_fun=fn, hat_est=est
    )

    flat = []
    for a in alphas:
        lo, hi = ci_map[a]
        flat.extend([lo, hi])

    return (est, se, AIse, *flat)

# ---------------------------------------------
# Parallel chunk executor
# ---------------------------------------------
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_chunk(df: pd.DataFrame, N_treated: int, N_control: int, M: int, method: str,
              n_jobs: int, n_tasks: int, base_seed: int,
              B_boot: int, alphas: List[float]):
    ss = np.random.SeedSequence(base_seed)
    int_seeds = ss.generate_state(n_tasks, dtype=np.uint64)

    if n_jobs <= 1:
        return [
            one_iteration(df, N_treated, N_control, M, method, int(seed), B_boot, alphas)
            for seed in int_seeds
        ]

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(one_iteration, df, N_treated, N_control, M, method, int(seed), B_boot, alphas)
            for seed in int_seeds
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results

# ---------------------------------------------
# Main
# ---------------------------------------------

def main():
    args = parse_args()

    alphas = parse_alpha_list(args.alphas)
    alpha_tags = [alpha_to_tag(a) for a in alphas]

    # NEW: parse M multipliers
    M_poly_list = [float(s) for s in str(getattr(args, "M_poly_list", "0.5,1,2,5")).split(",") if s.strip()]

    # Normalize paths
    if getattr(args, "input_feather_path", None):
        in_path = Path(args.input_feather_path)
        if not in_path.is_absolute():
            in_path = Path.cwd() / in_path
        if not in_path.exists():
            raise FileNotFoundError(f"Input feather not found: {in_path}")
        args.input_feather_path = str(in_path)

    if getattr(args, "out_root", None):
        out_root = Path(args.out_root)
        if not out_root.is_absolute():
            out_root = Path.cwd() / out_root
        job_id = os.environ.get("SLURM_JOB_ID")
        if getattr(args, "run_name", None):
            run_name = args.run_name
        elif job_id:
            run_name = f"job-{job_id}"
        else:
            run_name = f"local-{time.strftime('%Y%m%d-%H%M%S')}"
        out_dir = out_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_feather_path = str(out_dir / "out_shadish.feather")
    else:
        out_path = Path(args.output_feather_path)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_feather_path = str(out_path)
        out_dir = out_path.parent

    print(f"[paths] input={getattr(args, 'input_feather_path', None)}")
    print(f"[paths] output={args.output_feather_path}")
    print(f"[run] salt={getattr(args, 'salt', 0)}  workers={args.N_workers}  B_boot={args.B_boot}  N_runs={args.N_runs}")
    print(f"[M] multipliers={M_poly_list}")

    # Load data
    if not os.path.exists(args.input_feather_path):
        raise FileNotFoundError(f"Input feather not found: {args.input_feather_path}")
    data = pd.read_feather(args.input_feather_path)

    # Validate required columns
    required_cols = {TREAT_COL, OUTCOME_COL, *COVARIATES}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")

    # Experiment grid
    N_list = [600, 1200, 4800, 9600]

    # Resume
    runs_df = None
    if args.resume and os.path.exists(args.output_feather_path):
        try:
            runs_df = pd.read_feather(args.output_feather_path)
        except Exception:
            runs_df = None
    if runs_df is None:
        base_cols = ["N", "M", "method", "est", "se", "AIse", "B_boot", "alphas"]
        extra = []
        for tag in alpha_tags:
            extra += [f"boot_lo_{tag}", f"boot_hi_{tag}"]
        runs_df = pd.DataFrame(columns=base_cols + extra)

    method_names = ["BCM_ps"]

    print(f"starting iterations with input {args.input_feather_path}")

    iter_counter = 0
    # Fixed treated:control ratio = 79 : 131
    p_treated = 79 / (79 + 131)

    for N in N_list:
        N_treated = int(math.floor(N * p_treated))
        N_control = int(N - N_treated)

        # NEW: iterate over multipliers c and compute M = floor(c * N^(1/3))
        for c in M_poly_list:
            M = max(1, int(math.floor(float(c) * (N ** (1.0 / 3.0)))))

            for method in method_names:
                mask = ((runs_df["method"] == method) & (runs_df["N"] == N) & (runs_df["M"] == M))
                N_completed = int(mask.sum())
                print(f"N: {N} c: {c} M: {M} Method: {method} Completed: {N_completed}")

                while N_completed < args.N_runs:
                    N_tasks = min(max(1, args.N_workers) * args.N_chunk, max(0, args.N_runs - N_completed))

                    ss_base = np.random.SeedSequence([N, M, iter_counter, 123456789, int(getattr(args, "salt", 0))])
                    base_seed = ss_base.generate_state(1, dtype=np.uint32)[0].item()

                    chunk_results = run_chunk(
                        data, N_treated, N_control, M, method,
                        n_jobs=args.N_workers, n_tasks=N_tasks, base_seed=base_seed,
                        B_boot=args.B_boot, alphas=alphas
                    )

                    if (args.resume and os.path.exists(args.output_feather_path)) or (iter_counter > 0):
                        try:
                            runs_df = pd.read_feather(args.output_feather_path)
                        except Exception:
                            pass

                    cols = ["est", "se", "AIse"]
                    for tag in alpha_tags:
                        cols += [f"boot_lo_{tag}", f"boot_hi_{tag}"]

                    chunk_df = pd.DataFrame(chunk_results, columns=cols)
                    chunk_df.insert(0, "method", method)
                    chunk_df.insert(0, "M", M)
                    chunk_df.insert(0, "N", N)
                    chunk_df["B_boot"] = args.B_boot
                    chunk_df["alphas"] = ",".join(str(a) for a in alphas)

                    runs_df = pd.concat([runs_df, chunk_df], ignore_index=True)

                    N_completed += len(chunk_df)
                    print(f"N: {N} c: {c} M: {M} Method: {method} Completed: {N_completed} | Saving results…")
                    os.makedirs(os.path.dirname(args.output_feather_path) or ".", exist_ok=True)
                    runs_df.reset_index(drop=True).to_feather(args.output_feather_path)

                    iter_counter += 1

    print("All done.")

if __name__ == "__main__":
    main()
