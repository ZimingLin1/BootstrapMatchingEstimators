#!/usr/bin/env python3
import argparse, os
from typing import List
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------- helpers ----------
def parse_alpha_list(alpha_str: str) -> List[float]:
    alphas = []
    for s in alpha_str.split(","):
        s = s.strip()
        if s:
            alphas.append(float(s))
    return sorted(set(alphas))

def alpha_to_tag(a: float) -> str:
    # 0.10 -> p90, 0.05 -> p95
    lvl = int(round((1.0 - a) * 100))
    return f"p{lvl:d}"

def compute_ate_from_data_shadish(data: pd.DataFrame) -> float:
    """
    Shadish ATE per original R code:
      te = ((vm==1)-(vm==0)) * (mathall - mathall_cf)
      ate = wT*mean(te|vm==1) + wC*mean(te|vm==0), wT=79/(79+131)
    """
    need = {"vm", "mathall", "mathall_cf"}
    missing = need - set(data.columns)
    if missing:
        raise ValueError(f"datafile missing columns: {sorted(missing)}")
    vm = data["vm"].to_numpy()
    te = ((vm == 1).astype(int) - (vm == 0).astype(int)) * (data["mathall"] - data["mathall_cf"])
    wT = 79 / (79 + 131)
    wC = 1.0 - wT
    ate = wT * float(te[vm == 1].mean()) + wC * float(te[vm == 0].mean())
    return ate

# ---------- core ----------
def build_metrics(runs: pd.DataFrame, ate: float, alphas: List[float]) -> pd.DataFrame:
    """
    Compute Bias/SD/RMSE/MAE from *unscaled* error = est - ate,
    and coverage for SE/AISE/bootstrap.
    Compatible with variable M (multiple M values per N): we aggregate by (N, M).
    """
    runs = runs.copy()
    # raw errors (DO NOT SCALE per requirement)
    runs["error"] = runs["est"] - ate

    # coverage indicators for requested alphas (expect 0.10 and 0.05)
    for a in alphas:
        tag = alpha_to_tag(a)  # p90 / p95
        zcrit = norm.ppf(1.0 - a / 2.0)

        runs[f"covered_se_{tag}"]   = (np.abs(runs["error"]) < zcrit * runs["se"]).astype(int)
        runs[f"covered_AIse_{tag}"] = (np.abs(runs["error"]) < zcrit * runs["AIse"]).astype(int)

        lo_col, hi_col = f"boot_lo_{tag}", f"boot_hi_{tag}"
        if lo_col not in runs.columns or hi_col not in runs.columns:
            raise ValueError(f"Missing {lo_col}/{hi_col} in runfile; ensure alpha list includes {a}.")
        runs[f"covered_boot_{tag}"] = ((runs[lo_col] <= ate) & (ate <= runs[hi_col])).astype(int)

    # aggregate by (N, M) — supports multiple M per N
    g = runs.groupby(["N", "M"], as_index=False).agg(
        Bias=("error", "mean"),
        SD=("error", lambda s: float(np.sqrt(np.mean((s - s.mean()) ** 2)))),
        RMSE=("error", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        MAE=("error", lambda s: float(np.mean(np.abs(s)))),
        n_runs=("error", "size"),
        cov90_SE   = (f"covered_se_{alpha_to_tag(0.10)}",   "mean"),
        cov90_AISE = (f"covered_AIse_{alpha_to_tag(0.10)}", "mean"),
        cov90_BOOT = (f"covered_boot_{alpha_to_tag(0.10)}", "mean"),
        cov95_SE   = (f"covered_se_{alpha_to_tag(0.05)}",   "mean"),
        cov95_AISE = (f"covered_AIse_{alpha_to_tag(0.05)}", "mean"),
        cov95_BOOT = (f"covered_boot_{alpha_to_tag(0.05)}", "mean"),
    )

    return g.sort_values(["N","M"]).reset_index(drop=True)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Shadish postprocess -> CSV + LaTeX with coverage (SE/AISE/Bootstrap)")
    ap.add_argument("--runfile", type=str, default="result/out_Shadish.feather")
    ap.add_argument("--datafile", type=str, default="data/shadish_generated.feather")
    ap.add_argument("--outfile_csv", type=str, default="result/Shadish_res.csv")
    ap.add_argument("--outfile_tex", type=str, default="result/Coverage_Shadish.tex")
    ap.add_argument("--alphas", type=str, default="0.10,0.05", help="Must include 0.10 and 0.05 (90%/95%).")
    ap.add_argument("--scale_k", type=float, default=1000.0, help="(Unused) kept for compatibility.")
    ap.add_argument("--float_digits", type=int, default=3)
    args = ap.parse_args()

    # ensure output dirs
    os.makedirs(os.path.dirname(args.outfile_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.outfile_tex) or ".", exist_ok=True)

    # load data
    data = pd.read_feather(args.datafile)
    runs = pd.read_feather(args.runfile)

    # alpha checks
    alphas = parse_alpha_list(args.alphas)
    need = {0.10, 0.05}
    if not need.issubset(set(round(a, 2) for a in alphas)):
        raise ValueError("alphas must include 0.10 and 0.05 for 90%/95% coverage.")

    # compute metrics
    ate = compute_ate_from_data_shadish(data)
    g = build_metrics(runs, ate, alphas)

    # ----- CSV -----
    csv_df = g.rename(columns={"N":"n"})[
        ["n","M","RMSE","Bias","SD","MAE",
         "cov90_SE","cov90_AISE","cov90_BOOT",
         "cov95_SE","cov95_AISE","cov95_BOOT"]
    ].copy()
    csv_df.columns = ["n","M","RMSE","Bias","SD","MAE",
                      "90%_SE","90%_AISE","90%_BOOT",
                      "95%_SE","95%_AISE","95%_BOOT"]
    csv_df.to_csv(args.outfile_csv, index=False)
    print(f"[OK] CSV: {args.outfile_csv}")

    # ----- LaTeX -----
    tex_df = g.rename(columns={"N":"n"})[
        ["n","M",
         "cov90_SE","cov90_AISE","cov90_BOOT",
         "cov95_SE","cov95_AISE","cov95_BOOT"]
    ].copy()
    tex_df.columns = ["n","M",
                      "90%_SE","90%_AISE","90%_BOOT",
                      "95%_SE","95%_AISE","95%_BOOT"]

    latex = tex_df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.{args.float_digits}f}",
        caption="Coverage (SE / AISE / Bootstrap) at 90% and 95% — Shadish (by N and M)",
        label="tab:coverage_Shadish",
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=False,
        longtable=False,
    )
    with open(args.outfile_tex, "w") as f:
        f.write(latex)
    print(f"[OK] LaTeX: {args.outfile_tex}")

if __name__ == "__main__":
    main()
