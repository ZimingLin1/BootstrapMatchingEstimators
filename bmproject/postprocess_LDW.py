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

def compute_ate_from_data(data: pd.DataFrame) -> float:
    """
    LDW ATE per original R code:
      te = ((t==1)-(t==0)) * (re78 - re78_cf)
      ate = wT*mean(te|t==1) + wC*mean(te|t==0), wT=185/(185+260)
    """
    if "re78_cf" not in data.columns:
        raise ValueError("datafile missing column 're78_cf' required for LDW ATE.")
    t = data["t"].to_numpy()
    te = ((t == 1).astype(int) - (t == 0).astype(int)) * (data["re78"] - data["re78_cf"])
    wT = 185 / (185 + 260)
    wC = 1.0 - wT
    ate = wT * float(te[t == 1].mean()) + wC * float(te[t == 0].mean())
    return ate

# ---------- core ----------
def build_metrics(runs: pd.DataFrame, ate: float, alphas: List[float], scale_k: float) -> pd.DataFrame:
    """
    Compute Bias/SD/RMSE/MAE and coverage for SE/AISE/bootstrap by (N, M).
    Results are scaled by K (errors & SEs divided by K) for presentation only.
    Compatible with variable M (multiple M per N).
    """
    runs = runs.copy()
    K = float(scale_k)

    runs["error_scaled"] = (runs["est"] - ate) / K
    runs["se_scaled"]    = runs["se"] / K
    runs["AIse_scaled"]  = runs["AIse"] / K

    # coverage indicators
    for a in alphas:
        tag = alpha_to_tag(a)  # p90 / p95
        zcrit = norm.ppf(1.0 - a / 2.0)

        runs[f"covered_se_{tag}"]   = (np.abs(runs["error_scaled"]) < zcrit * runs["se_scaled"]).astype(int)
        runs[f"covered_AIse_{tag}"] = (np.abs(runs["error_scaled"]) < zcrit * runs["AIse_scaled"]).astype(int)

        lo_col, hi_col = f"boot_lo_{tag}", f"boot_hi_{tag}"
        if lo_col not in runs.columns or hi_col not in runs.columns:
            raise ValueError(f"Missing {lo_col}/{hi_col} in runfile; ensure alpha list includes {a}.")
        runs[f"covered_boot_{tag}"] = ((runs[lo_col] <= ate) & (ate <= runs[hi_col])).astype(int)

    # group by N, M（可变 M 时每个 (N,M) 都单独统计）
    g = runs.groupby(["N", "M"], as_index=False).agg(
        Bias=("error_scaled", "mean"),
        SD=("error_scaled", lambda s: float(np.sqrt(np.mean((s - s.mean()) ** 2)))),
        RMSE=("error_scaled", lambda s: float(np.sqrt(np.mean(s ** 2)))),
        MAE=("error_scaled", lambda s: float(np.mean(np.abs(s)))),
        n_runs=("error_scaled", "size"),
        cov90_SE   = (f"covered_se_{alpha_to_tag(0.10)}",   "mean"),
        cov90_AISE = (f"covered_AIse_{alpha_to_tag(0.10)}", "mean"),
        cov90_BOOT = (f"covered_boot_{alpha_to_tag(0.10)}", "mean"),
        cov95_SE   = (f"covered_se_{alpha_to_tag(0.05)}",   "mean"),
        cov95_AISE = (f"covered_AIse_{alpha_to_tag(0.05)}", "mean"),
        cov95_BOOT = (f"covered_boot_{alpha_to_tag(0.05)}", "mean"),
    )

    # final sort
    g = g.sort_values(["N","M"]).reset_index(drop=True)
    return g

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="LDW postprocess -> CSV + LaTeX with coverage (SE/AISE/Bootstrap)")
    ap.add_argument("--runfile", type=str, default="result/out_LDW.feather")
    ap.add_argument("--datafile", type=str, default="data/exp_generated.feather")
    ap.add_argument("--outfile_csv", type=str, default="result/LDW_res.csv")
    ap.add_argument("--outfile_tex", type=str, default="result/Coverage_LDW.tex")
    ap.add_argument("--alphas", type=str, default="0.10,0.05", help="Should include 0.10 and 0.05 for 90%/95%")
    ap.add_argument("--scale_k", type=float, default=1000.0, help="Divide errors/SEs by this K for presentation.")
    ap.add_argument("--float_digits", type=int, default=3)
    args = ap.parse_args()

    # prepare
    os.makedirs(os.path.dirname(args.outfile_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.outfile_tex) or ".", exist_ok=True)

    # load
    data = pd.read_feather(args.datafile)
    runs = pd.read_feather(args.runfile)

    # ensure required alphas
    alphas = parse_alpha_list(args.alphas)
    need = {0.10, 0.05}
    if not need.issubset(set(round(a, 2) for a in alphas)):
        raise ValueError("alphas must include 0.10 and 0.05 for 90%/95% coverage.")

    ate = compute_ate_from_data(data)
    g = build_metrics(runs, ate, alphas, args.scale_k)

    # ----- CSV: specific columns & names -----
    csv_df = g.rename(columns={"N":"n"})[
        ["n","M","RMSE","Bias","SD","MAE",
         "cov90_SE","cov90_AISE","cov90_BOOT",
         "cov95_SE","cov95_AISE","cov95_BOOT"]
    ].copy()
    # nicer column names
    csv_df.columns = ["n","M","RMSE","Bias","SD","MAE",
                      "90%_SE","90%_AISE","90%_BOOT",
                      "95%_SE","95%_AISE","95%_BOOT"]
    csv_df.to_csv(args.outfile_csv, index=False)
    print(f"[OK] CSV: {args.outfile_csv}")

    # ----- LaTeX: only n, M, coverage blocks -----
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
        caption="Coverage (SE / AISE / Bootstrap) at 90% and 95%",
        label="tab:coverage_LDW",
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
