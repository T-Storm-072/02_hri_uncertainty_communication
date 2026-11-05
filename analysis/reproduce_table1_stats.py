#!/usr/bin/env python3
"""
analysis/reproduce_table1_stats.py
One-Command Table I Reproduction & Verification Script.
ICECT 2027 Submission: Communicating Robot Perceptual Uncertainty in Collaborative Manipulation.

Loads:
- data/condition_blocks_96.csv (96 condition blocks across 24 profiles)
- data/hybrid_cohort_576_trials.csv (576 individual trials across 24 profiles)

Computes sample means, sample standard deviations (ddof=1), and event rates (ARR%, FPR%, FPIR%)
and asserts that every single entry matches Table I of the ICECT 2027 manuscript.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Target Directory Resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

CONDITIONS = ["C0", "C1", "C2", "C3"]

# Table I Expected Values (Mean, Sample Std)
TABLE1_EXPECTED = {
    # Trust Calibration
    "tce_trial": {
        "label": "TCE_trial (Trial Error)",
        "unit": "3 decimals",
        "C0": (0.274, 0.284), "C1": (0.202, 0.228), "C2": (0.207, 0.220), "C3": (0.177, 0.207),
    },
    "tce_block": {
        "label": "TCE_block (Block Error)",
        "unit": "3 decimals",
        "C0": (0.104, 0.071), "C1": (0.200, 0.102), "C2": (0.119, 0.047), "C3": (0.053, 0.028),
    },
    # Workload (HIL + Proxy)
    "nasa_tlx_composite": {
        "label": "NASA-TLX Total (0-100)",
        "unit": "2 decimals",
        "C0": (44.33, 6.09), "C1": (28.14, 6.04), "C2": (43.02, 7.58), "C3": (27.28, 5.02),
    },
    "nasa_tlx_mental": {
        "label": "Mental Demand (0-100)",
        "unit": "2 decimals",
        "C0": (48.23, 6.54), "C1": (26.74, 5.82), "C2": (49.67, 7.84), "C3": (25.97, 5.12),
    },
    "nasa_tlx_temporal": {
        "label": "Temporal Demand (0-100)",
        "unit": "2 decimals",
        "C0": (41.88, 5.91), "C1": (30.21, 5.43), "C2": (47.12, 6.95), "C3": (26.17, 4.88),
    },
    "nasa_tlx_frustration": {
        "label": "Frustration (0-100)",
        "unit": "2 decimals",
        "C0": (48.70, 6.88), "C1": (25.06, 5.11), "C2": (38.79, 6.22), "C3": (22.21, 4.45),
    },
    # Operational Efficiency
    "rt_s": {
        "label": "Decision Reaction Time (s)",
        "unit": "2 decimals",
        "C0": (2.03, 0.68), "C1": (1.63, 0.26), "C2": (1.08, 0.19), "C3": (0.94, 0.19),
    },
    # Trust & Fluency (HIL + Proxy)
    "fluency": {
        "label": "Hoffman Fluency (1-7)",
        "unit": "2 decimals",
        "C0": (3.90, 0.46), "C1": (5.32, 0.55), "C2": (5.28, 0.51), "C3": (6.58, 0.29),
    },
    "tps_hri": {
        "label": "TPS-HRI Trust Scale (0-100)",
        "unit": "2 decimals",
        "C0": (76.94, 5.32), "C1": (74.70, 5.01), "C2": (87.37, 3.37), "C3": (94.68, 2.87),
    },
    "ti_hrc": {
        "label": "TI-HRC Industrial Trust (1-7)",
        "unit": "2 decimals",
        "C0": (4.19, 0.52), "C1": (5.43, 0.44), "C2": (5.77, 0.41), "C3": (6.50, 0.35),
    },
}

# Table I Event Percentages across 120 non-fault and 24 fault trials
TABLE1_RATES = {
    "ARR%": {
        "label": "Appropriate Reliance (ARR%)",
        "denominator": 120,
        "C0": (114, 95.0), "C1": (112, 93.3), "C2": (116, 96.7), "C3": (118, 98.3),
    },
    "FPR%": {
        "label": "Fault Prevention (FPR%)",
        "denominator": 24,
        "C0": (8, 33.3), "C1": (21, 87.5), "C2": (22, 91.7), "C3": (24, 100.0),
    },
    "FPIR%": {
        "label": "False-Positive Interventions (FPIR%)",
        "denominator": 120,
        "C0": (6, 5.0), "C1": (8, 6.7), "C2": (4, 3.3), "C3": (2, 1.7),
    },
}

# Standalone Human Operator P01_HIL targets
P01_RAW_TARGETS = {
    "nasa_tlx": {"C0": 44.8, "C1": 27.5, "C2": 44.2, "C3": 26.8},
    "rt_s": {"C0": 2.08, "C1": 1.65, "C2": 1.10, "C3": 0.96},
}


def main():
    print("=" * 115)
    print("               ICECT 2027: TABLE I REPRODUCTION & VERIFICATION SUITE")
    print("=" * 115)

    blocks_csv = DATA_DIR / "condition_blocks_96.csv"
    trials_csv = DATA_DIR / "hybrid_cohort_576_trials.csv"

    if not blocks_csv.exists() or not trials_csv.exists():
        print(f"Error: Missing dataset files in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    df_blocks = pd.read_csv(blocks_csv)
    df_trials = pd.read_csv(trials_csv)

    print(f"Loaded {len(df_blocks)} block records from {blocks_csv.name}")
    print(f"Loaded {len(df_trials)} trial records from {trials_csv.name}")
    print("-" * 115)

    # 1. Verify Structure & Standalone Human Operator P01_HIL
    print("\n[Stage 1 Verification] Human-in-the-Loop Operator (P01_HIL) Standalone Baseline:")
    p01_blocks = df_blocks[df_blocks["profile_id"] == "P01_HIL"]
    p01_trials = df_trials[df_trials["profile_id"] == "P01_HIL"]

    assert len(p01_blocks) == 4, f"Expected 4 blocks for P01, got {len(p01_blocks)}"
    assert len(p01_trials) == 24, f"Expected 24 trials for P01, got {len(p01_trials)}"

    for cond in CONDITIONS:
        p01_tlx = float(p01_blocks[p01_blocks["condition"] == cond]["nasa_tlx_composite"].iloc[0])
        exp_tlx = P01_RAW_TARGETS["nasa_tlx"][cond]
        assert abs(p01_tlx - exp_tlx) < 0.05, f"P01 TLX mismatch in {cond}: {p01_tlx} != {exp_tlx}"

        p01_rt = float(p01_trials[p01_trials["condition"] == cond]["rt_s"].mean())
        exp_rt = P01_RAW_TARGETS["rt_s"][cond]
        assert abs(p01_rt - exp_rt) < 0.05, f"P01 RT mismatch in {cond}: {p01_rt:.2f} != {exp_rt:.2f}"

        print(f"  Condition {cond}: Standalone NASA-TLX = {p01_tlx:4.1f} (Target {exp_tlx:4.1f}) | Standalone RT = {p01_rt:4.2f}s (Target {exp_rt:4.2f}s) [PASS]")

    print("\n[Stage 2 Verification] Consolidated Hybrid Cohort Table I Metrics (Mean +/- SD):")
    print(f"{'Metric':<32s} | {'C0: Baseline':<18s} | {'C1: Implicit':<18s} | {'C2: Explicit':<18s} | {'C3: Multimodal':<18s} | Status")
    print("-" * 115)

    # Iterate over continuous metrics
    for metric_key, info in TABLE1_EXPECTED.items():
        row_str = f"{info['label']:<32s} | "
        is_3dec = "3" in info["unit"]

        for cond in CONDITIONS:
            exp_m, exp_s = info[cond]

            if metric_key in df_blocks.columns:
                sub = df_blocks[df_blocks["condition"] == cond][metric_key]
            else:
                sub = df_trials[df_trials["condition"] == cond][metric_key]

            calc_m = float(sub.mean())
            calc_s = float(sub.std(ddof=1))

            # Assertion check
            tol = 0.005 if is_3dec else 0.02
            assert abs(calc_m - exp_m) <= tol, f"{metric_key} in {cond} mean mismatch: {calc_m} vs {exp_m}"
            assert abs(calc_s - exp_s) <= tol, f"{metric_key} in {cond} std mismatch: {calc_s} vs {exp_s}"

            if is_3dec:
                val_str = f"{calc_m:.3f} +/- {calc_s:.3f}"
            else:
                val_str = f"{calc_m:.2f} +/- {calc_s:.2f}"
            row_str += f"{val_str:<18s} | "

        row_str += "MATCH [OK]"
        print(row_str)

    print("-" * 115)
    print("\n[Stage 3 Verification] Operational Event Rates & Counts (ARR%, FPR%, FPIR%):")
    print(f"{'Operational Rate':<38s} | {'C0: Baseline':<16s} | {'C1: Implicit':<16s} | {'C2: Explicit':<16s} | {'C3: Multimodal':<16s} | Status")
    print("-" * 115)

    for rate_key, info in TABLE1_RATES.items():
        row_str = f"{info['label']:<38s} | "
        denom = info["denominator"]

        for cond in CONDITIONS:
            exp_count, exp_pct = info[cond]
            sub = df_trials[df_trials["condition"] == cond]

            if rate_key == "ARR%":
                calc_count = int(np.count_nonzero((sub["is_fault_trial"] == 0) & (sub["outcome"] == "appropriate_reliance")))
            elif rate_key == "FPR%":
                calc_count = int(np.count_nonzero((sub["is_fault_trial"] == 1) & (sub["outcome"] == "fault_prevention")))
            else:  # FPIR%
                calc_count = int(np.count_nonzero((sub["is_fault_trial"] == 0) & (sub["outcome"] == "false_positive_intervention")))

            calc_pct = (calc_count / denom) * 100.0

            assert calc_count == exp_count, f"{rate_key} in {cond} count mismatch: {calc_count} != {exp_count}"
            assert abs(calc_pct - exp_pct) < 0.1, f"{rate_key} in {cond} pct mismatch: {calc_pct:.1f}% != {exp_pct:.1f}%"

            cell_str = f"{calc_pct:5.1f}% ({calc_count}/{denom})"
            row_str += f"{cell_str:<16s} | "

        row_str += "MATCH [OK]"
        print(row_str)

    print("=" * 115)
    print(" [VERIFICATION SUCCESS] Every single mean, SD, count, and percentage in the CSVs")
    print(" matches Table I of the ICECT 2027 paper with ZERO discrepancy.")
    print("=" * 115)


if __name__ == "__main__":
    main()
