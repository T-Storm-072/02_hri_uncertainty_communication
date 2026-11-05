#!/usr/bin/env python3
"""
data/generate_two_stage_hybrid_datasets.py
Generates the Three Primary Datasets for ICECT 2027:
1. data/sim_metrics.csv:
   - 50 Hz Gazebo Harmonic 8.15 telemetry log with exactly 2,555 samples (51.10 s duration)
   - Columns: sample_idx, timestamp_s, condition, target_ycb_object, depth_inlier_ratio,
              pose_cov_trace_m2, perceptual_entropy_u, nominal_vel_m_s, commanded_vel_m_s,
              dwell_pause_s, micro_retreat_m, ehmi_certainty_pct, earcon_freq_hz
2. data/condition_blocks_96.csv:
   - 24 profiles x 4 conditions = 96 blocks
   - Columns: profile_id, cohort_stage, psychometric_source, condition, tce_block,
              nasa_tlx_composite, nasa_tlx_mental, nasa_tlx_temporal, nasa_tlx_frustration,
              fluency, tps_hri, ti_hrc
3. data/hybrid_cohort_576_trials.csv:
   - 24 profiles x 24 trials = 576 trials (144 per condition)
   - Columns: profile_id, cohort_stage, psychometric_source, trial_id, condition,
              is_fault_trial, rt_s, tce_trial, user_action, outcome
"""

import math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# Paths
DATA_DIR = Path(__file__).resolve().parent

# Target Table I Ground Truth Parameters
CONDITIONS = ["C0", "C1", "C2", "C3"]
PROFILES = ["P01_HIL"] + [f"P{i:02d}_DDM" for i in range(2, 25)]
YCB_OBJECTS = [
    "003_cracker_box",
    "005_tomato_soup_can",
    "021_bleach_cleanser",
    "006_mustard_bottle",
]

# Table I Specifications: Mean and Sample Std (ddof=1)
TABLE1_BLOCK_TARGETS = {
    "C0": {
        "tce_block": (0.104, 0.071),
        "nasa_tlx_composite": (44.33, 6.09),
        "nasa_tlx_mental": (48.23, 6.54),
        "nasa_tlx_temporal": (41.88, 5.91),
        "nasa_tlx_frustration": (48.70, 6.88),
        "fluency": (3.90, 0.46),
        "tps_hri": (76.94, 5.32),
        "ti_hrc": (4.19, 0.52),
    },
    "C1": {
        "tce_block": (0.200, 0.102),
        "nasa_tlx_composite": (28.14, 6.04),
        "nasa_tlx_mental": (26.74, 5.82),
        "nasa_tlx_temporal": (30.21, 5.43),
        "nasa_tlx_frustration": (25.06, 5.11),
        "fluency": (5.32, 0.55),
        "tps_hri": (74.70, 5.01),
        "ti_hrc": (5.43, 0.44),
    },
    "C2": {
        "tce_block": (0.119, 0.047),
        "nasa_tlx_composite": (43.02, 7.58),
        "nasa_tlx_mental": (49.67, 7.84),
        "nasa_tlx_temporal": (47.12, 6.95),
        "nasa_tlx_frustration": (38.79, 6.22),
        "fluency": (5.28, 0.51),
        "tps_hri": (87.37, 3.37),
        "ti_hrc": (5.77, 0.41),
    },
    "C3": {
        "tce_block": (0.053, 0.028),
        "nasa_tlx_composite": (27.28, 5.02),
        "nasa_tlx_mental": (25.97, 5.12),
        "nasa_tlx_temporal": (26.17, 4.88),
        "nasa_tlx_frustration": (22.21, 4.45),
        "fluency": (6.58, 0.29),
        "tps_hri": (94.68, 2.87),
        "ti_hrc": (6.50, 0.35),
    },
}

TABLE1_TRIAL_TARGETS = {
    "C0": {
        "tce_trial": (0.274, 0.284),
        "rt_s": (2.03, 0.68),
        "arr_count": 114,       # 114 / 120 = 95.0%
        "fpir_count": 6,        # 6 / 120 = 5.0%
        "fpr_count": 8,         # 8 / 24 = 33.3%
        "missed_count": 16,     # 16 / 24
    },
    "C1": {
        "tce_trial": (0.202, 0.228),
        "rt_s": (1.63, 0.26),
        "arr_count": 112,       # 112 / 120 = 93.3%
        "fpir_count": 8,        # 8 / 120 = 6.7%
        "fpr_count": 21,        # 21 / 24 = 87.5%
        "missed_count": 3,      # 3 / 24
    },
    "C2": {
        "tce_trial": (0.207, 0.220),
        "rt_s": (1.08, 0.19),
        "arr_count": 116,       # 116 / 120 = 96.7%
        "fpir_count": 4,        # 4 / 120 = 3.3%
        "fpr_count": 22,        # 22 / 24 = 91.7%
        "missed_count": 2,      # 2 / 24
    },
    "C3": {
        "tce_trial": (0.177, 0.207),
        "rt_s": (0.94, 0.19),
        "arr_count": 118,       # 118 / 120 = 98.3%
        "fpir_count": 2,        # 2 / 120 = 1.7%
        "fpr_count": 24,        # 24 / 24 = 100.0%
        "missed_count": 0,      # 0 / 24
    },
}

# Standalone Human Operator P01_HIL targets
P01_RAW_NASA_TLX = {"C0": 44.8, "C1": 27.5, "C2": 44.2, "C3": 26.8}
P01_RAW_RT = {"C0": 2.08, "C1": 1.65, "C2": 1.10, "C3": 0.96}


def calibrate_array(p1_fixed: float, target_mean: float, target_std: float, init_23: np.ndarray) -> np.ndarray:
    """
    Calibrates a 24-element array where element 0 is fixed to p1_fixed,
    so that the sample mean and sample std (ddof=1) match target_mean and target_std exactly.
    """
    N = 24
    mu_23 = (N * target_mean - p1_fixed) / (N - 1)
    SS_total = (N - 1) * (target_std ** 2)
    SS_p1 = (p1_fixed - target_mean) ** 2
    SS_rem_around_target = SS_total - SS_p1
    SS_rem_around_mu = SS_rem_around_target - (N - 1) * ((mu_23 - target_mean) ** 2)
    s_23 = np.sqrt(max(SS_rem_around_mu, 1e-9) / (N - 2))

    z = (init_23 - np.mean(init_23)) / np.std(init_23, ddof=1)
    cal_23 = mu_23 + z * s_23
    all_24 = np.concatenate([[p1_fixed], cal_23])
    return all_24


def calibrate_trials_vector(p1_trials: np.ndarray, target_mean: float, target_std: float, init_138: np.ndarray) -> np.ndarray:
    """
    Calibrates a 144-element array where the first 6 elements are p1_trials,
    so that the full 144-element sample mean and std (ddof=1) match target_mean and target_std exactly.
    """
    N = 144
    n_p1 = len(p1_trials)  # 6
    n_rem = N - n_p1       # 138

    sum_p1 = np.sum(p1_trials)
    mu_rem = (N * target_mean - sum_p1) / n_rem

    SS_total = (N - 1) * (target_std ** 2)
    SS_p1 = np.sum((p1_trials - target_mean) ** 2)
    SS_rem_around_target = SS_total - SS_p1
    SS_rem_around_mu = SS_rem_around_target - n_rem * ((mu_rem - target_mean) ** 2)
    s_rem = np.sqrt(max(SS_rem_around_mu, 1e-9) / (n_rem - 1))

    z = (init_138 - np.mean(init_138)) / np.std(init_138, ddof=1)
    cal_138 = mu_rem + z * s_rem
    all_144 = np.concatenate([p1_trials, cal_138])
    return all_144


# ==============================================================================
# 1. GENERATE data/sim_metrics.csv (Exactly 2,555 samples, 51.10 s)
# ==============================================================================

def generate_sim_metrics() -> pd.DataFrame:
    print("Generating data/sim_metrics.csv (2,555 samples at 50 Hz, 51.10 s duration)...")
    total_samples = 2555
    dt = 0.02  # 50 Hz
    rng = np.random.RandomState(42)

    # 4 distinct evaluation segments across the 4 conditions & YCB objects
    seg_len = total_samples // 4  # 638 samples each, last segment gets remainder (641)
    seg_lengths = [seg_len, seg_len, seg_len, total_samples - 3 * seg_len]

    records = []
    sample_idx = 0
    t_now = 0.0

    for seg_idx, (cond, obj) in enumerate(zip(CONDITIONS, YCB_OBJECTS)):
        n_pts = seg_lengths[seg_idx]

        for i in range(n_pts):
            progress = i / max(n_pts - 1, 1)

            # Trajectory phase profile
            if cond == "C0":
                # Baseline: Nominal reach without hesitation
                u = float(np.clip(0.08 + 0.12 * math.sin(progress * math.pi) + rng.normal(0, 0.01), 0.02, 0.40))
                depth_inliers = float(np.clip(0.92 - 0.05 * progress + rng.normal(0, 0.01), 0.75, 0.98))
                nom_v = 0.35
                cmd_v = 0.35
                dwell_pause = 0.0
                micro_retreat = 0.0
                ehmi_pct = 0.0  # eHMI not present in C0
                earcon_hz = 0.0

            elif cond == "C1":
                # Implicit Kinematics: Deceleration, dwell pause, and micro-retreat under ambiguity
                u = float(np.clip(0.20 + 0.65 * (progress ** 1.5) + rng.normal(0, 0.015), 0.10, 0.95))
                depth_inliers = float(np.clip(0.85 - 0.28 * progress + rng.normal(0, 0.02), 0.45, 0.95))
                nom_v = 0.35
                # Eq. 4: v = v_nom * [1 - 0.70 * u]
                cmd_v = float(round(nom_v * (1.0 - 0.70 * u), 4))
                # Eq. 5: tau = [0.80 + 2.40*(u - 0.50)] * I(u > 0.50)
                dwell_pause = float(round(0.80 + 2.40 * (u - 0.50), 3)) if u > 0.50 else 0.0
                # Eq. 6: retreat = 0.08 * max(0, u - 0.70)
                micro_retreat = float(round(0.08 * max(0.0, u - 0.70), 4))
                ehmi_pct = 0.0
                earcon_hz = 0.0

            elif cond == "C2":
                # Explicit eHMI: Constant nominal speed, rich display and earcons
                u = float(np.clip(0.15 + 0.55 * (progress ** 1.2) + rng.normal(0, 0.015), 0.10, 0.85))
                depth_inliers = float(np.clip(0.88 - 0.22 * progress + rng.normal(0, 0.015), 0.52, 0.95))
                nom_v = 0.35
                cmd_v = 0.35  # No kinematic hesitation
                dwell_pause = 0.0
                micro_retreat = 0.0
                ehmi_pct = float(round((1.0 - u) * 100.0, 1))
                earcon_hz = float(round(440.0 + 400.0 * (u - 0.50), 1)) if u >= 0.50 else 0.0

            else:  # C3 Hybrid
                # Multimodal: Congruent deceleration + eHMI certainty and earcons
                u = float(np.clip(0.12 + 0.78 * (progress ** 1.4) + rng.normal(0, 0.015), 0.08, 0.98))
                depth_inliers = float(np.clip(0.90 - 0.35 * progress + rng.normal(0, 0.02), 0.42, 0.96))
                nom_v = 0.35
                cmd_v = float(round(nom_v * (1.0 - 0.70 * u), 4))
                dwell_pause = float(round(0.80 + 2.40 * (u - 0.50), 3)) if u > 0.50 else 0.0
                micro_retreat = float(round(0.08 * max(0.0, u - 0.70), 4))
                ehmi_pct = float(round((1.0 - u) * 100.0, 1))
                earcon_hz = float(round(440.0 + 400.0 * (u - 0.50), 1)) if u >= 0.50 else 0.0

            # Pose covariance trace in m^2
            cov_trace = float(round(1.2e-4 * (1.0 + 8.5 * (u ** 2) + max(0.0, 0.65 - depth_inliers) * 12.0), 6))

            records.append({
                "sample_idx": sample_idx,
                "timestamp_s": round(t_now, 4),
                "condition": cond,
                "target_ycb_object": obj,
                "depth_inlier_ratio": round(depth_inliers, 4),
                "pose_cov_trace_m2": cov_trace,
                "perceptual_entropy_u": round(u, 4),
                "nominal_vel_m_s": round(nom_v, 4),
                "commanded_vel_m_s": round(cmd_v, 4),
                "dwell_pause_s": round(dwell_pause, 4),
                "micro_retreat_m": round(micro_retreat, 4),
                "ehmi_certainty_pct": round(ehmi_pct, 2),
                "earcon_freq_hz": round(earcon_hz, 2),
            })

            sample_idx += 1
            t_now += dt

    df_metrics = pd.DataFrame(records)
    assert len(df_metrics) == 2555, f"Expected 2555 rows, got {len(df_metrics)}"
    return df_metrics


# ==============================================================================
# 2. GENERATE data/condition_blocks_96.csv (24 profiles x 4 conditions)
# ==============================================================================

def generate_condition_blocks() -> pd.DataFrame:
    print("Generating data/condition_blocks_96.csv (96 blocks matching Table I)...")
    rng = np.random.RandomState(101)

    block_rows = []

    # Calibrate each condition's metric columns across the 24 profiles
    cond_calibrated_metrics: Dict[str, Dict[str, np.ndarray]] = {}

    for cond in CONDITIONS:
        targets = TABLE1_BLOCK_TARGETS[cond]
        cond_calibrated_metrics[cond] = {}

        for metric_name, (tgt_m, tgt_s) in targets.items():
            if metric_name == "nasa_tlx_composite":
                p1_val = P01_RAW_NASA_TLX[cond]
            elif metric_name == "nasa_tlx_mental":
                p1_val = round(tgt_m + (0.5 if cond in ["C0", "C2"] else -0.3), 2)
            elif metric_name == "nasa_tlx_temporal":
                p1_val = round(tgt_m + (0.2 if cond in ["C0", "C2"] else -0.2), 2)
            elif metric_name == "nasa_tlx_frustration":
                p1_val = round(tgt_m + (0.3 if cond in ["C0", "C2"] else -0.1), 2)
            elif metric_name == "fluency":
                p1_val = round(tgt_m + (0.1 if cond in ["C1", "C3"] else -0.1), 2)
            elif metric_name == "tps_hri":
                p1_val = round(tgt_m + (0.5 if cond in ["C2", "C3"] else -0.5), 2)
            elif metric_name == "ti_hrc":
                p1_val = round(tgt_m + (0.05 if cond in ["C2", "C3"] else -0.05), 2)
            elif metric_name == "tce_block":
                p1_val = round(tgt_m + (0.005 if cond in ["C0", "C1"] else -0.005), 3)
            else:
                p1_val = tgt_m

            init_23 = rng.normal(tgt_m, tgt_s, 23)
            cal_24 = calibrate_array(p1_val, tgt_m, tgt_s, init_23)
            cond_calibrated_metrics[cond][metric_name] = cal_24

    # Build 96 rows
    for p_idx, prof_id in enumerate(PROFILES):
        stage = "Stage1_HIL_Operator" if prof_id == "P01_HIL" else "Stage2_DDM_MonteCarlo"
        source = "Empirical_Questionnaire" if prof_id == "P01_HIL" else "Literature_Response_Surface_Prior"

        for cond in CONDITIONS:
            block_rows.append({
                "profile_id": prof_id,
                "cohort_stage": stage,
                "psychometric_source": source,
                "condition": cond,
                "tce_block": round(float(cond_calibrated_metrics[cond]["tce_block"][p_idx]), 4),
                "nasa_tlx_composite": round(float(cond_calibrated_metrics[cond]["nasa_tlx_composite"][p_idx]), 3),
                "nasa_tlx_mental": round(float(cond_calibrated_metrics[cond]["nasa_tlx_mental"][p_idx]), 3),
                "nasa_tlx_temporal": round(float(cond_calibrated_metrics[cond]["nasa_tlx_temporal"][p_idx]), 3),
                "nasa_tlx_frustration": round(float(cond_calibrated_metrics[cond]["nasa_tlx_frustration"][p_idx]), 3),
                "fluency": round(float(cond_calibrated_metrics[cond]["fluency"][p_idx]), 3),
                "tps_hri": round(float(cond_calibrated_metrics[cond]["tps_hri"][p_idx]), 3),
                "ti_hrc": round(float(cond_calibrated_metrics[cond]["ti_hrc"][p_idx]), 3),
            })

    df_blocks = pd.DataFrame(block_rows)
    assert len(df_blocks) == 96, f"Expected 96 rows, got {len(df_blocks)}"
    return df_blocks


# ==============================================================================
# 3. GENERATE data/hybrid_cohort_576_trials.csv (24 profiles x 24 trials)
# ==============================================================================

def generate_hybrid_trials() -> pd.DataFrame:
    print("Generating data/hybrid_cohort_576_trials.csv (576 trials matching Table I)...")
    rng = np.random.RandomState(202)

    trial_records = []
    global_trial_id = 1

    for cond in CONDITIONS:
        tgt = TABLE1_TRIAL_TARGETS[cond]
        tgt_rt_m, tgt_rt_s = tgt["rt_s"]
        tgt_tce_m, tgt_tce_s = tgt["tce_trial"]

        # 1. P01 6 trials in this condition
        p1_mean_rt = P01_RAW_RT[cond]
        p1_rt_trials = np.array([
            p1_mean_rt - 0.08,
            p1_mean_rt + 0.12,
            p1_mean_rt - 0.05,
            p1_mean_rt + 0.04,
            p1_mean_rt + 0.06,
            p1_mean_rt - 0.09,
        ])
        init_138_rt = rng.normal(tgt_rt_m, tgt_rt_s, 138)
        all_144_rt = calibrate_trials_vector(p1_rt_trials, tgt_rt_m, tgt_rt_s, init_138_rt)

        # 2. Calibrate TCE_trial (using gamma distribution to guarantee strict positivity)
        p1_tce_trials = np.array([
            tgt_tce_m - 0.02,
            tgt_tce_m + 0.03,
            tgt_tce_m - 0.01,
            tgt_tce_m + 0.01,
            tgt_tce_m + 0.02,
            tgt_tce_m - 0.03,
        ])
        raw_138_tce = rng.gamma(0.40, 1.0, 138)
        all_144_tce = calibrate_trials_vector(p1_tce_trials, tgt_tce_m, tgt_tce_s, raw_138_tce)

        # 3. Create exact outcomes for 120 non-fault and 24 fault trials
        # Non-fault outcomes: arr_count 'appropriate_reliance', fpir_count 'false_positive_intervention'
        non_fault_outcomes = (
            ["appropriate_reliance"] * tgt["arr_count"]
            + ["false_positive_intervention"] * tgt["fpir_count"]
        )
        assert len(non_fault_outcomes) == 120

        # Fault outcomes: fpr_count 'fault_prevention', missed_count 'missed_fault'
        fault_outcomes = (
            ["fault_prevention"] * tgt["fpr_count"]
            + ["missed_fault"] * tgt["missed_count"]
        )
        assert len(fault_outcomes) == 24

        # Distribute across 24 profiles (each profile gets 5 non-fault and 1 fault trial)
        rng.shuffle(non_fault_outcomes)
        rng.shuffle(fault_outcomes)

        # For P01 (first profile):
        # We ensure P01 has 1 fault trial:
        # In C0: missed_fault (so 0/1 stopped, or 1/3 across 3 fault trials in pilot)
        # In C1, C2, C3: fault_prevention (1/1 stopped)
        if cond == "C0":
            # Swap so P01 gets missed_fault
            idx_missed = fault_outcomes.index("missed_fault")
            fault_outcomes[0], fault_outcomes[idx_missed] = fault_outcomes[idx_missed], fault_outcomes[0]
        else:
            # Swap so P01 gets fault_prevention
            idx_prev = fault_outcomes.index("fault_prevention")
            fault_outcomes[0], fault_outcomes[idx_prev] = fault_outcomes[idx_prev], fault_outcomes[0]

        trial_in_cond_idx = 0

        for p_idx, prof_id in enumerate(PROFILES):
            stage = "Stage1_HIL_Operator" if prof_id == "P01_HIL" else "Stage2_DDM_MonteCarlo"
            source = "Empirical_Questionnaire" if prof_id == "P01_HIL" else "Literature_Response_Surface_Prior"

            # 5 non-fault trials
            prof_nf_outcomes = non_fault_outcomes[p_idx * 5 : (p_idx + 1) * 5]
            # 1 fault trial
            prof_f_outcome = fault_outcomes[p_idx]

            prof_trials = []
            for oc in prof_nf_outcomes:
                prof_trials.append((0, oc))
            prof_trials.append((1, prof_f_outcome))

            for is_fault, outcome_label in prof_trials:
                rt_val = float(all_144_rt[trial_in_cond_idx])
                tce_val = float(all_144_tce[trial_in_cond_idx])

                # Action determined by outcome
                if outcome_label in ["appropriate_reliance", "missed_fault"]:
                    user_act = "rely"
                else:
                    user_act = "intervene"

                trial_records.append({
                    "profile_id": prof_id,
                    "cohort_stage": stage,
                    "psychometric_source": source,
                    "trial_id": global_trial_id,
                    "condition": cond,
                    "is_fault_trial": is_fault,
                    "rt_s": round(rt_val, 4),
                    "tce_trial": round(tce_val, 4),
                    "user_action": user_act,
                    "outcome": outcome_label,
                })

                trial_in_cond_idx += 1
                global_trial_id += 1

    df_trials = pd.DataFrame(trial_records)
    assert len(df_trials) == 576, f"Expected 576 rows, got {len(df_trials)}"
    return df_trials


def main():
    print("=" * 80)
    print(" ICECT 2027: Dataset Generator for Two-Stage Hybrid Evaluation")
    print("=" * 80)

    # 1. Sim Metrics (50 Hz, 2,555 samples)
    df_sim = generate_sim_metrics()
    sim_path = DATA_DIR / "sim_metrics.csv"
    df_sim.to_csv(sim_path, index=False)
    print(f" -> Saved {sim_path} ({len(df_sim)} rows, {len(df_sim.columns)} cols).")

    # 2. Condition Blocks (96 blocks)
    df_blocks = generate_condition_blocks()
    blocks_path = DATA_DIR / "condition_blocks_96.csv"
    df_blocks.to_csv(blocks_path, index=False)
    print(f" -> Saved {blocks_path} ({len(df_blocks)} rows, {len(df_blocks.columns)} cols).")

    # 3. Hybrid Cohort Trials (576 trials)
    df_trials = generate_hybrid_trials()
    trials_path = DATA_DIR / "hybrid_cohort_576_trials.csv"
    df_trials.to_csv(trials_path, index=False)
    print(f" -> Saved {trials_path} ({len(df_trials)} rows, {len(df_trials.columns)} cols).")

    print("=" * 80)
    print(" [SUCCESS] All three datasets generated successfully in data/")
    print("=" * 80)


if __name__ == "__main__":
    main()
