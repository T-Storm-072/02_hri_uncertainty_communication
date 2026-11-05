# Communicating Robot Perceptual Uncertainty in Collaborative Manipulation

**Official Companion Research Repository & Reproducibility Suite**  
*Submitted to the 5th International Conference on Engineering & Computing Technologies (ICECT 2027, NUML Islamabad)*

[![Conference](https://img.shields.io/badge/Conference-ICECT%202027%20(NUML)-00629B.svg)](https://numl.edu.pk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Simulation: Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic%208.15-orange.svg)](https://gazebosim.org)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproduction: Verified](https://img.shields.io/badge/Table%20I%20Reproduction-100%25%20Verified-brightgreen.svg)](analysis/reproduce_table1_stats.py)

---

## Authors & Affiliation
- **Syed Ali Turab Kazmi** — *Department of Robotics and Intelligent Systems, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, H-11 Campus, Islamabad, Pakistan* (`syedaliturab072@gmail.com`)
- **Dr. Muhammad Hasan Danish Khan** (*Corresponding Author*) — *Department of Electrical Engineering, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, H-11 Campus, Islamabad, Pakistan* (`hasandanish.buic@bahria.edu.pk`)

---

## Executive Summary
In close-proximity collaborative manipulation, conveying perceptual uncertainty is vital for preventing catastrophic overtrust (unwarranted reliance during failure modes) and productivity-destroying undertrust (premature human intervention). However, existing communication paradigms face a fundamental trade-off: **explicit external Human-Machine Interfaces (eHMI)** deliver rich diagnostic information but divert operator visual attention from the shared workspace, whereas **implicit kinematic cues** communicate naturally through motion but may lack diagnostic precision.

This repository provides the complete open-source implementation, simulation assets, vision pipelines, controllers, and verification suites for our paper:
1. **RGB-D Computer Vision Front-End (`vision/`):** Real-time 30 Hz streaming, 6-DoF pose covariance estimation, valid depth inlier tracking, and normalized Shannon perceptual entropy computation across $K = 4$ YCB benchmark classes.
2. **Kinematic Hesitation & eHMI Controller (`control/`):** Closed-form continuous modulation laws governing entropy-dependent velocity scaling (Eq. 4), pre-grasp dwell pausing (Eq. 5), micro-retreat retraction (Eq. 6), and chromatic/earcon eHMI encoding.
3. **Two-Stage Hybrid Evaluation Datasets (`data/`):** 50 Hz Gazebo Harmonic telemetry logs (2,555 samples) and 576 evaluation trials across a hybrid cohort ($1$ HIL operator across 24 counterbalanced trials + $23$ literature-calibrated DDM observer profiles across 552 simulated trials).
4. **One-Command Statistical Reproduction Suite (`analysis/`):** Strict verification asserting that every mean, standard deviation, and percentage in the dataset matches Table I of the paper with zero discrepancy.

---

## Repository Architecture
```
02_hri_uncertainty_icect2027/
├── README.md                                  # Comprehensive documentation & reproduction guide
├── LICENSE                                    # MIT Open Access License
├── vision/
│   └── rgbd_entropy_estimator.py              # 30 Hz RGB-D vision & Shannon entropy front-end
├── control/
│   ├── kinematic_hesitation_controller.py     # Eqs. (4)-(6) hesitation & multimodal eHMI controller
│   ├── sim_video.mp4                          # Composite 3D simulation video
│   ├── trial_c0_baseline.mp4                  # Trial recording: C0 Baseline
│   ├── trial_c1_implicit.mp4                  # Trial recording: C1 Implicit Kinematics
│   ├── trial_c2_explicit.mp4                  # Trial recording: C2 Explicit eHMI
│   ├── trial_c3_hybrid.mp4                    # Trial recording: C3 Hybrid Multimodal
│   └── worlds/
│       └── collaborative_tabletop.sdf         # High-fidelity Franka Emika Panda Gazebo world
├── data/
│   ├── sim_metrics.csv                        # 50 Hz telemetry log (2,555 samples, 51.10 s duration)
│   ├── condition_blocks_96.csv                # 96 block evaluations (24 profiles x 4 conditions)
│   ├── hybrid_cohort_576_trials.csv           # 576 trial logs (24 profiles x 24 trials)
│   └── statistical_results.json               # Full inferential ANOVA & t-test statistics
├── analysis/
│   ├── reproduce_table1_stats.py              # Automated Table I replication & assertion suite
│   └── run_statistical_analysis.py            # Repeated-measures ANOVA & paired t-tests
├── figures/
│   ├── fig1_profiles.png                      # Kinematic velocity & hesitation profiles
│   ├── fig2_gazebo.png                        # Gazebo Harmonic 3D simulation keyframes
│   ├── fig3_tce.png                           # Trust calibration error ANOVA distributions
│   └── fig4_workload.png                      # Cognitive workload & fluency trade-off Pareto
└── paper/
    ├── main.tex                               # Camera-ready IEEEtran LaTeX manuscript
    ├── paper.pdf                              # Compiled 6-page IEEE conference paper
    ├── references.bib                         # Verified BibTeX bibliography
    └── figures/                               # Embedded publication figures
```

---

## Mathematical Formulation

### 1. Normalized Shannon Perceptual Entropy ($u(t) \in [0, 1]$)
Given a posterior categorical distribution $P(c \mid \mathbf{z}_t)$ over candidate YCB objects $\mathcal{C} = \{c_1, \dots, c_K\}$ with $K = 4$:
$$u(t) = - \frac{1}{\log_2 K} \sum_{k=1}^{K} P(c_k \mid \mathbf{z}_t) \log_2 P(c_k \mid \mathbf{z}_t) \in [0, 1]$$

Industrial uncertainty is stratified into three actionable regimes:
- **Nominal Regime ($u \le 0.50$):** Standard minimum-jerk trajectory execution without distracting cues.
- **Moderate Ambiguity ($0.50 < u \le 0.70$):** Modulates approach kinematics (velocity scaling and dwell pause) within the shared workspace. Alerts the worker ambiently without elevating cognitive workload.
- **Critical Ambiguity ($u > 0.70$):** Escalates to multimodal communication, combining micro-retreat with explicit visual status confirmation and auditory prompts.

### 2. Kinematic Hesitation Control Law (Eqs. 4–6)
- **Velocity Damping (Eq. 4):**
  $$v(t) = v_{\text{nom}}(t) \cdot [1 - \alpha \cdot u(t)]$$
  with $v_{\text{nom}} = 0.35\text{ m/s}$ and $\alpha = 0.70$, slowing smoothly down to $0.105\text{ m/s}$ at $u = 1$.
- **Pre-Grasp Dwell Pause (Eq. 5):**
  $$\tau_{\text{pause}}(u) = [\tau_0 + \beta(u(t) - 0.50)] \cdot \mathbb{I}(u(t) > 0.50)$$
  with $\tau_0 = 0.80\text{ s}$ and $\beta = 2.40\text{ s}$, introducing an $0.80\text{--}2.00\text{ s}$ pause prior to grasp commitment.
- **Micro-Retreat Retraction (Eq. 6):**
  $$d_{\text{retreat}}(u) = \gamma \cdot \max(0, u(t) - 0.70)$$
  with $\gamma = 0.08\text{ m}$, yielding up to $24\text{ mm}$ of bodily retreat along the approach normal $\hat{\mathbf{n}}$.

### 3. Explicit eHMI Encoding
- **Certainty Percentage:** $C = (1 - u) \times 100\%$
- **RGB Status Bar:** Green ($u \le 0.50$), Amber ($0.50 < u \le 0.70$), Red ($u > 0.70$)
- **Acoustic Earcon:** $f(u) = 440 + 400(u - 0.50)\text{ Hz}$ for $u \ge 0.50$ (muted in nominal regime)
- **Speech Alert:** High ambiguity verbal advisory for $u > 0.65$.

---

## Verified Table I Reproduction

Every single metric reported in Table I of the paper is programmatically replicated from the raw evaluation CSVs.

```bash
python analysis/reproduce_table1_stats.py
```

### Official Table I Replication Output:
```
===================================================================================================================
               ICECT 2027: TABLE I REPRODUCTION & VERIFICATION SUITE
===================================================================================================================
[Stage 1 Verification] Human-in-the-Loop Operator (P01_HIL) Standalone Baseline:
  Condition C0: Standalone NASA-TLX = 44.8 (Target 44.8) | Standalone RT = 2.08s (Target 2.08s) [PASS]
  Condition C1: Standalone NASA-TLX = 27.5 (Target 27.5) | Standalone RT = 1.65s (Target 1.65s) [PASS]
  Condition C2: Standalone NASA-TLX = 44.2 (Target 44.2) | Standalone RT = 1.10s (Target 1.10s) [PASS]
  Condition C3: Standalone NASA-TLX = 26.8 (Target 26.8) | Standalone RT = 0.96s (Target 0.96s) [PASS]

[Stage 2 Verification] Consolidated Hybrid Cohort Table I Metrics (Mean +/- SD):
Metric                           | C0: Baseline       | C1: Implicit       | C2: Explicit       | C3: Multimodal     | Status
-------------------------------------------------------------------------------------------------------------------
TCE_trial (Trial Error)          | 0.274 +/- 0.284    | 0.202 +/- 0.228    | 0.207 +/- 0.220    | 0.177 +/- 0.207    | MATCH [OK]
TCE_block (Block Error)          | 0.104 +/- 0.071    | 0.200 +/- 0.102    | 0.119 +/- 0.047    | 0.053 +/- 0.028    | MATCH [OK]
NASA-TLX Total (0-100)           | 44.33 +/- 6.09     | 28.14 +/- 6.04     | 43.02 +/- 7.58     | 27.28 +/- 5.02     | MATCH [OK]
Mental Demand (0-100)            | 48.23 +/- 6.54     | 26.74 +/- 5.82     | 49.67 +/- 7.84     | 25.97 +/- 5.12     | MATCH [OK]
Temporal Demand (0-100)          | 41.88 +/- 5.91     | 30.21 +/- 5.43     | 47.12 +/- 6.95     | 26.17 +/- 4.88     | MATCH [OK]
Frustration (0-100)              | 48.70 +/- 6.88     | 25.06 +/- 5.11     | 38.79 +/- 6.22     | 22.21 +/- 4.45     | MATCH [OK]
Decision Reaction Time (s)       | 2.03 +/- 0.68      | 1.63 +/- 0.26      | 1.08 +/- 0.19      | 0.94 +/- 0.19      | MATCH [OK]
Hoffman Fluency (1-7)            | 3.90 +/- 0.46      | 5.32 +/- 0.55      | 5.28 +/- 0.51      | 6.58 +/- 0.29      | MATCH [OK]
TPS-HRI Trust Scale (0-100)      | 76.94 +/- 5.32     | 74.70 +/- 5.01     | 87.37 +/- 3.37     | 94.68 +/- 2.87     | MATCH [OK]
TI-HRC Industrial Trust (1-7)    | 4.19 +/- 0.52      | 5.43 +/- 0.44      | 5.77 +/- 0.41      | 6.50 +/- 0.35      | MATCH [OK]
-------------------------------------------------------------------------------------------------------------------

[Stage 3 Verification] Operational Event Rates & Counts (ARR%, FPR%, FPIR%):
Operational Rate                       | C0: Baseline     | C1: Implicit     | C2: Explicit     | C3: Multimodal   | Status
-------------------------------------------------------------------------------------------------------------------
Appropriate Reliance (ARR%)            |  95.0% (114/120) |  93.3% (112/120) |  96.7% (116/120) |  98.3% (118/120) | MATCH [OK]
Fault Prevention (FPR%)                |  33.3% (8/24)    |  87.5% (21/24)   |  91.7% (22/24)   | 100.0% (24/24)   | MATCH [OK]
False-Positive Interventions (FPIR%)   |   5.0% (6/120)   |   6.7% (8/120)   |   3.3% (4/120)   |   1.7% (2/120)   | MATCH [OK]
===================================================================================================================
 [VERIFICATION SUCCESS] Every single mean, SD, count, and percentage in the CSVs
 matches Table I of the ICECT 2027 paper with ZERO discrepancy.
===================================================================================================================
```

---

## Quickstart & Replication

### 1. Environment Setup
```bash
git clone https://github.com/T-Storm-072/02_hri_uncertainty_icect2027.git
cd 02_hri_uncertainty_icect2027
pip install numpy scipy pandas statsmodels matplotlib seaborn
```

### 2. Run Computer Vision Front-End Unit Tests
```bash
python vision/rgbd_entropy_estimator.py
```

### 3. Run Controller Law Assertions
```bash
python control/kinematic_hesitation_controller.py
```

### 4. Reproduce Inferential Statistics & ANOVA Tests
```bash
python analysis/run_statistical_analysis.py
```

---

## Citation
If you use this work, codebase, or dataset in your research, please cite our ICECT 2027 conference paper:

```bibtex
@inproceedings{kazmi2027communicating,
  author    = {Kazmi, Syed Ali Turab and Khan, Muhammad Hasan Danish},
  title     = {Communicating Robot Perceptual Uncertainty in Collaborative Manipulation: Comparing Implicit Kinematic Hesitation and Explicit Visual-Verbal Cues on Human Trust Calibration and Task Performance},
  booktitle = {Proceedings of the 5th International Conference on Engineering \& Computing Technologies (ICECT)},
  address   = {Islamabad, Pakistan},
  year      = {2027}
}
```

---

## License
This project is open-source software licensed under the [MIT License](LICENSE).
