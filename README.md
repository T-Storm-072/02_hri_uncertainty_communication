# Communicating Robot Perceptual Uncertainty in Collaborative Manipulation

**Master Research Companion Repository & Full Archival Edition**

[![IEEE Preprint](https://img.shields.io/badge/IEEE-Preprint-blue.svg)](paper/paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Simulation: Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic%208.15-orange.svg)](https://gazebosim.org)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Authors
- **Syed Ali Turab Kazmi** — *Department of Robotics and Intelligent Systems, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, H-11 Campus, Islamabad, Pakistan* (`syedaliturab072@gmail.com`)
- **Dr. Muhammad Hasan Danish Khan** (*Corresponding Author*) — *Department of Electrical Engineering, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, H-11 Campus, Islamabad, Pakistan* (`hasandanish.buic@bahria.edu.pk`)

---

## Comprehensive Abstract
In close-proximity human-robot collaborative manipulation, conveying perceptual uncertainty is critical for preventing catastrophic overtrust (unwarranted reliance during failure modes) and productivity-destroying undertrust (premature human intervention). However, existing communication paradigms face a fundamental trade-off: explicit external Human-Machine Interfaces (eHMI) deliver rich diagnostic information but divert operator visual attention from the shared manipulation workspace, whereas implicit kinematic cues communicate naturally through motion but may lack diagnostic precision. In this work, we present an information-theoretic framework for robot uncertainty communication and evaluate it through a systematic $2 \times 2$ factorial experiment comparing implicit kinematic hesitation (entropy-modulated velocity scaling, pre-grasp dwell pause, and micro-retreat) against explicit visual-verbal eHMI (chromatic status bars, confidence percentages, and acoustic earcons). Evaluated across $N = 24$ observer profiles ($576$ trials total) combining synchronous Human-in-the-Loop operator validation and literature-calibrated drift-diffusion models in high-fidelity 50 Hz Gazebo Harmonic simulations with a 7-DoF Franka Emika Panda robot, our findings reveal striking behavioral dissociations. Explicit eHMI produces a severe $52.9\%$ surge in subjective cognitive workload (NASA-TLX: $43.02 \pm 3.79$ vs. $28.14 \pm 4.38$, $p < 10^{-8}$, Cohen's $d = -2.03$) caused by the visual split-attention effect. Conversely, implicit kinematic hesitation preserves attentional focus within the physical workspace while substantially improving trust calibration. Most notably, congruent multimodal communication achieves optimal performance: cutting block-level error ($\text{TCE}_{\text{block}} = 0.053 \pm 0.028$, $p < 10^{-5}, d = 1.37$), eliminating missed collisions ($100\%$ fault prevention rate vs. $33.3\%$ in baseline), reducing false-positive interventions to $1.7\%$, and accelerating decision reaction time to $0.94$\,s ($p < 10^{-11}$).

---

## Master Repository Structure
```
├── README.md                           # Master Documentation & Reproducibility Guide
├── LICENSE                             # MIT Open Access License
├── paper/
│   ├── main.tex                        # Full 9-page IEEEtran journal manuscript source
│   ├── paper.pdf                       # Compiled 9-page clean PDF (clean headers, no watermark)
│   ├── references.bib                  # Peer-reviewed BibTeX library
│   ├── literature_log.json             # Literature search & review audit log
│   ├── main.log                        # Complete compilation transcript
│   └── figures/                        # High-resolution vector/raster figures
├── figures/                            # Publication-grade figures (Fig 1–4)
└── sim/
    ├── generate_hri_sim_and_telemetry.py # 50 Hz Gazebo kinematics & DDM observer simulator
    ├── run_statistical_analysis.py       # Repeated-measures ANOVA & Welch t-tests
    ├── render_sim_video.py               # 3D keyframe & simulation video generator
    ├── sim_metrics.csv                   # High-frequency trajectory telemetry
    ├── hri_trials.csv                    # Participant trial log (576 trials, N=24)
    ├── hri_surveys.csv                   # NASA-TLX & Trust Perception surveys
    ├── statistical_results.json          # Complete inferential statistics & p-values
    ├── sim_video.mp4                     # Full composite simulation demo video
    ├── trial_c0_baseline.mp4             # Condition C0 trial recording
    ├── trial_c1_implicit.mp4             # Condition C1 trial recording
    ├── trial_c2_explicit.mp4             # Condition C2 trial recording
    ├── trial_c3_hybrid.mp4               # Condition C3 trial recording
    └── worlds/
        └── collaborative_tabletop.sdf    # High-fidelity Franka Panda Gazebo Harmonic SDF world
```

---

## License
This project is open-source under the MIT License.
