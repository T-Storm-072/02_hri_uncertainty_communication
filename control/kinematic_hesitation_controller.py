#!/usr/bin/env python3
"""
Kinematic hesitation and multimodal eHMI controller translating normalized Shannon
entropy into continuous approach deceleration, pre-grasp dwell pauses, and micro-retreats.

Key capabilities:
- Velocity Damping (Eq. 4): v(t) = v_{nom}(t) * [1 - alpha * u(t)]
- Pre-Grasp Dwell Pause (Eq. 5): tau_{pause}(u) = [tau_0 + beta * (u(t) - 0.50)] * I(u(t) > 0.50)
- Micro-Retreat Retraction (Eq. 6): d_{retreat}(u) = gamma * max(0, u(t) - 0.70)
- Explicit eHMI: Certainty percentage, chromatic RGB interpolation, acoustic earcon, and speech alerts.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class ControllerOutput:
    """Output commands from the hesitation and eHMI controller."""
    # Inputs
    u: float
    nominal_vel_m_s: float
    # Kinematic Modulations (Implicit bodily cues)
    commanded_vel_m_s: float
    velocity_damping_factor: float
    dwell_pause_s: float
    micro_retreat_m: float
    micro_retreat_displacement_m: np.ndarray  # 3D vector along retraction normal
    # Explicit eHMI Cues
    ehmi_certainty_pct: float
    ehmi_rgb_color: Tuple[int, int, int]
    ehmi_rgb_hex: str
    earcon_freq_hz: float
    speech_alert: Optional[str]
    regime: str


class KinematicHesitationController:
    """
    Continuous Information-Theoretic Controller translating normalized Shannon
    entropy u in [0, 1] into physical hesitation kinematics and multimodal eHMI cues.
    """

    def __init__(
        self,
        v_nom: float = 0.35,     # Nominal reach velocity [m/s]
        alpha: float = 0.70,     # Damping gain (alpha = 0.70 -> 0.105 m/s at u=1)
        tau_0: float = 0.80,     # Base dwell pause [s]
        beta: float = 2.40,      # Dwell pause slope [s]
        gamma: float = 0.08,     # Micro-retreat scale [m] (0.08 * 0.30 = 0.024 m = 24 mm)
    ):
        self.v_nom = float(v_nom)
        self.alpha = float(alpha)
        self.tau_0 = float(tau_0)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def compute_velocity(self, u: float, v_nominal: Optional[float] = None) -> Tuple[float, float]:
        """
        Velocity Damping (Eq. 4):
        v(t) = v_{nom}(t) * [1 - alpha * u(t)]
        Returns (commanded_vel, damping_factor).
        """
        u_clipped = float(np.clip(u, 0.0, 1.0))
        base_v = self.v_nom if v_nominal is None else float(v_nominal)
        factor = float(1.0 - self.alpha * u_clipped)
        v_cmd = float(base_v * factor)
        return v_cmd, factor

    def compute_dwell_pause(self, u: float) -> float:
        """
        Pre-Grasp Dwell Pause (Eq. 5):
        tau_{pause}(u) = [tau_0 + beta * (u(t) - 0.50)] * I(u(t) > 0.50)
        """
        u_clipped = float(np.clip(u, 0.0, 1.0))
        if u_clipped > 0.50:
            tau = self.tau_0 + self.beta * (u_clipped - 0.50)
            return float(round(tau, 4))
        return 0.0

    def compute_micro_retreat(
        self, u: float, normal_hat: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Micro-Retreat Retraction (Eq. 6):
        d_{retreat}(u) = gamma * max(0, u(t) - 0.70)
        Returns (distance_in_meters, 3D_displacement_vector).
        """
        u_clipped = float(np.clip(u, 0.0, 1.0))
        if normal_hat is None:
            # Default retreat along -Z or approach normal [0, 0, -1]
            n_hat = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            norm = np.linalg.norm(normal_hat)
            n_hat = normal_hat / max(norm, 1e-6)

        retreat_dist = float(self.gamma * max(0.0, u_clipped - 0.70))
        retreat_vec = retreat_dist * n_hat
        return retreat_dist, retreat_vec

    def compute_ehmi(self, u: float) -> Tuple[float, Tuple[int, int, int], str, float, Optional[str]]:
        """
        Explicit eHMI Encoding:
        - Certainty percentage: C = (1 - u) * 100%
        - RGB status bar color interpolation
        - Acoustic earcon frequency: f(u) = 440 + 400 * (u - 0.50) Hz for u >= 0.50
        - Speech alert for u > 0.65
        """
        u_clipped = float(np.clip(u, 0.0, 1.0))
        certainty_pct = float(round((1.0 - u_clipped) * 100.0, 2))

        # Chromatic RGB Status Bar Interpolation
        if u_clipped <= 0.50:
            # Nominal: Pure Green (0, 255, 0)
            r, g, b = 0, 255, 0
        elif u_clipped <= 0.70:
            # Moderate: Amber / Yellow (255, 191, 0)
            fraction = (u_clipped - 0.50) / 0.20
            r = int(0 + fraction * 255)
            g = int(255 - fraction * (255 - 191))
            b = 0
        else:
            # Critical: Red (255, 0, 0)
            fraction = min(1.0, (u_clipped - 0.70) / 0.30)
            r = 255
            g = int(191 * (1.0 - fraction))
            b = 0

        hex_color = f"#{r:02X}{g:02X}{b:02X}"

        # Acoustic Earcon Frequency
        if u_clipped >= 0.50:
            earcon_f = float(round(440.0 + 400.0 * (u_clipped - 0.50), 2))
        else:
            earcon_f = 0.0  # Muted in nominal regime

        # Speech Alert
        speech = None
        if u_clipped > 0.65:
            speech = "Caution: High ambiguity in grasp classification. Human verification advised."

        return certainty_pct, (r, g, b), hex_color, earcon_f, speech

    def step(
        self,
        u: float,
        v_nominal: Optional[float] = None,
        retraction_normal: Optional[np.ndarray] = None,
    ) -> ControllerOutput:
        """Evaluates complete controller response for uncertainty index u."""
        u_val = float(np.clip(u, 0.0, 1.0))
        base_v = self.v_nom if v_nominal is None else float(v_nominal)

        # Kinematics
        v_cmd, damp_factor = self.compute_velocity(u_val, base_v)
        tau_pause = self.compute_dwell_pause(u_val)
        d_retreat, retreat_vec = self.compute_micro_retreat(u_val, retraction_normal)

        # eHMI
        c_pct, rgb, hex_c, earcon_hz, speech = self.compute_ehmi(u_val)

        # Regime tag
        if u_val <= 0.50:
            regime_str = "Nominal Certain (u <= 0.50)"
        elif u_val <= 0.70:
            regime_str = "Moderate Ambiguity (0.50 < u <= 0.70)"
        else:
            regime_str = "Critical Ambiguity (u > 0.70)"

        return ControllerOutput(
            u=round(u_val, 4),
            nominal_vel_m_s=round(base_v, 4),
            commanded_vel_m_s=round(v_cmd, 4),
            velocity_damping_factor=round(damp_factor, 4),
            dwell_pause_s=tau_pause,
            micro_retreat_m=round(d_retreat, 4),
            micro_retreat_displacement_m=retreat_vec,
            ehmi_certainty_pct=c_pct,
            ehmi_rgb_color=rgb,
            ehmi_rgb_hex=hex_c,
            earcon_freq_hz=earcon_hz,
            speech_alert=speech,
            regime=regime_str,
        )


# ==============================================================================
# VERIFICATION CLI & UNIT TESTS
# ==============================================================================

def main():
    print("=" * 80)
    print(" Kinematic Hesitation & Multimodal eHMI Controller Verification")
    print(" Equations: Eq. (4) Velocity Damping | Eq. (5) Dwell Pause | Eq. (6) Micro-Retreat")
    print("=" * 80)

    controller = KinematicHesitationController(v_nom=0.35, alpha=0.70, tau_0=0.80, beta=2.40, gamma=0.08)

    # 1. Assert Velocity Damping Eq. (4)
    # At u = 0.0: v = 0.35 * (1 - 0) = 0.35 m/s
    v0, _ = controller.compute_velocity(0.0)
    assert abs(v0 - 0.35) < 1e-6, f"Expected 0.35 at u=0, got {v0}"

    # At u = 0.5: v = 0.35 * (1 - 0.70 * 0.5) = 0.35 * 0.65 = 0.2275 m/s
    v_mid, _ = controller.compute_velocity(0.50)
    assert abs(v_mid - 0.2275) < 1e-6, f"Expected 0.2275 at u=0.5, got {v_mid}"

    # At u = 1.0: v = 0.35 * (1 - 0.70 * 1.0) = 0.35 * 0.30 = 0.105 m/s
    v1, _ = controller.compute_velocity(1.0)
    assert abs(v1 - 0.105) < 1e-6, f"Expected 0.105 at u=1, got {v1}"
    print("[PASS] Eq. (4) Velocity Damping verified across full dynamic range.")

    # 2. Assert Pre-Grasp Dwell Pause Eq. (5)
    # At u <= 0.50: tau = 0.0
    assert controller.compute_dwell_pause(0.50) == 0.0, "Expected tau=0.0 at u=0.50"
    assert controller.compute_dwell_pause(0.30) == 0.0, "Expected tau=0.0 at u=0.30"

    # At u = 0.51: tau = 0.80 + 2.40 * 0.01 = 0.824 s
    tau_51 = controller.compute_dwell_pause(0.51)
    assert abs(tau_51 - 0.824) < 1e-3, f"Expected 0.824 at u=0.51, got {tau_51}"

    # At u = 1.0: tau = 0.80 + 2.40 * 0.50 = 2.00 s
    tau_1 = controller.compute_dwell_pause(1.0)
    assert abs(tau_1 - 2.00) < 1e-6, f"Expected 2.00 at u=1.0, got {tau_1}"
    print("[PASS] Eq. (5) Pre-Grasp Dwell Pause verified (0.80--2.00 s).")

    # 3. Assert Micro-Retreat Retraction Eq. (6)
    # At u <= 0.70: d = 0.0
    d_70, _ = controller.compute_micro_retreat(0.70)
    assert d_70 == 0.0, f"Expected d=0.0 at u=0.70, got {d_70}"

    # At u = 1.0: d = 0.08 * (1.0 - 0.70) = 0.024 m = 24 mm
    d_1, _ = controller.compute_micro_retreat(1.0)
    assert abs(d_1 - 0.024) < 1e-6, f"Expected 0.024 m (24 mm) at u=1.0, got {d_1}"
    print("[PASS] Eq. (6) Micro-Retreat verified (up to 24 mm).")

    # 4. Assert eHMI Cues
    # At u = 0.20: C = 80%, Earcon = 0 Hz, Speech = None
    c, rgb, _, earcon, speech = controller.compute_ehmi(0.20)
    assert c == 80.0 and earcon == 0.0 and speech is None

    # At u = 0.50: Earcon = 440 Hz
    _, _, _, earcon_50, _ = controller.compute_ehmi(0.50)
    assert abs(earcon_50 - 440.0) < 1e-3

    # At u = 1.0: Earcon = 440 + 400 * 0.5 = 640 Hz, Speech active
    c_1, _, _, earcon_1, speech_1 = controller.compute_ehmi(1.0)
    assert c_1 == 0.0 and abs(earcon_1 - 640.0) < 1e-3 and speech_1 is not None
    print("[PASS] Multimodal eHMI encoding verified (Certainty %, Earcon 440--640 Hz, Speech alerts).")

    # 5. Full step walkthrough
    print("\n--- Controller Step Evaluation Across Escalation Spectrum ---")
    test_u_values = [0.10, 0.45, 0.55, 0.65, 0.75, 0.95]
    for u_val in test_u_values:
        out = controller.step(u_val)
        print(
            f"u={out.u:4.2f} | v_cmd={out.commanded_vel_m_s:5.3f} m/s | "
            f"Pause={out.dwell_pause_s:4.2f}s | Retreat={out.micro_retreat_m*1000:4.1f}mm | "
            f"eHMI={out.ehmi_certainty_pct:5.1f}% ({out.ehmi_rgb_hex}) | "
            f"Earcon={out.earcon_freq_hz:5.1f}Hz | {out.regime}"
        )

    print("\n" + "=" * 80)
    print(" [OK] Kinematic hesitation controller verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
