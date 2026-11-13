"""Reproducible, non-gating 0.7.1 LM diagnostics.

Run from an editable Moro checkout:
    python scripts/diagnose_ik_071.py

This script calls the *actual Moro implementation*. It does not assert that
the 0.7.0 default solver succeeds: known failures are the object of study.
No production code or public IK return contracts are modified.
"""

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import sympy as sp

import moro.inverse_kinematics as ik
from moro.abc import q1, q2, q3
from moro.core import Robot


def cases(scale):
    L = 200.0 * scale
    return (
        ("RR", Robot((L, 0, 0, q1), (L, 0, 0, q2)),
         np.array([L, L, 0.0]), [0.1, 0.1]),
        ("RRR", Robot((0, sp.pi / 2, L, q1),
                      (L, 0, 0, q2), (L, 0, 0, q3)),
         np.array([L, L, L]), [0.5, 0.5, 0.5]),
    )


@contextmanager
def capture_lm_step_proposals():
    """Observe private LM step inputs without changing their output."""
    events = []
    original = ik._compute_ik_step

    def observed(J, residual, *, method, damping):
        step = original(J, residual, method=method, damping=damping)
        if method == "lm":
            events.append({
                "damping": float(damping),
                "jacobian_norm": float(np.linalg.norm(J)),
                "residual_norm": float(np.linalg.norm(residual)),
                "proposed_step_norm": float(np.linalg.norm(step)),
            })
        return step

    with patch.object(ik, "_compute_ik_step", side_effect=observed):
        yield events


def result_line(label, sol):
    return (f"{label:<26} converged={str(sol.converged):<5} "
            f"iters={sol.iterations:<3} error={sol.error:.8e} "
            f"message={sol.message}")


def main():
    for scale, unit in ((1.0, "mm"), (0.001, "m")):
        print(f"\n=== Numerical representation: {unit} ===")
        for name, robot, target, q0 in cases(scale):
            print(f"\n{name}: link size={200 * scale:g}, target={target}")
            for stall_limit in (5, 6, 10):
                with capture_lm_step_proposals() as events:
                    sol = ik.solve_position_ik(
                        robot, target, q0=q0,
                        stagnation_iterations=stall_limit,
                        # These are absolute Cartesian tolerances; rescale
                        # along with geometric lengths and target position.
                        tol=1e-6 * scale,
                        error_change_tol=1e-12 * scale,
                    )
                print(result_line(f"stagnation={stall_limit}", sol))
                if stall_limit == 6:
                    print("  first LM attempts: "
                          + ", ".join(f"{e['damping']:g}" for e in events[:12]))
                    print("  proposed step norms: "
                          + ", ".join(f"{e['proposed_step_norm']:.3g}" for e in events[:8]))

            # Fixed seeds: q0=None draws one initial configuration per run,
            # not a multi-start search. This is observational, not a CI gate.
            for stall_limit in (5, 6, 10):
                successes = 0
                iterations = []
                for seed in range(30):
                    sol = ik.solve_position_ik(
                        robot, target,
                        q0=None, random_state=seed,
                        tol=1e-6 * scale,
                        error_change_tol=1e-12 * scale,
                        stagnation_iterations=stall_limit,
                    )
                    successes += bool(sol.converged)
                    iterations.append(sol.iterations)
                print(f"  random_state=0..29, stagnation={stall_limit}: "
                      f"{successes}/30 converged, "
                      f"median iterations={np.median(iterations):.1f}")


if __name__ == "__main__":
    main()
