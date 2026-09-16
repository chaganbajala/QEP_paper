"""Time schedules for the adiabatic sweep and for switching on the QEP nudge.

Two families are provided:

* **smooth** -- built from ``ascend_func`` (a raised cosine), whose first derivative
  vanishes at both ends of every ramp.  Because ``d H / d tau = 0`` at ``tau = 0``,
  the boundary coefficient ``B_{n0}(0)`` of QEP_GPT-12.pdf Eq. (33) vanishes and the
  O(1) gradient bias is suppressed.
* **stiff** -- piecewise-linear ramps.  ``d H / d tau != 0`` at ``tau = 0``, so the
  O(1) oscillating bias survives at arbitrarily large T.

Both families use the same three-stage (Omega, Delta) sweep of the cluster project
(``/u/qinwang/notebooks/QEP_sparse/fixTN_T=*_gd.py``):

    Omega: 0 -> 1 on [0, 0.2 T],  held at 1 on [0.2 T, 0.8 T],  1 -> Omega on [0.8 T, T]
    Delta: -1 -> Delta on [0.2 T, 0.8 T] (held at the end points outside)
"""
import jax.numpy as jnp

__all__ = [
    "ascend_func", "linear_func",
    "make_modulation_funcs", "SMOOTH", "STIFF", "get_schedule",
]


def ascend_func(x):
    """Smooth 0 -> 1 ramp on [0, 1] with vanishing first derivative at both ends."""
    return jnp.where(
        x <= 0, 0.0,
        jnp.where(x >= 1, 1.0, 0.5 * (1.0 - jnp.cos(jnp.pi * x))),
    )


def linear_func(x):
    """Piecewise-linear ("stiff") 0 -> 1 ramp on [0, 1]; kinked at both ends."""
    return jnp.where(x <= 0, 0.0, jnp.where(x >= 1, 1.0, x))


def make_modulation_funcs(ramp=ascend_func):
    """Return ``(func_Omega, func_Delta, nudge_func)`` built from a 0->1 ramp."""

    def func_Omega(t, Omega, T):
        Omega_0 = 1.0
        y1 = Omega_0 * ramp(t / (0.2 * T))
        y2 = Omega_0
        y3 = Omega_0 + (Omega - Omega_0) * ramp((t - 0.8 * T) / (0.2 * T))
        return jnp.where(t < 0.2 * T, y1, jnp.where(t < 0.8 * T, y2, y3))

    def func_Delta(t, Delta, T):
        D0 = -1.0
        return D0 + (Delta - D0) * ramp((t - 0.2 * T) / (0.6 * T))

    def nudge_func(t, beta, T_nudge):
        return beta * ramp(t / T_nudge)

    return func_Omega, func_Delta, nudge_func


SMOOTH = make_modulation_funcs(ascend_func)
STIFF = make_modulation_funcs(linear_func)


def get_schedule(name):
    """``name`` in {"smooth", "stiff"} -> ``(func_Omega, func_Delta, nudge_func)``."""
    table = {"smooth": SMOOTH, "stiff": STIFF}
    if name not in table:
        raise ValueError(f"unknown schedule {name!r}; expected one of {sorted(table)}")
    return table[name]
