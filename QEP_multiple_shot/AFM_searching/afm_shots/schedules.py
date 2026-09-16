"""Time-schedules for the open-chain AFM model.

Ported verbatim from Adaptive_bn/qep_shot/schedules.py (`ascend_func` and
`make_modulation_funcs` only -- the H_learning-side `fA`/`fB` schedules are not needed
here). See that file for the derivation/history notes.
"""
import jax.numpy as jnp


def ascend_func(x):
    """Smooth ramp on [0,1] (zero first derivative at both ends), clamped outside."""
    return jnp.where(
        x <= 0,
        0.0,
        jnp.where(x >= 1, 1.0, (-jnp.cos(x * jnp.pi) + 1.) / 2),
    )


def make_modulation_funcs(ascend_func=ascend_func):
    """(Omega, Delta) adiabatic ramp + nudge-strength ramp."""
    def func_Omega(t, Omega, T):
        Omega_0 = 1.0
        x1 = t / (0.2 * T)                 # ramp from 0 to Omega_0 over [0, 0.2T]
        x3 = (t - 0.8 * T) / (0.2 * T)     # ramp from Omega_0 to Omega over [0.8T, T]
        y1 = Omega_0 * ascend_func(x1)
        y2 = Omega_0
        y3 = Omega_0 + (Omega - Omega_0) * ascend_func(x3)
        return jnp.where(t < 0.2 * T, y1, jnp.where(t < 0.8 * T, y2, y3))

    def func_Delta(t, Delta, T):
        D0 = -1.0
        return D0 + (Delta - D0) * ascend_func((t - 0.2 * T) / (0.6 * T))

    def nudge_func(t, beta, T_nudge):
        return beta * ascend_func(t / T_nudge)

    return func_Omega, func_Delta, nudge_func
