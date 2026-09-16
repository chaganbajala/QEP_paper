"""Boundary cancellation: STIFF vs SMOOTH switch-on, and the prepared ground state -- 2026-09-09.

Context (QEP_GPT-12.pdf). The first-order adiabatic amplitude is, after one integration by
parts, a pure BOUNDARY term

    a^(1)_n(1) ~ -i hbar [ B_n0(1) e^{iT(gamma_n-gamma_0)} - B_n0(0) ] ,
    B_n0(tau) = <phi_n| d_tau H |phi_0> / (E_0-E_n)^2 ,

so everything hinges on d_tau H AT THE ENDPOINTS. For this project's ramp

    H(tau) = -[A0 + (A-A0) f(tau)] sigma^z - [B0 + (B-B0) f(tau)] sigma^x
    =>  d_tau H = -f'(tau) [ (A-A0) sigma^z + (B-B0) sigma^x ]   ~   f'(tau)

the boundary coefficient is EXACTLY proportional to the schedule slope f'(0), f'(1). The two
schedules of networks.py are

    ascend_func(x)    = (1 - cos pi x)/2   (smooth)  f'(0) = f'(1) = 0     -> B_n0(0)=B_n0(1)=0
    ascend_func_st(x) = x                  (stiff)   f'(0) = f'(1) = 1     -> B_n0(0),B_n0(1) != 0

identical to qep.ramp(x,'smooth') and qep.ramp(x,'linear').

PREDICTION for the state (as opposed to the gradient). |a^(1)|/T is the excited admixture, so
the prep infidelity is 1-F = |a^(1)|^2/T^2:
  stiff   B_n0(0) != 0 survives  ->  1-F ~ 1/T^2, oscillating in T at period 2pi/Delta gamma
                                     between (|B(1)|-|B(0)|)^2 and (|B(1)|+|B(0)|)^2 over T^2;
  smooth  the whole boundary term cancels -> the next order takes over -> 1-F ~ 1/T^4.
(This is Campos Venuti-Lidar boundary cancellation; the note's Sec. 4.4 is the general
statement: d_tau^k H = 0 for k=1..N at the ends => error o(1/T^(N+1)).)
Note the state error decays either way -- it is the GRADIENT that keeps an O(1) bias, because
d_theta hits the dynamical phase e^{-iT gamma} and pulls out a factor T. This driver measures
the STATE; the gradient is run_bandwidth.py.

Setup: fx = 0 (no probe), start (A0,B0) = (1,0) i.e. H(0) = -sigma^z, evolve to (A,B), and
overlap the final state with the EXACT ground state of -A sigma^z - B sigma^x.

Numerics: tol = 1e-13, not the study-wide 1e-9 -- the smooth 1/T^4 curve reaches ~1e-8 by
T=100 and saturates on a 1e-9 solver (measured: 5.7e-8 instead of 1.0e-8). exp_prep_fid_vs_T
saves an explicit tolerance-convergence check at the largest T.

Run:  PYTHONPATH=. python -u run_boundary.py [all|<names>]
Outputs ../data/bd_*.npz  (consumed by draw_boundary_figs.py).
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import dynamiqs as dq
import qep

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)

FX = 0.0                                        # no probe field
NS = 2                                          # endpoints only; adaptive solver
TOL = 1e-13                                     # see module docstring
SCHEDS = ("linear", "smooth")                   # = ascend_func_st, ascend_func
POINTS = np.array([[0.8, 1.5],                  # primary: the study target h
                   [1.2, 0.4]])                 # genericity check (the bandwidth-study point)
T_MIN, T_MAX, NT = 3.0, 160.0, 629              # dT = 0.25, ~10 samples per oscillation
T_FIT_MIN = 20.0                                # log-log slope fitted on T >= this

# --- exp_grad_vs_T (Fig. 1(b) of the note) -------------------------------------------------
EPS = 1e-4                                      # central-difference step in (A,B)
TG_MIN, TG_MAX, NTG = 2.0, 160.0, 633           # dT = 0.25
GSCHEDS = {"linear": "linear", "smooth": "smooth"}


# --- the schedule series shared by both gradient figures ------------------------------------
LAM_SET = (0.1, 0.5, 1.0, 5.0, 20.0, 100.0)     # the parametrized_ascend family
QBETA = 0.02                                    # the study-wide nudge strength


def grad_series():
    """(tag, schedule) for every curve in the two gradient figures: the lambda family, then the
    two schedules the note itself uses -- the cosine ramp and the stiff ramp."""
    out = [(f"lam{j}", qep.sched_param(l)) for j, l in enumerate(LAM_SET)]
    out += [("smooth", "smooth"), ("linear", "linear")]
    return out


def save(name, **kw):
    np.savez(os.path.join(DATA, name + ".npz"), **kw)
    print("   saved data/%s.npz" % name, flush=True)


def infidelity(p, T, schedule, tol=TOL):
    """1 - |<phi_0(A,B) | psi(T)>|^2 for the free (beta=0), unprobed (fx=0) evolution."""
    psi = qep.equilibrium(p, FX, T, 0.0, 0.0, NS, "adiabatic", schedule, tol=tol)
    phi = qep.gs_of(-p[0] * qep.SZ - p[1] * qep.SX)
    ov = jnp.abs(jnp.vdot(phi.to_jax().ravel(), psi.to_jax().ravel())) ** 2
    return 1.0 - ov


def dgamma(p, schedule, n=200001):
    """Dimensionless dynamical-phase difference  int_0^1 (E_1-E_0) dtau = int_0^1 2R dtau.

    The 1/T boundary term carries e^{-iT(gamma_1-gamma_0)}, so the infidelity oscillates in T
    with period 2 pi / dgamma. Used to size the envelope window.
    """
    tau = jnp.linspace(0.0, 1.0, n)
    lam = qep.ramp(tau, schedule)
    A = qep.A0 + (p[0] - qep.A0) * lam
    B = qep.B0 + (p[1] - qep.B0) * lam
    return float(jnp.trapezoid(2.0 * jnp.sqrt(A ** 2 + B ** 2), tau))


def envelope(y, w):
    """Rolling RMS of y over a centred window of w samples (edges use the truncated window)."""
    w = max(3, int(w) | 1)
    pad = w // 2
    yy = np.pad(y ** 2, pad, mode="reflect")
    ker = np.ones(w) / w
    return np.sqrt(np.convolve(yy, ker, mode="valid"))


def measured_period(T, y, tmin):
    """Mean spacing of the local maxima of y(T) for T >= tmin.

    Independent check of the theory: the surviving boundary term carries
    e^{-iT(gamma_1-gamma_0)}, so the infidelity must oscillate in T at period 2 pi / dgamma.
    Returns nan if fewer than three maxima are found (the smooth schedule has no such term).
    """
    m = T >= tmin
    t, v = T[m], y[m]
    i = np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
    return float(np.mean(np.diff(t[i]))) if len(i) >= 3 else float("nan")


def loglog_slope(T, y, tmin):
    """Least-squares d log y / d log T over T >= tmin."""
    m = (T >= tmin) & (y > 0)
    return float(np.polyfit(np.log(T[m]), np.log(y[m]), 1)[0])


def exp_schedule_shapes():
    """The two [0,1]->[0,1] schedules, their slopes, and the boundary coefficient |B_10(tau)|.

    Panel-3 quantity is the actual object of Eq. (34) of the note,
        B_10(tau) = <phi_1| d_tau H |phi_0> / (E_0-E_1)^2 ,
    evaluated along the ramp at the primary (A,B). It is what must vanish at tau=0 (and, for
    the T-oscillating part, at tau=1); it is proportional to f'(tau) by construction.
    """
    x = np.linspace(0.0, 1.0, 1001)
    out = {"x": x, "points": POINTS, "primary": POINTS[0]}

    for s in SCHEDS:
        out[f"f_{s}"] = np.array(qep.ramp(jnp.asarray(x), s))
    # analytic slopes/curvatures (jnp.clip makes autodiff unreliable exactly at x=0,1)
    out["df_linear"] = np.ones_like(x)
    out["df_smooth"] = 0.5 * np.pi * np.sin(np.pi * x)
    out["d2f_linear"] = np.zeros_like(x)
    out["d2f_smooth"] = 0.5 * np.pi ** 2 * np.cos(np.pi * x)
    # verify the analytic slopes against autodiff of the shipped qep.ramp (interior only)
    xi = jnp.linspace(0.02, 0.98, 97)
    for s in SCHEDS:
        ad = np.array(jax.vmap(jax.grad(lambda z: qep.ramp(z, s)))(xi))
        an = np.interp(np.array(xi), x, out[f"df_{s}"])
        err = float(np.max(np.abs(ad - an)))
        print(f"   {s:6s}: max|autodiff f' - analytic f'| = {err:.2e}", flush=True)
        assert err < 1e-9
        out[f"dfcheck_{s}"] = err

    # boundary coefficient B_10(tau) along the ramp, at the primary point
    p = POINTS[0]
    dA, dB = p[0] - qep.A0, p[1] - qep.B0
    for s in SCHEDS:
        lam = np.array(qep.ramp(jnp.asarray(x), s))
        A = qep.A0 + dA * lam
        B = qep.B0 + dB * lam
        dtH = -out[f"df_{s}"][:, None, None] * (dA * np.array([[1, 0], [0, -1]])
                                                + dB * np.array([[0, 1], [1, 0]]))
        Bc = np.empty_like(x)
        for i in range(len(x)):
            Hm = -A[i] * np.array([[1, 0], [0, -1]]) - B[i] * np.array([[0, 1], [1, 0]])
            w, v = np.linalg.eigh(Hm)
            Bc[i] = abs(v[:, 1].conj() @ dtH[i] @ v[:, 0]) / (w[0] - w[1]) ** 2
        out[f"B10_{s}"] = Bc
        print(f"   {s:6s}: |B_10(0)| = {Bc[0]:.4f}, |B_10(1)| = {Bc[-1]:.4f}", flush=True)
    save("bd_schedule_shapes", **out)


def exp_prep_fid_vs_T():
    """Prep fidelity vs T for both schedules: 1-F ~ 1/T^2 (stiff) vs 1/T^4 (smooth)."""
    Ts = np.linspace(T_MIN, T_MAX, NT)
    out = {"T": Ts, "points": POINTS, "tol": TOL, "fx": FX, "fit_Tmin": T_FIT_MIN,
           "start": np.array([qep.A0, qep.B0])}
    for j, p in enumerate(POINTS):
        pj = jnp.asarray(p)
        for s in SCHEDS:
            t0 = time.time()
            f = jax.jit(jax.vmap(lambda T: infidelity(pj, T, s)))
            inf = np.array(f(jnp.asarray(Ts)))
            dg = dgamma(p, s)
            per = 2.0 * np.pi / dg
            env = envelope(inf, round(per / (Ts[1] - Ts[0])))
            sl = loglog_slope(Ts, env, T_FIT_MIN)
            out[f"inf_{s}_p{j}"] = inf
            out[f"env_{s}_p{j}"] = env
            out[f"dgamma_{s}_p{j}"] = dg
            out[f"period_{s}_p{j}"] = per
            out[f"slope_{s}_p{j}"] = sl
            pm = measured_period(Ts, inf, T_FIT_MIN)
            out[f"period_meas_{s}_p{j}"] = pm
            print(f"   p=({p[0]:.1f},{p[1]:.1f}) {s:6s}: 1-F  {inf[0]:.3e} (T={T_MIN:g})"
                  f" -> {inf[-1]:.3e} (T={T_MAX:g});  envelope slope on T>={T_FIT_MIN:g}:"
                  f" {sl:+.3f};  period  predicted 2pi/dgamma={per:.3f}"
                  f"  measured={pm:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # tolerance convergence at the largest T, smooth schedule (the smallest quantity measured)
    tols = np.array([1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14])
    tc = np.array([float(infidelity(jnp.asarray(POINTS[0]), T_MAX, "smooth", tol=float(t)))
                   for t in tols])
    out["tolcheck_tols"] = tols; out["tolcheck_inf"] = tc; out["tolcheck_T"] = T_MAX
    print("   tolerance check (smooth, p=%s, T=%g):" % (POINTS[0], T_MAX), flush=True)
    for t, v in zip(tols, tc):
        print(f"      tol={t:.0e} -> 1-F = {v:.6e}", flush=True)
    save("bd_prep_fid_vs_T", **out)


def y_adiabatic(p, T, schedule, tol=TOL):
    """Output y = <sigma^z> of the adiabatic final state (free phase, fx=0)."""
    psi = qep.equilibrium(p, FX, T, 0.0, 0.0, NS, "adiabatic", schedule, tol=tol)
    return jnp.real(dq.expect(qep.SZ, psi))


def grad_adiabatic(pv, T, schedule):
    """Central-difference d y_ad / d(A,B) -- the note's own estimator for d_theta <sigma^z>."""
    out = []
    for i in (0, 1):
        yp = y_adiabatic(pv.at[i].add(EPS), T, schedule)
        ym = y_adiabatic(pv.at[i].add(-EPS), T, schedule)
        out.append((yp - ym) / (2.0 * EPS))
    return jnp.stack(out)


def exp_grad_vs_T():
    """Fig. 1(b) of the note, in this project's model at (A,B)=(0.8,1.5), fx=0.

    y = <sigma^z> of the adiabatic final state; the parameter gradient d y / d(A,B) is taken by
    central difference exactly as the note does ("the difference of <sigma^z> between theta-dtheta
    and theta+dtheta"). Reference is the EXACT ground state, for which y = A/sqrt(A^2+B^2) has the
    closed form

        dy/dA = B^2 / R^3 ,   dy/dB = -A B / R^3 ,   R = sqrt(A^2+B^2).

    PREDICTION (Eq. 33). The deviation is
        2 Re sum_n hbar (d_theta gamma_n - d_theta gamma_0) B_n0(0) e^{-iT(gamma_n-gamma_0)} O_n0
    -> O(1) and OSCILLATING for the stiff schedule (B_n0(0) != 0), and NOT decaying with T; for a
    schedule with lambda'(0)=lambda'(1)=0 the boundary term cancels and the residual is O(1/T).
    This is the note's headline: infinite T does not guarantee an accurate parameter gradient.
    """
    p = POINTS[0]
    A, B = float(p[0]), float(p[1])
    R = float(np.hypot(A, B))
    gex = np.array([B ** 2 / R ** 3, -A * B / R ** 3])          # analytic exact-GS gradient

    # cross-check the closed form against a central difference on the exact ground state
    pv = jnp.asarray(p)
    def _y_exact(q):
        return float(jnp.real(dq.expect(qep.SZ, qep.equilibrium(
            q, FX, 20.0, 0.0, 0.0, NS, "exact"))))
    gnum = np.array([(_y_exact(pv.at[i].add(EPS)) - _y_exact(pv.at[i].add(-EPS))) / (2 * EPS)
                     for i in (0, 1)])
    print(f"   exact grad: analytic {np.round(gex, 8)}  central-diff {np.round(gnum, 8)}"
          f"  max|diff| {np.max(np.abs(gex - gnum)):.2e}", flush=True)
    assert np.max(np.abs(gex - gnum)) < 1e-7

    Ts = np.linspace(TG_MIN, TG_MAX, NTG)
    ng = float(np.linalg.norm(gex))
    out = {"T": Ts, "p": np.array(p), "g_exact": gex, "g_exact_numeric": gnum,
           "eps": EPS, "fx": FX, "tol": TOL, "fit_Tmin": T_FIT_MIN,
           "start": np.array([qep.A0, qep.B0]), "sched_names": np.array(list(GSCHEDS))}
    for name, sch in GSCHEDS.items():
        t0 = time.time()
        G = np.array(jax.jit(jax.vmap(lambda T: grad_adiabatic(pv, T, sch)))(jnp.asarray(Ts)))
        dev = np.linalg.norm(G - gex[None], axis=-1) / ng
        dg = dgamma(p, sch)
        per = 2.0 * np.pi / dg
        env = envelope(dev, round(per / (Ts[1] - Ts[0])))
        sl = loglog_slope(Ts, env, T_FIT_MIN)
        out[f"grad_{name}"] = G
        out[f"dev_{name}"] = dev
        out[f"env_{name}"] = env
        out[f"dgamma_{name}"] = dg
        out[f"period_{name}"] = per
        # measure the period on the SIGNED component deviation: |dev| is rectified, so its own
        # period is half the phase period 2 pi / dgamma.
        out[f"period_meas_{name}"] = measured_period(Ts, G[:, 0] - gex[0], T_FIT_MIN)
        out[f"slope_{name}"] = sl
        print(f"   {name:6s}: rel dev  {dev[0]:.3e} (T={TG_MIN:g}) -> {dev[-1]:.3e}"
              f" (T={TG_MAX:g});  envelope on T>={T_FIT_MIN:g}: T^{sl:+.3f}"
              f" (mean {env[Ts >= T_FIT_MIN].mean():.4f});  period predicted {per:.3f}"
              f" measured {out[f'period_meas_{name}']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    save("bd_grad_vs_T", **out)


# --- exp_qep_grad_vs_T: the ACTUAL QEP gradient, not the output derivative ------------------
# exp_grad_vs_T above differentiates the OUTPUT y = <sigma^z> directly. That is not what QEP
# extracts. QEP reads the gradient off TWO equilibria via Onsager reciprocity (Eq. 6 of the note):
# with H_beta = H_sys + beta*O, O the MEASURED operator (here sigma^z),
#
#     g_theta = ( <d_theta H>_nudge - <d_theta H>_free ) / beta ,
#     d_A H = -sigma^z ,  d_B H = -sigma^x .
#
# Onsager says this equals d y / d theta as beta -> 0, so the exact-ground-state limit is the SAME
# closed form (B^2/R^3, -A B/R^3) -- but the estimator is completely different: two adiabatic
# evolutions, whose O(1) boundary biases largely cancel in the difference, and then a division by
# beta that re-amplifies whatever is left. Whether the O(1) gradient bias survives that is the
# question this experiment answers.
#
# Reference here is the EXACT-equilibrium QEP gradient at the SAME beta (the convention of
# run_bandwidth.py), so the deviation isolates the adiabatic error and not the O(beta) finite-nudge
# error; the analytic d y/d theta is saved alongside so the O(beta) offset is visible too.
# The nudge is ramped with the same schedule as (A,B) -- qep.equilibrium's shipped behaviour, and
# what the boundary analysis requires (ALL tau-dependence of H must have the vanishing derivative).
BETA = 0.02                                     # project-wide nudge strength


def qep_grad(pv, T, schedule, method="adiabatic", beta=BETA):
    """(<d_theta H>_nudge - <d_theta H>_free)/beta at unit error signal, d_theta H = (-sz, -sx)."""
    pf = qep.equilibrium(pv, FX, T, 0.0, 0.0, NS, method, schedule, tol=TOL)
    pn = qep.equilibrium(pv, FX, T, beta, 1.0, NS, method, schedule, tol=TOL)
    df = jnp.array([-jnp.real(dq.expect(qep.SZ, pf)), -jnp.real(dq.expect(qep.SX, pf))])
    dn = jnp.array([-jnp.real(dq.expect(qep.SZ, pn)), -jnp.real(dq.expect(qep.SX, pn))])
    return (dn - df) / beta


def exp_qep_grad_vs_T():
    """Same three panels as exp_grad_vs_T, but for the two-phase QEP gradient."""
    p = POINTS[0]
    A, B = float(p[0]), float(p[1])
    R = float(np.hypot(A, B))
    pv = jnp.asarray(p)
    g_analytic = np.array([B ** 2 / R ** 3, -A * B / R ** 3])       # beta -> 0 limit
    gex = np.array(qep_grad(pv, 20.0, "smooth", method="exact"))    # exact equilibria, same beta
    print(f"   analytic dy/d(A,B) = {np.round(g_analytic, 8)}", flush=True)
    print(f"   exact-QEP at beta={BETA} = {np.round(gex, 8)}"
          f"   (O(beta) offset {np.max(np.abs(gex - g_analytic)):.2e})", flush=True)

    Ts = np.linspace(TG_MIN, TG_MAX, NTG)
    ng = float(np.linalg.norm(gex))
    out = {"T": Ts, "p": np.array(p), "g_exact": gex, "g_analytic": g_analytic, "beta": BETA,
           "fx": FX, "tol": TOL, "fit_Tmin": T_FIT_MIN, "start": np.array([qep.A0, qep.B0])}
    for name, sch in GSCHEDS.items():
        t0 = time.time()
        G = np.array(jax.jit(jax.vmap(lambda T: qep_grad(pv, T, sch)))(jnp.asarray(Ts)))
        dev = np.linalg.norm(G - gex[None], axis=-1) / ng
        per = 2.0 * np.pi / dgamma(p, sch)
        env = envelope(dev, round(per / (Ts[1] - Ts[0])))
        sl = loglog_slope(Ts, env, T_FIT_MIN)
        out[f"grad_{name}"] = G; out[f"dev_{name}"] = dev; out[f"env_{name}"] = env
        out[f"period_{name}"] = per; out[f"slope_{name}"] = sl
        out[f"period_meas_{name}"] = measured_period(Ts, G[:, 0] - gex[0], T_FIT_MIN)
        print(f"   {name:6s}: rel dev  {dev[0]:.3e} (T={TG_MIN:g}) -> {dev[-1]:.3e}"
              f" (T={TG_MAX:g});  envelope on T>={T_FIT_MIN:g}: T^{sl:+.3f}"
              f" (mean {env[Ts >= T_FIT_MIN].mean():.4f});  period predicted {per:.3f}"
              f" measured {out[f'period_meas_{name}']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    save("bd_qep_grad_vs_T", **out)


# --- exp_grad_vs_T_lam: Fig. 1(b) of the note, with the parametrized_ascend family ----------
def exp_grad_vs_T_lam():
    """Fig. 1(b) of the note, repeated with `parametrized_ascend` at several bandwidths lam.

    Same model, same estimator and same tolerance as exp_grad_vs_T: (A,B)=(0.8,1.5), fx=0, the
    parameter gradient d y / d(A,B) of y=<sigma^z> by central difference, reference the analytic
    exact-ground-state value dy/dA = B^2/R^3, dy/dB = -A B/R^3.

    The whole family has f'(0)=f'(1)=0, so the O(1) boundary term of Eq. (33) is cancelled for
    every lam and every curve must converge -- unlike the stiff schedule, which is kept as a
    reference and does not. What lam controls is the RESIDUAL: with f''(0)=lam the leading
    survivor is ~ lam/T, so larger lam means a larger oscillation at the same T, crossing over to
    the stiff O(1) behaviour once lam >~ T.
    """
    p = POINTS[0]
    A, B = float(p[0]), float(p[1])
    R = float(np.hypot(A, B))
    gex = np.array([B ** 2 / R ** 3, -A * B / R ** 3])
    pv = jnp.asarray(p)
    Ts = np.linspace(TG_MIN, TG_MAX, NTG)
    ng = float(np.linalg.norm(gex))
    out = {"T": Ts, "p": np.array(p), "g_exact": gex, "lams": np.array(LAM_SET),
           "eps": EPS, "fx": FX, "tol": TOL, "fit_Tmin": T_FIT_MIN,
           "start": np.array([qep.A0, qep.B0]), "smooth_lam": float(np.pi ** 2 / 2)}
    for name, sch in grad_series():
        t0 = time.time()
        G = np.array(jax.jit(jax.vmap(lambda T: grad_adiabatic(pv, T, sch)))(jnp.asarray(Ts)))
        dev = np.linalg.norm(G - gex[None], axis=-1) / ng
        per = 2.0 * np.pi / dgamma(p, sch)
        env = envelope(dev, round(per / (Ts[1] - Ts[0])))
        out[f"grad_{name}"] = G
        out[f"dev_{name}"] = dev
        out[f"env_{name}"] = env
        out[f"period_{name}"] = per
        out[f"slope_{name}"] = loglog_slope(Ts, env, T_FIT_MIN)
        print(f"   {name:7s}: rel dev {dev[0]:.3e} (T={TG_MIN:g}) -> {dev[-1]:.3e} (T={TG_MAX:g});"
              f"  envelope on T>={T_FIT_MIN:g}: T^{out[f'slope_{name}']:+.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    save("bd_grad_vs_T_lam", **out)


# --- exp_qep_grad_vs_T_lam: the same figure, but with the REAL QEP gradient -----------------
def dHdp_of(state):
    """<d H / d(A,B)> = (-<sigma^z>, -<sigma^x>), both from ONE state."""
    return jnp.stack([-jnp.real(dq.expect(qep.SZ, state)),
                      -jnp.real(dq.expect(qep.SX, state))])


def qep_grad_output(pv, T, schedule, beta, method="adiabatic"):
    """QEP two-phase estimate of d y / d(A,B) with a BARE nudge (error signal = 1).

    Onsager reciprocity (Eq. 6 of the note) says d<O_i>/dtheta = d<d_theta H>/d lambda_i at
    lambda_i = 0, so nudging along the MEASURED operator sigma^z with unit error signal and
    differencing <d_theta H> gives exactly the same dy/dtheta that grad_adiabatic() obtains by
    central difference -- but through the QEP estimator, so it also carries QEP's own O(beta)
    linearisation bias. Note the whole 2-vector comes from ONE pair of evolutions (2 solves),
    against 4 for the central difference.
    """
    pf = qep.equilibrium(pv, FX, T, 0.0, 0.0, NS, method, schedule, tol=TOL)
    pn = qep.equilibrium(pv, FX, T, beta, 1.0, NS, method, schedule, tol=TOL)
    return (dHdp_of(pn) - dHdp_of(pf)) / beta


def exp_qep_grad_vs_T_lam():
    """Figure 5's twin, with the real QEP gradient instead of a finite difference in (A,B).

    Same point, schedule family, T grid and tolerance. The new feature is a SECOND error source:
    QEP's one-sided nudge is O(beta) and T-INDEPENDENT, so each curve descends only until it
    reaches its own beta floor and then stops -- unlike the central-difference gradient, which
    keeps decaying. The floor is measured directly, with EXACT equilibria at the same beta.
    """
    p = POINTS[0]
    A, B = float(p[0]), float(p[1])
    R = float(np.hypot(A, B))
    gex = np.array([B ** 2 / R ** 3, -A * B / R ** 3])
    pv = jnp.asarray(p)
    Ts = np.linspace(TG_MIN, TG_MAX, NTG)
    ng = float(np.linalg.norm(gex))
    out = {"T": Ts, "p": np.array(p), "g_exact": gex, "lams": np.array(LAM_SET),
           "beta": QBETA, "fx": FX, "tol": TOL,
           "fit_Tmin": T_FIT_MIN, "start": np.array([qep.A0, qep.B0]),
           "smooth_lam": float(np.pi ** 2 / 2)}

    def series(tag, sch, beta):
        t0 = time.time()
        G = np.array(jax.jit(jax.vmap(
            lambda T: qep_grad_output(pv, T, sch, beta)))(jnp.asarray(Ts)))
        dev = np.linalg.norm(G - gex[None], axis=-1) / ng
        env = envelope(dev, round(2.0 * np.pi / dgamma(p, sch) / (Ts[1] - Ts[0])))
        # the beta floor: same beta, EXACT equilibria -> no adiabatic error at all
        gfl = np.array(qep_grad_output(pv, 20.0, sch, beta, method="exact"))
        floor = float(np.linalg.norm(gfl - gex) / ng)
        out[f"grad_{tag}"] = G; out[f"dev_{tag}"] = dev; out[f"env_{tag}"] = env
        out[f"floor_{tag}"] = floor
        out[f"slope_{tag}"] = loglog_slope(Ts, env, T_FIT_MIN)
        print(f"   {tag:12s}: dev {dev[0]:.3e} -> {dev[-1]:.3e};  beta floor {floor:.3e};"
              f"  envelope T^{out[f'slope_{tag}']:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

    for tag, sch in grad_series():
        series(tag, sch, QBETA)
    save("bd_qep_grad_vs_T_lam", **out)


# --- exp_grad_scatter_sched: repeat grad_scatter / grad_cos_scatter per schedule ------------
# The detection study produced two scatter figures that appear to disagree AT THE SAME T=20,
# the same 40 points and the same beta:
#   grad_scatter      QEP-adiabatic (y) vs NUMERICAL-adiabatic (x)  -> a mess, values to +-20
#   grad_cos_scatter  QEP-adiabatic (y) vs EXACT-QEP        (x)  -> tight, median cos 0.989
# Two things differ, and this experiment separates them:
#   (i)  THE REFERENCE. grad_scatter's x-axis is the finite-difference gradient of the ADIABATIC
#        cost, which carries its own O(1) boundary error -- it is not ground truth. Worse, both
#        axes are summed over K=9 probes, so nine O(1) errors add. grad_cos_scatter's x-axis is
#        the exact-equilibrium QEP gradient: T-independent and clean.
#   (ii) THE AXIS LIMITS. grad_cos_scatter clips to [-2,2] (it says so in its title), so the
#        outliers that dominate grad_scatter's panel are simply not drawn.
# Faithful repeat: same PRNGKey(0), same 40 points, same probe set and target as
# run_experiments.py, so the numbers are comparable to the published figures. The schedule
# argument ramps BOTH (A,B) AND the nudge (qep.equilibrium has no separate control), so
# "stiff" here means a stiff switch-on of the nudging phase too.
SCAT_H = (0.8, 1.5)
SCAT_FXS = jnp.arange(-1.5, 1.51, 0.5)          # K = 7 probes on [-1.5, 1.5]
SCAT_TS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
SCAT_NUM_T = 20.0                               # T at which the numerical reference is taken
SCAT_SCHEDS = ("linear", "smooth")
WEDGE_SLOPE = 0.5          # the wedge |B| < (1-A)*WEDGE_SLOPE, marked red (not excluded)


def qep_grad_probes(pv, T, schedule, method="adiabatic", beta=BETA, fxs=None):
    """QEP gradient summed over the probe set, at UNIT error signal -- NO TARGET.

    For each probe f_x: free phase = equilibrium of H_sys - f_x sx, nudge phase = the same plus
    beta*sigma^z (the MEASURED operator), with eps == 1 rather than eps = y - y_tau. So

        g = mean_i ( <d_theta H>_{beta,i} - <d_theta H>_{0,i} ) / beta ,

    AVERAGED over the probe set (not summed), so it is the per-probe gradient and does not
    scale with K. By Onsager it is mean_i d y_i / d theta.
    beta is a constant 0.02 for every probe and every point; nothing here depends on a target.
    """
    def one(fx):
        pf = qep.equilibrium(pv, fx, T, 0.0, 0.0, NS, method, schedule)
        pn = qep.equilibrium(pv, fx, T, beta, 1.0, NS, method, schedule)
        df = jnp.array([-jnp.real(dq.expect(qep.SZ, pf)), -jnp.real(dq.expect(qep.SX, pf))])
        dn = jnp.array([-jnp.real(dq.expect(qep.SZ, pn)), -jnp.real(dq.expect(qep.SX, pn))])
        return (dn - df) / beta
    return jnp.mean(jax.vmap(one)(SCAT_FXS if fxs is None else fxs), axis=0)


def num_grad_probes(pv, T, schedule, method="adiabatic", eps=1e-4):
    """Central-difference reference for the same object: d/d(A,B) of mean_i y_i. No target."""
    def cost(q):
        return jnp.mean(jax.vmap(lambda fx: qep.measure(q, fx, T, NS, method, schedule))(SCAT_FXS))
    return jnp.array([(cost(pv.at[i].add(eps)) - cost(pv.at[i].add(-eps))) / (2 * eps)
                      for i in (0, 1)])


def exp_grad_scatter_sched():
    """adiabatic-QEP vs exact-QEP at 40 random points, per schedule, vs T. NO TARGET, eps == 1."""
    # (A,B) uniform on [-3,3]^2 -- ALL 40 points are kept. The wedge is no longer a rejection
    # rule, it is a LABEL: in_wedge flags the points inside {A < 1 and |B| < (1-A)*WEDGE_SLOPE},
    # and the figures draw those red.
    #
    # What the wedge means. The ramp is (A(tau),B(tau)) = (1,0) + f(tau)*(A-1, B) -- a straight
    # SEGMENT in the (A,B) plane from the start to the endpoint. The gap 2|(A(tau),B(tau))|
    # vanishes only at the origin, so the risky endpoints are those whose segment passes close to
    # (0,0). The perpendicular distance from the origin to it is |B|/sqrt((A-1)^2+B^2), so the rays
    # B = +- WEDGE_SLOPE*(A-1) out of (1,0) bound a cone about the -A direction; inside it the path
    # aims at the origin. At slope 1/2 the cone guarantees a closest approach below 1/sqrt(5)=0.447.
    #
    # CAVEAT worth keeping in view: this is the UNPROBED path. With a probe on,
    # H = -A sz - (B+fx) sx, so the segment runs from (1,fx) to (A,B+fx) -- shifted vertically by
    # fx -- and the wedge does not describe those. Measured: excluding the slope-1/3 wedge left the
    # A<0 deviation at 16.4 (vs 17.2 uncut) because the worst probe still reached within 0.036 of
    # the origin. The red marking therefore flags a NECESSARY-ish but not sufficient condition.
    cand = jax.random.uniform(jax.random.PRNGKey(0), (40, 2), minval=-3.0, maxval=3.0)
    pts = cand
    in_wedge = np.array((cand[:, 0] < 1.0)
                        & (jnp.abs(cand[:, 1]) < (1.0 - cand[:, 0]) * WEDGE_SLOPE))
    print("   %d of %d points fall inside the wedge (slope %.4g)"
          % (in_wedge.sum(), len(cand), WEDGE_SLOPE), flush=True)
    out = {"pts": np.array(pts), "beta": BETA, "fxs": np.array(SCAT_FXS),
           "T": np.array(SCAT_TS), "num_T": SCAT_NUM_T, "eps_signal": 1.0,
           "in_wedge": in_wedge, "wedge_slope": WEDGE_SLOPE}

    gqe = np.array(jax.vmap(lambda q: qep_grad_probes(q, 20.0, "smooth", method="exact"))(pts))
    gne = np.array(jax.vmap(lambda q: num_grad_probes(q, 20.0, "smooth", method="exact"))(pts))
    out["g_qep_exact"] = gqe
    out["g_num_exact"] = gne
    print(f"   exact equilibria: max|QEP - numerical| = {np.max(np.abs(gqe - gne)):.2e}"
          f"  (validates the QEP estimator itself; the O(beta) offset is the floor here)",
          flush=True)

    def cosines(a_, b_):
        return (np.sum(a_ * b_, -1)
                / (np.linalg.norm(a_, axis=-1) * np.linalg.norm(b_, axis=-1) + 1e-15))

    for sc in SCAT_SCHEDS:
        for T in SCAT_TS:
            t0 = time.time()
            gq = np.array(jax.vmap(lambda q: qep_grad_probes(q, float(T), sc))(pts))
            c = cosines(gqe, gq)
            rel = np.linalg.norm(gq - gqe, axis=-1) / (np.linalg.norm(gqe, axis=-1) + 1e-15)
            out[f"gq_{sc}_T{int(T)}"] = gq
            out[f"cos_{sc}_T{int(T)}"] = c
            out[f"med_{sc}_T{int(T)}"] = float(np.median(c))
            out[f"rel_{sc}_T{int(T)}"] = rel
            pos = np.array(pts)[:, 0] > 0
            print(f"   {sc:6s} T={T:5.1f}: median cos {np.median(c[pos]):+.4f}"
                  f"   median rel dev {np.median(rel[pos]):.4f}   (A>0 subset)"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
        gn = np.array(jax.vmap(lambda q: num_grad_probes(q, SCAT_NUM_T, sc))(pts))
        out[f"gn_{sc}_T{int(SCAT_NUM_T)}"] = gn
    save("bd_grad_scatter_sched", **out)


# --- exp_grad_scatter_probes: the same scatter with FEWER probing fields --------------------
# Figure 11 uses K=7 probes on [-1.5,1.5]. This repeats it for K=3 (f_x = -1, 0, 1) and K=1
# (f_x = 0), at the same 40 points, to see what the probe set is worth. Recall the gradient is the
# AVERAGE over probes, so shrinking K removes the partial cancellation of per-probe boundary
# errors -- outside the wedge that cancellation was measured to be worth up to ~5x.
# Note K=1 at f_x=0 is degenerate in a second way: y = A/sqrt(A^2+B^2) depends only on the polar
# angle of (A,B), so grad y is purely angular and carries NO radial information at all.
PROBE_SETS = {"K3": jnp.array([-1.0, 0.0, 1.0]), "K1": jnp.array([0.0])}


def exp_grad_scatter_probes():
    """Figure 11', the K=3 and K=1 counterparts of exp_grad_scatter_sched."""
    cand = jax.random.uniform(jax.random.PRNGKey(0), (40, 2), minval=-3.0, maxval=3.0)
    pts = cand
    in_wedge = np.array((cand[:, 0] < 1.0)
                        & (jnp.abs(cand[:, 1]) < (1.0 - cand[:, 0]) * WEDGE_SLOPE))
    out = {"pts": np.array(pts), "beta": BETA, "T": np.array(SCAT_TS),
           "in_wedge": in_wedge, "wedge_slope": WEDGE_SLOPE,
           "set_names": np.array(list(PROBE_SETS))}

    def cosines(a_, b_):
        return (np.sum(a_ * b_, -1)
                / (np.linalg.norm(a_, axis=-1) * np.linalg.norm(b_, axis=-1) + 1e-15))

    for tag, fxs in PROBE_SETS.items():
        out[f"fxs_{tag}"] = np.array(fxs)
        gqe = np.array(jax.vmap(lambda q: qep_grad_probes(q, 20.0, "smooth", method="exact",
                                                          fxs=fxs))(pts))
        out[f"g_qep_exact_{tag}"] = gqe
        for sc in SCAT_SCHEDS:
            for T in SCAT_TS:
                t0 = time.time()
                gq = np.array(jax.vmap(lambda q: qep_grad_probes(q, float(T), sc, fxs=fxs))(pts))
                c = cosines(gqe, gq)
                rel = (np.linalg.norm(gq - gqe, axis=-1)
                       / (np.linalg.norm(gqe, axis=-1) + 1e-15))
                out[f"gq_{tag}_{sc}_T{int(T)}"] = gq
                out[f"cos_{tag}_{sc}_T{int(T)}"] = c
                out[f"rel_{tag}_{sc}_T{int(T)}"] = rel
                print(f"   {tag} {sc:6s} T={T:5.1f}: median cos {np.median(c[~in_wedge]):+.4f}"
                      f"   median rel dev {np.median(rel[~in_wedge]):.4f}   (outside the wedge)"
                      f"   ({time.time()-t0:.0f}s)", flush=True)
    save("bd_grad_scatter_probes", **out)


ALL = {"schedule_shapes": exp_schedule_shapes,
       "prep_fid_vs_T": exp_prep_fid_vs_T,
       "grad_vs_T": exp_grad_vs_T,
       "qep_grad_vs_T": exp_qep_grad_vs_T,
       "grad_scatter_sched": exp_grad_scatter_sched,
       "grad_scatter_probes": exp_grad_scatter_probes,
       "grad_vs_T_lam": exp_grad_vs_T_lam,
       "qep_grad_vs_T_lam": exp_qep_grad_vs_T_lam}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    for n in (list(ALL) if args == ["all"] else args):
        print(f"[{n}]", flush=True)
        t0 = time.time()
        ALL[n]()
        print(f"   done in {time.time()-t0:.0f}s", flush=True)
