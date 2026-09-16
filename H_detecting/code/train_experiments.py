"""Training experiments for the H-detecting QEP study (intro.md section 'Training').

The update uses the AVERAGE QEP gradient over the probe dataset (reduce='mean'); the cost
reported is the AVERAGE quadratic cost  C = mean_i 0.5 (y_i - y_tau_i)^2.

Experiments
  grad_directions   per-probe gradient direction for each (f_x, y_tau) pair vs the average
                    direction, at a point; plus single-probe vs average descent vector fields.
  prep_fid_vs_T     adiabatic ground-state-preparation fidelity vs evolution time T.
  training_mean     average-gradient training curves (cost, |p-h|) vs epoch, exact + adiabatic.
  perf_vs_T         final AVERAGE cost vs T over many initial parameters (matches the
                    single_qubit perf_vs_T_linear style: individual runs, mean +/- std, best).

Outputs -> ../data/train_*.npz  (consumed by draw_train_figs.py).
Run:  python train_experiments.py [test|full]
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import qep

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)

H = (0.8, 1.5)                          # target field (h_z, h_x)
FXS = jnp.linspace(-2.0, 2.0, 9)        # probe set
K = FXS.shape[0]
NS = 2
BETA = 0.02
LR = 2.0                                # tuned for the mean (averaged) gradient
Y_TAU = qep.make_dataset(jnp.asarray(H), FXS)


def avg_cost(p, fxs, y_taus, T, method):
    """Average quadratic cost C = mean_i 0.5 (y_i - y_tau_i)^2 for one params point."""
    return float(qep.cost(jnp.asarray(p), fxs, y_taus, T, NS, method)) / fxs.shape[0]


def avg_cost_traj(P, fxs, y_taus, T, method):
    """Average cost for a (M,2) array of params, vmapped (compiles once -> fast).

    Equivalent to [avg_cost(p, ...) for p in P] but avoids re-tracing the solver per point.
    """
    K = fxs.shape[0]
    return np.array(jax.vmap(lambda p: qep.cost(p, fxs, y_taus, T, NS, method))(jnp.asarray(P))) / K


def save(name, **kw):
    np.savez(os.path.join(DATA, name + ".npz"), **kw)
    print("   saved data/%s.npz" % name, flush=True)


# ----------------------------------------------------------------- experiments
def exp_grad_directions():
    # (1) per-probe gradient directions at a point vs the averaged direction
    p0 = jnp.array([1.0, 1.0])
    ys, gs = qep.qep_grad_persample(p0, FXS, Y_TAU, 20.0, BETA, NS, "exact")   # gs: (K,2)
    g_avg = np.array(jnp.mean(gs, axis=0))
    # (2) descent vector fields over the (A,B) plane: single probe (fx=0) vs averaged dataset
    ga = np.linspace(0.2, 1.8, 13); gb = np.linspace(0.2, 2.3, 13)
    AA, BB = np.meshgrid(ga, gb)
    P = jnp.array(np.stack([AA.ravel(), BB.ravel()], 1))
    fx1 = jnp.array([0.0]); yt1 = qep.make_dataset(jnp.asarray(H), fx1)
    Gsingle = np.array(jax.vmap(lambda p: qep.qep_grad_batch(p, fx1, yt1, 20.0, BETA, NS, "exact", reduce="mean")[1])(P))
    Gavg = np.array(jax.vmap(lambda p: qep.qep_grad_batch(p, FXS, Y_TAU, 20.0, BETA, NS, "exact", reduce="mean")[1])(P))
    traj = np.array(qep.train_scan((1.0, 1.0), FXS, Y_TAU, 20.0, BETA, LR, 200, NS, "exact", reduce="mean"))
    save("train_grad_directions", p0=np.array(p0), fxs=np.array(FXS), y_tau=np.array(Y_TAU),
         grads=np.array(gs), grad_avg=g_avg, h=np.array(H),
         AA=AA, BB=BB, Gsingle=Gsingle.reshape(AA.shape + (2,)),
         Gavg=Gavg.reshape(AA.shape + (2,)), traj=traj)


def exp_prep_fid_vs_T():
    import dynamiqs as dq
    configs = jnp.array([[0.8, 1.5], [1.0, 0.0], [0.3, 1.0]])     # params
    fxs = jnp.array([-1.0, 0.0, 1.0])
    Ts = np.array([2., 3., 4., 6., 9., 14., 20., 32., 50., 80., 130.])
    def fid(p, fx, T):
        psiT = qep.equilibrium(p, fx, float(T), 0.0, 0.0, NS, "adiabatic")
        gs = qep.gs_of(-p[0] * qep.SZ - (p[1] + fx) * qep.SX)
        return jnp.abs(dq.overlap(psiT, gs)) ** 2
    # F[config, fx, T]
    F = np.array([[[float(fid(p, fx, T)) for T in Ts] for fx in fxs] for p in configs])
    save("train_prep_fid_vs_T", params=np.array(configs), fxs=np.array(fxs), T=Ts, fidelity=F)


def exp_training_mean(N=150):
    P0 = (1.0, 1.0)
    out = {"epoch": np.arange(N + 1), "h": np.array(H), "init": np.array(P0)}
    Pe = np.array(qep.train_scan(P0, FXS, Y_TAU, 20.0, BETA, LR, N, NS, "exact", reduce="mean"))
    out["cost_exact"] = avg_cost_traj(Pe, FXS, Y_TAU, 20.0, "exact")
    out["perr_exact"] = qep.param_error(Pe, H)
    for T in (10.0, 20.0, 40.0):
        Pa = np.array(qep.train_scan(P0, FXS, Y_TAU, T, BETA, LR, N, NS, "adiabatic", reduce="mean"))
        out["cost_T%d" % int(T)] = avg_cost_traj(Pa, FXS, Y_TAU, T, "adiabatic")
        out["perr_T%d" % int(T)] = qep.param_error(Pa, H)
    save("train_training_mean", **out)


def exp_perf_vs_T(N=150, n_init=16):
    # train from MANY initial parameters at several T; y-axis = final AVERAGE cost.
    # Inits are sampled in the physical basin A>0 (the target has h_z=0.8>0); the A<0 sign
    # sector is a separate global-optimisation issue that converges too slowly to probe the
    # adiabatic-time dependence, and would only inflate the mean with optimisation (not T) error.
    key = jax.random.PRNGKey(1)
    A = jax.random.uniform(key, (n_init,), minval=0.3, maxval=1.6)
    B = jax.random.uniform(jax.random.fold_in(key, 1), (n_init,), minval=0.0, maxval=2.4)
    inits = jnp.stack([A, B], axis=1)
    Ts = [5.0, 8.0, 12.0, 20.0, 32.0, 50.0, 80.0]
    Cfin = np.zeros((len(Ts), n_init))
    for i, T in enumerate(Ts):
        def one(p0):
            return qep.train_scan(p0, FXS, Y_TAU, T, BETA, LR, N, NS, "adiabatic", reduce="mean")[-1]
        pend = np.array(jax.vmap(one)(inits))
        Cfin[i] = avg_cost_traj(pend, FXS, Y_TAU, T, "adiabatic")
        print("   T=%.0f  mean avg-cost=%.3e" % (T, Cfin[i].mean()), flush=True)
    save("train_perf_vs_T", T=np.array(Ts), cost=Cfin, inits=np.array(inits), h=np.array(H))


def exp_multi_init_paths(N=150):
    # parameter trajectories from several initial parameters, for SHORT / MEDIUM / LONG evolution
    # time (T=5,20,80) plus the exact (T-independent) reference -> shows how the convergence point
    # and path sharpen as T grows.
    inits = jnp.array([[1.0, 1.0], [0.3, 0.4], [1.7, 0.2], [0.2, 2.1], [1.6, 2.0], [1.0, 0.2]])
    Ts = [5.0, 20.0, 80.0]
    out = {"inits": np.array(inits), "h": np.array(H), "T": np.array(Ts)}
    out["paths_exact"] = np.array(jax.vmap(
        lambda p0: qep.train_scan(p0, FXS, Y_TAU, 20.0, BETA, LR, N, NS, "exact", reduce="mean"))(inits))
    for T in Ts:
        Pa = np.array(jax.vmap(
            lambda p0: qep.train_scan(p0, FXS, Y_TAU, T, BETA, LR, N, NS, "adiabatic", reduce="mean"))(inits))
        out["paths_T%d" % int(T)] = Pa
        # final-parameter error per init, to annotate convergence quality
        out["perr_T%d" % int(T)] = qep.param_error(Pa[:, -1], H)
    save("train_multi_init_paths", **out)


def exp_response_evolution(N=150):
    # response curve y(f_x) of H_0 BEFORE / DURING / AFTER training, vs the target response.
    # We save BOTH the exact ground-state response (the true response of the learned H_0) and the
    # adiabatically-prepared response at T=20 (what training actually uses). At the off init the
    # adiabatic "before" curve is non-smooth: for probe fields where the ramp crosses a small gap
    # the prep is non-adiabatic (Landau-Zener), so it deviates from the exact GS -- this disappears
    # as the parameters move to the large-gap region during training.
    p0 = (0.2, 2.1)                                   # a deliberately off init (small A -> small gap)
    P = np.array(qep.train_scan(p0, FXS, Y_TAU, 20.0, BETA, LR, N, NS, "adiabatic", reduce="mean"))
    snap_ep = [0, 5, 20, N]                           # before, two during, after
    snaps = np.array([P[e] for e in snap_ep])
    fg = jnp.linspace(-3.0, 3.0, 200)
    curves_adiab = np.stack([np.array(qep.response_curve(jnp.asarray(s), fg, 20.0, NS, "adiabatic")) for s in snaps])
    curves_exact = np.stack([np.array(qep.response_curve(jnp.asarray(s), fg, 20.0, NS, "exact")) for s in snaps])
    target = np.array(qep.y_analytic(jnp.asarray(H), fg))
    costs_adiab = np.array([avg_cost(s, FXS, Y_TAU, 20.0, "adiabatic") for s in snaps])
    costs_exact = np.array([avg_cost(s, FXS, Y_TAU, 20.0, "exact") for s in snaps])
    save("train_response_evolution", fg=np.array(fg), target=target,
         curves=curves_adiab, curves_adiab=curves_adiab, curves_exact=curves_exact,
         snap_epoch=np.array(snap_ep), snap_params=snaps,
         costs=costs_adiab, costs_adiab=costs_adiab, costs_exact=costs_exact,
         fxs=np.array(FXS), y_tau=np.array(Y_TAU), h=np.array(H), init=np.array(p0))


def exp_multi_init_paths_TS(N=150):
    """train_multi_init_paths repeated over a T x schedule grid, with the ramped nudge.

    Same training rule as the rest of the training study -- mean QEP gradient, LR=2.0, N=150,
    beta=0.02, 9 probes -- and the same 6 initial parameters. What varies here is the pair
    (T, schedule) over T = 1, 2, 5, 10, 20 and schedule in {smooth, stiff}.

    The nudge is switched on smoothly in BOTH columns: `nudge_ramp=True` (the shipped default)
    applies `ramp(t/T, schedule)` to the nudge term as well as to the (A,B) ramp, so "stiff" here
    means the SCHEDULE f(x)=x is stiff -- the nudge still follows that same f. The comparison is
    therefore purely about the schedule shape, not about whether the nudge is ramped.

    T=1 and T=2 are far outside the adiabatic regime (the gap is O(1), so T*gap ~ 1): the free
    phase does not prepare a ground state at all, the QEP gradient is not an approximation to
    anything, and these panels are included to show what training does when that assumption fails.
    """
    inits = jnp.array([[1.0, 1.0], [0.3, 0.4], [1.7, 0.2], [0.2, 2.1], [1.6, 2.0], [1.0, 0.2]])
    Ts = [1.0, 2.0, 5.0, 10.0, 20.0]
    scheds = ("smooth", "linear")
    out = {"inits": np.array(inits), "h": np.array(H), "T": np.array(Ts),
           "scheds": np.array(scheds), "N": N, "lr": LR, "beta": BETA, "nudge_ramp": True}
    # exact (T-independent) reference, for the faint guide path in every panel
    out["paths_exact"] = np.array(jax.vmap(lambda p0: qep.train_scan(
        p0, FXS, Y_TAU, 20.0, BETA, LR, N, NS, "exact", reduce="mean"))(inits))
    for s in scheds:
        for T in Ts:
            P = np.array(jax.vmap(lambda p0: qep.train_scan(
                p0, FXS, Y_TAU, T, BETA, LR, N, NS, "adiabatic", schedule=s,
                reduce="mean"))(inits))
            key = "%s_T%g" % (s, T)
            out["paths_" + key] = P
            err = qep.param_error(P[:, -1], H)
            out["perr_" + key] = err
            out["nfin_" + key] = int(np.sum(~np.isfinite(P[:, -1]).all(-1)))
            print(f"   {s:6s} T={T:5.1f}: median |p-h| = {np.median(err):.3e}"
                  f"   worst {np.max(err):.3e}   non-finite endpoints: {out['nfin_'+key]}",
                  flush=True)
    save("train_multi_init_paths_TS", **out)


# probe sets for exp_probe_sets_TS: (tag, label, fxs). K9 is the study-wide baseline.
PROBE_SETS = (
    ("K9", r"$K=9$: linspace$(-2,2,9)$",      np.linspace(-2.0, 2.0, 9)),
    ("K7", r"$K=7$: $\pm1.5$, step $0.5$",    np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])),
    ("K5", r"$K=5$: $\pm2$, step $1$",        np.array([-2.0, -1.0, 0.0, 1.0, 2.0])),
    ("K3", r"$K=3$: $-1,0,1$",                np.array([-1.0, 0.0, 1.0])),
)


def exp_probe_sets_TS(N=150):
    """How few probe fields can train? exp_multi_init_paths_TS repeated over four probe sets.

    Everything else is held at the training study's values: mean QEP gradient (so the effective
    step size does NOT depend on K), LR=2.0, N=150, beta=0.02, the same 6 inits, ramped nudge,
    T = 1, 2, 5, 10, 20, both schedules. y_tau is of course recomputed per probe set.

    Why the RANGE matters as much as the count: the response curve y(f_x)=A/sqrt(A^2+(B+f_x)^2)
    peaks at f_x = -B and has width ~A. At the target (A,B)=(0.8,1.5) that peak sits at f_x=-1.5,
    so K7 samples it exactly, K5 only brackets it (-2 and -1, spacing 1.0 against a width of 0.8),
    and K3 never reaches it -- K3 sees only one flank. Fewer points is therefore two changes at
    once, and the K5-vs-K7 pair separates them: K5 has fewer points but the wider span.
    """
    inits = jnp.array([[1.0, 1.0], [0.3, 0.4], [1.7, 0.2], [0.2, 2.1], [1.6, 2.0], [1.0, 0.2]])
    Ts = [1.0, 2.0, 5.0, 10.0, 20.0]
    scheds = ("smooth", "linear")
    out = {"inits": np.array(inits), "h": np.array(H), "T": np.array(Ts),
           "scheds": np.array(scheds), "N": N, "lr": LR, "beta": BETA, "nudge_ramp": True,
           "set_tags": np.array([t for t, _, _ in PROBE_SETS]),
           "set_labels": np.array([l for _, l, _ in PROBE_SETS])}
    for tag, _, fxs in PROBE_SETS:
        out["fxs_" + tag] = np.asarray(fxs)
        out["K_" + tag] = int(np.size(fxs))
    for tag, _, fxs in PROBE_SETS:
        fx = jnp.asarray(fxs)
        y = qep.make_dataset(jnp.asarray(H), fx)
        out["y_tau_" + tag] = np.asarray(y)
        out[f"paths_{tag}_exact"] = np.array(jax.vmap(lambda p0: qep.train_scan(
            p0, fx, y, 20.0, BETA, LR, N, NS, "exact", reduce="mean"))(inits))
        out[f"perr_{tag}_exact"] = qep.param_error(out[f"paths_{tag}_exact"][:, -1], H)
        for s in scheds:
            for T in Ts:
                P = np.array(jax.vmap(lambda p0: qep.train_scan(
                    p0, fx, y, T, BETA, LR, N, NS, "adiabatic", schedule=s,
                    reduce="mean"))(inits))
                key = f"{tag}_{s}_T{T:g}"
                out["paths_" + key] = P
                out["perr_" + key] = qep.param_error(P[:, -1], H)
        print(f"   {tag} (K={out['K_'+tag]}): exact {np.median(out[f'perr_{tag}_exact']):.2e}"
              + "".join(f" | smooth T={T:g}:{np.median(out[f'perr_{tag}_smooth_T{T:g}']):.2e}"
                        for T in Ts), flush=True)
    save("train_probe_sets_TS", **out)


def exp_probe_sets_costs():
    """Average cost along the training paths of exp_probe_sets_TS (post-processing, no re-training).

    Reads data/train_probe_sets_TS.npz and evaluates, at every epoch of every stored trajectory,
    the average cost on THAT SET'S OWN probes -- the objective the run is actually minimising.
    Uses EXACT equilibria: the question is the quality of the parameters at each epoch, not the
    adiabatic error, and it makes this cheap (a 2x2 eigh per probe, no ODE solve).

    Note the costs of different sets are different functions (C_K3 averages 3 probes, C_K9 nine),
    so read progress per set; |p-h| (exp_probe_sets_TS) is the probe-set-independent measure.
    """
    d = np.load(os.path.join(DATA, "train_probe_sets_TS.npz"))
    tags = [str(t) for t in d["set_tags"]]
    Ts = list(d["T"])
    scheds = [str(s) for s in d["scheds"]]
    out = {"T": np.array(Ts), "set_tags": np.array(tags), "scheds": np.array(scheds),
           "set_labels": d["set_labels"], "h": d["h"], "N": d["N"]}
    for tag in tags:
        fx = jnp.asarray(d["fxs_" + tag]); y = jnp.asarray(d["y_tau_" + tag])
        for s in scheds:
            for T in Ts:
                P = d[f"paths_{tag}_{s}_T{T:g}"]                 # (ninit, N+1, 2)
                out[f"cost_{tag}_{s}_T{T:g}"] = avg_cost_traj(
                    jnp.asarray(P.reshape(-1, 2)), fx, y, 20.0, "exact").reshape(P.shape[:2])
        print(f"   {tag}: final median cost (smooth) "
              + " ".join("T=%g:%.1e" % (T, np.median(out[f"cost_{tag}_smooth_T{T:g}"][:, -1]))
                         for T in Ts), flush=True)
    save("train_probe_sets_costs", **out)


ALL = [exp_grad_directions, exp_prep_fid_vs_T, exp_training_mean, exp_perf_vs_T,
       exp_multi_init_paths, exp_multi_init_paths_TS, exp_probe_sets_TS,
       exp_probe_sets_costs, exp_response_evolution]


def run_test():
    p0 = (1.0, 1.0)
    Pe = np.array(qep.train_scan(p0, FXS, Y_TAU, 20.0, BETA, LR, 200, NS, "exact", reduce="mean"))
    print("exact mean-grad train end", np.round(Pe[-1], 4), " target", H,
          " avg-cost=%.2e" % avg_cost(Pe[-1], FXS, Y_TAU, 20.0, "exact"), flush=True)
    Pa = np.array(qep.train_scan(p0, FXS, Y_TAU, 40.0, BETA, LR, 200, NS, "adiabatic", reduce="mean"))
    print("adiab T=40 train end ", np.round(Pa[-1], 4),
          " avg-cost=%.2e" % avg_cost(Pa[-1], FXS, Y_TAU, 40.0, "adiabatic"), flush=True)
    print("TEST OK", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "test":
        run_test()
    else:
        # run named experiments (e.g. "multi_init_paths response_evolution") or all if none given
        names = args if args and args[0] != "full" else [fn.__name__.replace("exp_", "") for fn in ALL]
        by_name = {fn.__name__.replace("exp_", ""): fn for fn in ALL}
        for nm in names:
            fn = by_name[nm]
            t0 = time.time(); print("[%s]" % fn.__name__, flush=True); fn()
            print("   %.0fs" % (time.time() - t0), flush=True)
        print("TRAINING EXPERIMENTS DONE:", ", ".join(names), flush=True)
