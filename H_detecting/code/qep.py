"""H-detecting Quantum Equilibrium Propagation (QEP) -- core model, gradient, training.

Companion to ../single_qubit (which used the LINEAR cost C=<H_tau> and therefore recovered
only the RAY (A,B) ~ (h_z,h_x), never the amplitude). Here a PROBING FIELD breaks the
amplitude degeneracy.

Model / target:
    H_sys(A,B) = -A sigma^z - B sigma^x ,   H_tau = -h_z sigma^z - h_x sigma^x .

Probe: add a field f_x along x and measure the z-magnetization of the ground state of
    H(f_x) = H_sys - f_x sigma^x = -A sigma^z - (B+f_x) sigma^x ,
    y(f_x) = <sigma^z> = A / sqrt(A^2 + (B+f_x)^2)        (exact, single qubit).
The SHAPE of this response curve (peak at f_x=-B, width ~A) fixes the absolute (A,B), not just
their ratio. We fit the curve with the QUADRATIC cost over a set of probes
    C = 1/2 sum_i (y(f_{x,i}) - y^tau_i)^2 ,   error signal eps_i = y_i - y^tau_i = dC/dy_i ,
and nudge along the MEASURED operator sigma^z:
    H_beta = H_sys - f_x sigma^x + beta eps sigma^z .

QEP gradient (same two-phase formula as single_qubit; d_A H = -sigma^z, d_B H = -sigma^x):
    g = (<d_theta H>_nudge - <d_theta H>_free) / beta ,   summed over the probe dataset.

Schedule fixes carried over from single_qubit: SMOOTH ramp (zero time-derivative at the
boundaries) + the nudge ramped on smoothly => the O(1) adiabatic boundary bias cancels; tight
solver tolerance because the QEP gradient divides an O(beta) signal by beta.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import dynamiqs as dq
import optax

jax.config.update("jax_enable_x64", True)

SZ, SX = dq.sigmaz(), dq.sigmax()
A0, B0 = 1.0, 0.0                       # fixed adiabatic-schedule starting point


def ramp(x, schedule="smooth"):
    """Schedule lambda(tau) on [0,1], lambda(0)=0, lambda(1)=1.

    'linear': lambda=tau (stiff -> O(1) adiabatic boundary bias).
    'smooth': lambda=(1-cos(pi tau))/2 (zero derivative at tau=0,1 -> the bias cancels).
    callable: any user schedule f(x); used for the bandwidth family `parametrized_ascend`,
              e.g. schedule=sched_param(lam). Applied to the (A,B) ramp AND the nudge, which
              is what the boundary-term analysis requires (all tau-dependence of H must have
              the vanishing derivative, not just part of it).
    """
    if callable(schedule):
        return schedule(jnp.clip(x, 0.0, 1.0))
    x = jnp.clip(x, 0.0, 1.0)
    if schedule == "smooth":
        return 0.5 * (1.0 - jnp.cos(jnp.pi * x))
    return x


def parametrized_ascend(x, lam):
    """Bandwidth-controlled ascending function f_lam on [0,1] (QEP_GPT-12.pdf follow-up).

    With  B_lam(x) = (1-e^{-lam x})(1-e^{-lam(1-x)})/(1-e^{-lam}),  I_lam = coth(lam/2)-2/lam,
    H_lam(x) = int_0^x B_lam,  and  S(x) = 10x^3-15x^4+6x^5,
        f_lam(x) = H_lam(x) + (1-I_lam) S(x).
    Properties (all verified numerically to machine precision):
      f(0)=0, f(1)=1;  f'(0)=f'(1)=0;  monotone increasing;
      f''(0)= +lam,  f''(1)= -lam   <-- lam IS the boundary second derivative;
      lam -> infinity recovers the linear (stiff) schedule f(x)=x.
    Because f'(0)=f'(1)=0 the O(1) gradient bias of the linear schedule is cancelled for every
    lam; lam then tunes the leading RESIDUAL, which the theory predicts scales as lam/T.
    Anchor: the project's standard smooth ramp (1-cos pi x)/2 has f''(0)=pi^2/2 ~ 4.9348, so
    lam = pi^2/2 is its counterpart in this family.
    """
    r = jnp.exp(-lam)
    I = 1.0 / jnp.tanh(lam / 2.0) - 2.0 / lam
    H = ((1.0 + r) * x
         - (1.0 - jnp.exp(-lam * x)) / lam
         - (jnp.exp(-lam * (1.0 - x)) - r) / lam) / (1.0 - r)
    S = 10.0 * x ** 3 - 15.0 * x ** 4 + 6.0 * x ** 5
    return H + (1.0 - I) * S


def d_parametrized_ascend(x, lam):
    """Analytic derivative f_lam'(x) = B_lam(x) + 30(1-I_lam) x^2 (1-x)^2."""
    I = 1.0 / jnp.tanh(lam / 2.0) - 2.0 / lam
    B = ((1.0 - jnp.exp(-lam * x)) * (1.0 - jnp.exp(-lam * (1.0 - x)))
         / (1.0 - jnp.exp(-lam)))
    return B + 30.0 * (1.0 - I) * x ** 2 * (1.0 - x) ** 2


def sched_param(lam):
    """The `schedule=` argument implementing parametrized_ascend at bandwidth lam."""
    return lambda x: parametrized_ascend(x, lam)


SMOOTH_LAM = float(np.pi ** 2 / 2)      # f''(0) of the standard (1-cos pi x)/2 ramp


def htau(h):
    """Target Hamiltonian -h_z sigma^z - h_x sigma^x."""
    return -h[0] * SZ - h[1] * SX


def E0(h):
    return -float(jnp.linalg.norm(jnp.asarray(h, dtype=jnp.float64)))


def gs_of(Hq):
    """Ground state (lowest eigenvector) of a Hamiltonian QArray, as a ket QArray."""
    _, v = jnp.linalg.eigh(Hq.to_jax())
    return dq.asqarray(v[:, 0:1], dims=(2,))


def y_analytic(p, fxs):
    """Closed-form response y(f_x) = A / sqrt(A^2 + (B+f_x)^2) for params p=(A,B)."""
    A, B = p[0], p[1]
    g = B + jnp.asarray(fxs)
    return A / jnp.sqrt(A ** 2 + g ** 2)


def equilibrium(p, fx, T, beta, err, Nsteps, method, schedule="smooth",
                nudge_ramp=True, probe_ramp=False, tol=1e-9):
    """Equilibrium (ground) state of  H_sys(p) - fx sigma^x + beta*err*sigma^z  at t=T.

    method='exact'     -> exact diagonalisation (T-independent reference)
    method='adiabatic' -> Schroedinger evolution under the smooth/linear schedule

    Two protocol switches (both defaults reproduce the original study bit-for-bit; they
    exist to test the conventions of QEP_multiple_shot/Adaptive_bn/qep_shot):

    nudge_ramp=True  (default) -- the nudge is switched ON SMOOTHLY, beta*err*ramp(t/T),
                     exactly as afm.py's QM.total_H/tune_func = ascend_func(t/T) does.
                     nudge_ramp=False is the KNOWN BUG variant (nudge at full strength
                     from t=0, as in ../single_qubit/data/make_param_paths.py); kept only
                     as a control to quantify what the ramp buys. No-op for method='exact'.

    probe_ramp=False (default) -- the probe field fx is held at full strength for the whole
                     run, and psi0 is the exact GS of the start Hamiltonian INCLUDING the
                     probe, so the evolution still begins in an eigenstate.
    probe_ramp=True  -- the h_learning/QEP_grad_PF convention: the probe is folded into B
                     (add_params -> (A, B+F)) and ramped with it, so psi0 EXCLUDES the probe.
                     No-op for method='exact' (same t=T Hamiltonian either way).

    tol -- Tsit5 rtol=atol. The default 1e-9 is the study-wide setting (tight because the QEP
           gradient divides an O(beta) signal by beta). Tighten it only for observables that
           are themselves smaller than the 1e-9 floor: the state-prep infidelity of a smooth
           schedule falls as 1/T^4 and saturates on the solver at ~5e-8 by T=100, so
           run_boundary.py uses tol=1e-13. No-op for method='exact'.
    """
    A, B = p[0], p[1]
    if method == "exact":
        return gs_of(-A * SZ - (B + fx) * SX + (beta * err) * SZ)
    tl = jnp.linspace(0.0, T, Nsteps)
    fx0 = 0.0 if probe_ramp else fx                  # is the probe present at t=0?
    psi0 = gs_of(-A0 * SZ - (B0 + fx0) * SX)         # exact GS of the start Hamiltonian
    Af = lambda t: A0 + (A - A0) * ramp(t / T, schedule)
    if probe_ramp:
        Bf = lambda t: B0 + (B + fx - B0) * ramp(t / T, schedule)      # probe ramped with B
    else:
        Bf = lambda t: B0 + (B - B0) * ramp(t / T, schedule) + fx      # probe constant
    nf = ((lambda t: beta * err * ramp(t / T, schedule)) if nudge_ramp
          else (lambda t: beta * err + 0.0 * t))     # ramped 0->beta*err, or full from t=0
    H = (dq.modulated(lambda t: -Af(t), SZ)
         + dq.modulated(lambda t: -Bf(t), SX)
         + dq.modulated(nf, SZ))
    return dq.sesolve(H, psi0, tl, exp_ops=[],
                      method=dq.method.Tsit5(rtol=tol, atol=tol),
                      options=dq.Options(progress_meter=False)).states[-1]


def measure(p, fx, T, Nsteps, method, schedule="smooth", nudge_ramp=True, probe_ramp=False):
    """Output y = <sigma^z> for the free-phase equilibrium under probe fx."""
    return jnp.real(dq.expect(SZ, equilibrium(p, fx, T, 0.0, 0.0, Nsteps, method, schedule,
                                              nudge_ramp, probe_ramp)))


def response_curve(p, fxs, T, Nsteps, method, schedule="smooth", nudge_ramp=True, probe_ramp=False):
    """y(fx) for every probe in fxs (vectorised)."""
    return jax.vmap(lambda f: measure(p, f, T, Nsteps, method, schedule,
                                      nudge_ramp, probe_ramp))(fxs)


def make_dataset(h, fxs):
    """Target response y_tau(fx) from the EXACT ground state of H_tau - fx sigma^x."""
    return jax.vmap(lambda f: jnp.real(dq.expect(SZ, gs_of(htau(h) - f * SX))))(fxs)


def qep_grad_single(p, fx, y_tau, T, beta, Nsteps, method, schedule="smooth",
                    nudge_ramp=True, probe_ramp=False):
    """QEP estimate of dC_i/d(A,B) for one probe, C_i=0.5*(y-y_tau)^2. Returns (y, grad)."""
    pf = equilibrium(p, fx, T, 0.0, 0.0, Nsteps, method, schedule, nudge_ramp, probe_ramp)
    y = jnp.real(dq.expect(SZ, pf))
    err = y - y_tau                                   # error_signal = dC/dy
    pn = equilibrium(p, fx, T, beta, err, Nsteps, method, schedule, nudge_ramp, probe_ramp)
    df = jnp.array([-jnp.real(dq.expect(SZ, pf)), -jnp.real(dq.expect(SX, pf))])
    dn = jnp.array([-jnp.real(dq.expect(SZ, pn)), -jnp.real(dq.expect(SX, pn))])
    return y, (dn - df) / beta


def qep_grad_persample(p, fxs, y_taus, T, beta, Nsteps, method, schedule="smooth",
                       nudge_ramp=True, probe_ramp=False):
    """Per-probe QEP gradients; returns (ys (K,), grads (K,2)).

    Used to visualise the gradient DIRECTION of each single input-output pair (f_x, y_tau)
    before they are averaged into the update direction.
    """
    return jax.vmap(lambda f, yt: qep_grad_single(p, f, yt, T, beta, Nsteps, method, schedule,
                                                 nudge_ramp, probe_ramp))(fxs, y_taus)


def qep_grad_batch(p, fxs, y_taus, T, beta, Nsteps, method, schedule="smooth", reduce="sum",
                   nudge_ramp=True, probe_ramp=False):
    """Reduce the per-probe QEP gradients over the dataset; returns (y_vector, grad).

    reduce='mean' -> AVERAGE gradient over the dataset (the update rule used for training,
                     see train_experiments.py / the training report).
    reduce='sum'  -> total gradient (default; the convention of run_experiments.py, which
                     compares against num_grad_batch of the summed cost).
    """
    ys, gs = qep_grad_persample(p, fxs, y_taus, T, beta, Nsteps, method, schedule,
                                nudge_ramp, probe_ramp)
    g = jnp.mean(gs, axis=0) if reduce == "mean" else jnp.sum(gs, axis=0)
    return ys, g


def cost(p, fxs, y_taus, T, Nsteps, method, schedule="smooth", nudge_ramp=True, probe_ramp=False):
    """Total fitting cost C = 0.5 * sum_i (y_i - y_tau_i)^2."""
    ys = response_curve(p, fxs, T, Nsteps, method, schedule, nudge_ramp, probe_ramp)
    return 0.5 * jnp.sum((ys - y_taus) ** 2)


def num_grad_batch(p, fxs, y_taus, T, Nsteps, method, eps=1e-4, schedule="smooth",
                   nudge_ramp=True, probe_ramp=False):
    """Central-difference gradient of the total cost (reference for the QEP gradient)."""
    p = jnp.asarray(p, dtype=jnp.float64)
    C = lambda q: cost(q, fxs, y_taus, T, Nsteps, method, schedule, nudge_ramp, probe_ramp)
    return jnp.array([(C(p.at[i].add(eps)) - C(p.at[i].add(-eps))) / (2 * eps)
                      for i in range(p.shape[0])])


def train_scan(p0, fxs, y_taus, T, beta, lr, N, Nsteps, method,
               use_adam=False, schedule="smooth", momentum=0.0, reduce="sum",
               nudge_ramp=True, probe_ramp=False, max_rel_step=None):
    """Train (A,B) for N epochs on the probe dataset. Returns the (N+1, 2) trajectory.

    reduce='mean' -> the AVERAGE QEP gradient over the dataset is used for each update
                     (the convention of the training report). 'sum' is the default to keep
                     run_experiments.py reproducible.

    max_rel_step -- OPT-IN relative trust region: cap each update at max_rel_step * |p|, i.e.
                    no step may move the parameters by more than that fraction of their current
                    distance from the origin. None (default) leaves every existing result
                    untouched.

                    Why relative and not a plain smaller lr: for a single probe at f_x = 0 the
                    output y = A/sqrt(A^2+B^2) depends only on the POLAR ANGLE, so the gradient
                    is purely tangential with |grad y| = |B|/R^2 -- it DIVERGES at the origin.
                    An init close to the origin therefore takes a step comparable to R itself
                    and overshoots the angle. Lowering lr does not fix that (it makes the walker
                    linger in the high-gradient region and oscillate for longer); capping the
                    step relative to R does, and is scale-free, so it stops binding as soon as
                    the trajectory reaches a well-conditioned radius.
    """
    p0 = jnp.asarray(p0, dtype=jnp.float64)
    v0 = jnp.zeros_like(p0)
    if use_adam:
        opt = optax.adam(lr); st0 = opt.init(p0)
    else:
        st0 = v0

    def body(carry, _):
        p, st = carry
        _, g = qep_grad_batch(p, fxs, y_taus, T, beta, Nsteps, method, schedule, reduce,
                              nudge_ramp, probe_ramp)
        if use_adam:
            u, st = opt.update(g, st); p = optax.apply_updates(p, u)
        else:
            st = g + momentum * st
            step = lr * st
            if max_rel_step is not None:
                cap = max_rel_step * jnp.linalg.norm(p)
                nrm = jnp.linalg.norm(step)
                step = step * jnp.minimum(1.0, cap / (nrm + 1e-30))
            p = p - step
        return (p, st), p

    (_, _), tr = lax.scan(body, (p0, st0), None, length=N)
    return jnp.concatenate([p0[None], tr], axis=0)


def param_error(P, h):
    """Euclidean distance |p - h| for a (..,2) array of params to the target h."""
    return np.linalg.norm(np.asarray(P) - np.asarray(h), axis=-1)
