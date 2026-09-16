# Why finite-beta QEP decays although the infinitesimal response oscillates

15 September 2026. Diagnostic of the current N=1 amplitude figures.

## Conclusion

The current data do not refute the O(1) derivative mechanism in
`../QEP_GPT-12.pdf`. After this diagnostic, the user pointed to the explicit
QEP extension in `/Users/qwang/ShareLatex/QEP/main.tex`, section 2.4.4,
lines 317–340. That section applies the same phase-derivative argument
directly to the nudging derivative at nu=0. **The QEP extension is therefore
already present; the main remaining distinction is infinitesimal response
versus fixed finite beta**, not the absence of a QEP argument. Two distinctions
help identify exactly which result is tested:

1. Equation (33) is a derivative **through the finite-time evolved output**.
   The production plots instead show a **finite-beta QEP response** of final
   Hamiltonian observables. Ground-state reciprocity does not identify these
   two quantities for a generic finite-time unitary state. However, section
   2.4.4 does not require such an identification: it separately differentiates
   the evolved expectation of A_theta with respect to nu. This supplies the
   appropriate infinitesimal-QEP version of the mechanism.
2. For this N=1 implementation, the infinitesimal QEP response also has a
   nearly T-independent oscillatory RMS, but beta=0.1 ceases to approximate
   that infinitesimal response at long T. The new beta sweep isolates this
   finite-nudge effect. Solver error is far too small to explain the decay.

Thus the primary observed crossover is a **nonuniform expansion in beta**,
not evidence that higher orders in the adiabatic 1/T expansion cancel the
leading derivative oscillation. This conclusion does not certify every
intermediate equation or coefficient in the PDF.

## The three quantities

Let C=Z for N=1, y_T=<C> in the free evolved state, and L=(y_T-1)^2.
The diagnostic distinguishes

\[
 D_\theta(T)=\partial_\theta L_T
 =2(y_T-1)\partial_\theta y_T,
\]

\[
 G_\theta(T,0)=\left.\partial_\beta
 \langle A_\theta\rangle_{\mathrm{nudged},T,\beta}\right|_{\beta=0},
 \qquad A_\theta=\partial_\theta H_{\mathrm{final}},
\]

\[
 G_\theta(T,\beta)=
 \frac{\langle A_\theta\rangle_{\mathrm{nudged},T,\beta}
       -\langle A_\theta\rangle_{\mathrm{free},T}}{\beta}.
\]

The PDF Eq. (33), p.10, concerns the output derivative in the first expression.
Production `code/afmqep/normalized_ae.py:39–56` computes the third at beta=0.1.
The newer manuscript section 2.4.4 concerns the second expression, with
nu=beta*epsilon and epsilon held fixed during differentiation. Its leading
phase factor differentiates to -i*T*(partial_nu Phi), canceling the 1/T
prefactor. The new calculation supports that mechanism directly.
The nudge coefficient is beta*2(y_T-1), switched on by a raised cosine, even
when the free Hamiltonian schedule is stiff. Both free and nudged sweeps start
from the same initial state; this is ordinary AE, not cached AE.

The equilibrium identity in PDF Eq. (6), p.4, uses an exact ground state.
Its extension to finite-time prepared states needs a separate dynamical
argument. The original QEP reference likewise derives a **static ground-state
response**, not arbitrary finite-time reciprocity:
[Wanjura and Marquardt, Nature Communications 16, 6595 (2025), Eqs. (1)–(3)](https://www.nature.com/articles/s41467-025-61665-6).
The earlier local audit `main.tex`, section on estimator/theory mismatch,
already identified this distinction; the present work adds a direct test.

## Independent numerical experiment

New reproducible code: `check_n1_gradient_limits.py`.
Outputs: `n1_gradient_limits_20260915/summary.json` and three window NPZs.
No production physics, cluster jobs, original PDF, or canonical figures were
modified.

Independently implement the one-spin Bloch equation, with hbar=1:

\[
 \partial_s r=T h(s)\times r,\quad
 h=(\Omega(s),0,-\Delta(s)+2\beta\epsilon a(s)),\quad
 \epsilon=2(y_T-1),\quad a(s)=[1-\cos(\pi s)]/2.
\]

The omitted identity part of H contributes only a global phase. Initial
r=(0,0,-1). Base endpoint (Omega,Delta)=(1,2.6), with the same 0.2/0.8
schedule joins as production. Direct derivatives are calculated with tangent
equations rather than finite parameter differences:

\[
 \partial_s v_\theta=T[h\times v_\theta+
 (\partial_\theta h)\times r].
\]

A separate tangent for an infinitesimal smooth Z nudge gives G(T,0), avoiding
subtraction cancellation. Finite beta uses full nonlinear-in-beta evolution
at beta=0.1,0.01,0.001. The error signal is taken from the free finite-T state,
exactly as in production. All integrations are split at the internal corners.
Sample 81 T values at spacing 0.25 in each [T0-10,T0+10] window, for
T0=100,500,2000. RMS is standard deviation about the local window mean.

### Stiff Omega-component QEP RMS

| T0 | beta=0.1 | beta=0.01 | beta=0.001 | exact beta derivative |
|---:|---:|---:|---:|---:|
| 100 | 0.1604873 | 0.1665115 | 0.1664831 | 0.1664731 |
| 500 | 0.0424470 | 0.1591026 | 0.1601417 | 0.1601074 |
| 2000 | 0.0066942 | 0.1406951 | 0.1647132 | 0.1648249 |

At T=2000, beta=0.001 is within about 0.07% of the infinitesimal-response RMS,
while beta=0.1 is smaller by a factor of about 24.6. The plateau is recovered
by decreasing beta; simply increasing T at fixed beta tests a different limit.

The **direct loss derivative**, independently calculated via the free tangent
equation, has stiff Delta RMS 0.068154, 0.065528, 0.067433 at these same three
windows. Its Omega RMS is 0.007847, 0.006079, 0.006192. It likewise retains
oscillations, but is **not numerically equal** to the infinitesimal QEP response.
As a control, the smooth direct-loss RMS at T=2000 is only
(0.00008591,0.00097737). These finite-window observations support the mechanism;
they are not a proof of a strictly constant envelope at every T.

### Numerical controls

- Independent Bloch beta=0.1 gradients reproduce the stored normalized
  wavefunction gradients within **2.23e-8** across all 243 sampled T values.
- Maximum Bloch-vector norm drift is below **1.9e-13**.
- Tightening the diagnostic tolerance from 1e-11/1e-13 to 1e-13/1e-15 at
  T=100 and 2000 changes direct derivatives by at most **2.96e-10**, and
  finite-beta gradients (including beta=0.001) by at most **1.59e-9**.
- Independent SciPy DOP853 integration of the free Bloch/tangent equations
  agrees with Diffrax Dopri8 to **2.64e-8** or better at those two T values.
- The script records production input hashes and its own source hash.
- An additional production-wavefunction centered-parameter-difference check
  at T=100/2000, with parameter steps 1e-6 and 5e-7, reproduces the independent
  direct-loss tangent derivative within 7.5e-8 (within 2.8e-8 for the smaller
  step). This is a diagnostic check, not how the tangent results were obtained.

These errors are many orders below the approximately 0.16 versus 0.0067
amplitude difference. Numerical damping is not a plausible explanation here.

## Why the beta expansion is not uniform in T

For illustration, suppose one boundary contribution to an expectation is

\[
 R(T,\beta)=\frac{A}{T}\sin[T(\Phi+\beta K)].
\]

Its infinitesimal response is O(1):

\[
 \partial_\beta R(T,0)=AK\cos(T\Phi).
\]

But the finite difference is

\[
 \frac{R(T,\beta)-R(T,0)}{\beta}
 =AK\,\mathrm{sinc}\!\left(\frac{\beta TK}{2}\right)
 \cos\!\left(T\Phi+\frac{\beta TK}{2}\right),
 \qquad \mathrm{sinc}(x)=\frac{\sin x}{x}.
\]

It approximates the derivative only when |beta*T*K| is small. At fixed
nonzero beta its envelope is bounded by 2|A|/(|beta|T). Expanding the phase
in beta produces terms enhanced by powers of T: they cannot be discarded
uniformly as T grows. This is distinct from including higher orders in 1/T.
In section 2.4.4's notation, K=epsilon*partial_nu Phi, so the condition is
|beta*epsilon*T*partial_nu Phi| much smaller than one for each relevant phase.

For the actual N=1 free path, using the equilibrium error signal as a leading
phase-scale estimate gives K approximately 0.0980 for the full gap integral.
The exact beta=0.1 phase-integral shift in that approximation is 0.0098198,
so the accumulated shift is already approximately 1.96 radians at T=200 and
19.64 radians at T=2000. For beta=0.001 the leading shift at T=2000 is only
0.196 radians. The corresponding internal-corner phase origins also acquire
large shifts at beta=0.1. The code records these integrals. This explains the
scale of the crossover, but is not a one-frequency fit: multiple boundary
and corner terms interfere, and their coefficients depend on beta and y_T.
The PDF itself discusses interior boundary terms in section 6, pp.13–14.

At fixed beta, convergence of both state preparations to their equilibrium
references also bounds their QEP expectation difference. The dependence of the
nudged Hamiltonian on y_T adds a converging term if the final ground-state
projector is continuous. No nonzero asymptotic preparation error is required
in this finite-beta estimator. In contrast, a parameter derivative of a
vanishing oscillatory state error can remain O(1).

## Implication for the study

Keep the existing beta=0.1 figures as valid **finite-nudge estimator** results.
Do not call their decay a disproof of the PDF's derivative mechanism or an
asymptotic test of the infinitesimal response. For that test, use sensitivity
equations or a beta-versus-T convergence study, and distinguish direct
output/loss derivatives from QEP response. The current test establishes this
for N=1; it does not by itself validate the corresponding asymptotics for all
larger chains or the 4×4 trajectories.

For the manuscript, the useful clarification is not removal of section
2.4.4, but an explicit statement of this finite-beta validity condition and
order of limits when comparing its derivative formula to finite-nudge plots.
The ShareLaTeX source was read only and has not been edited.
