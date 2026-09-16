# QEP paper figures

Code and figures for three Quantum Equilibrium Propagation (QEP) studies
built on the [QEP method](https://www.nature.com/articles/s41467-025-61665-6)
(two-phase adiabatic evolution for computing gradients of quantum systems).

**This package intentionally does not include the raw result data
(`.npz`/`.pkl`/`.json`)** used to build the figures — only the code, the
notebooks, and the already-rendered figure images. To reproduce a figure
from scratch, first run the data-generation script(s) listed below for that
project (they write into a local `data/` directory next to the notebook),
then run the notebook.

## Dependencies

```bash
pip install jax dynamiqs qutip numpy matplotlib diffrax optax scipy
```

## Layout

```
QEP/                          core QEP package (model, gradient, training loop)
AFM_searching/                antiferromagnetic order-parameter search, chains + 2D lattices
QEP_multiple_shot/AFM_searching/   finite-shot-noise version of the same search
H_detecting/                  detecting a probe Hamiltonian via QEP-trained response
```
