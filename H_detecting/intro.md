# In this project we use QEP to approximate known Hamiltonians

## Single qubit

In single_qubit test, we made it tp approach the Hamiltonian up to an amplitude. 

Now we try to fix this by using a probing field. Specifically, assume that the target Hamiltonian is $H_\tau = - h_x \sigma^x - h_z \sigma^z$. 

Assume that at the begining our Hamiltonian is $H_0 = -B \sigma^x - A \sigma^x$. The process is done as below: 

1. We perturbe $H_0$ with a probing field along x-axis by making $H = -f_x \sigma^x + H_0(A,B)$
2. We get the ground state of $H$, note it as $|\psi_0\rangle$
3. We measure it along z-axis and get $y = \langle \psi_0 | \sigma^z | \psi_0 \rangle$
4. Assume that we knoe the result $y_\tau = \langle \psi | \sigma^z | \psi \rangle$ for $|\psi\rangle$ to be the ground state of $H_\tau$, we apply quantum equilibrium propagation according to this difference. Specifically, we apply a nugde term to $H_0$ and have $H_\beta = H_0(A,B)  -f_x \sigma^x + \beta \epsilon \sigma^z$ where $\epsilon = y - y^\tau$ is the error signal, and $\beta$ is a small number in quantum equilibrium propagation. 

What one should do: 

1. Prepare the measurement result in different amplitude of probing field got from the target Hamiltonian $H_\tau$, save as pairs of input and output $(f_x, y_\tau)$. 
2. Perturb $H_0$ with the same set of $f_x$ and calculate $y(f_x)$. Use single probing field or a subset of dataset. 
3. Run quantum equilibrium propagation, get the gradient of the parameters.
4. Updat $A$ and $B$.
5. Repeat 1-4

## Test example

Test the validity of code with the following samples:

Assume that the target Hamiltonian is $H_\tau = -\sigma^z$ and we start with $H_0 = -\sigma^x$. Then the perturbation along the x-axis will lead to no perturbation in the result of $H_0$ (the state stays $x$-polarised, so $\langle\sigma^z\rangle = 0$ for all $f_x$), and for $H_\tau$ we should have $y = \frac{1}{\sqrt{1 + f_x^2}}$.

In general case, for the target, we should have a response curve of the form

$$ y(f_x) = \langle \sigma^z \rangle = \frac{h_z}{\sqrt{h_z^2 + (f_x + h_x)^2}} . $$

(The exact ground-state magnetization carries a **square root** and saturates at $1$; the peak sits at $f_x = -h_x$ with width $\sim h_z$, and the tail decays as $1/|f_x|$.)

Note: at the single field $f_x = -B$ the total $\sigma^x$ coefficient of $H_0$ vanishes; if additionally $A = 0$ (as for $H_0 = -\sigma^x$) the whole Hamiltonian is zero and its ground state is degenerate, so $\langle\sigma^z\rangle$ is undefined there — that probe value is excluded.

## Training

Train the network with gradient descent. 

The gradient should be obtained through QEP with the process mentioned above. The parameters are updated with the average gradient obtained from the all the points in the dataset. 

Try to draw the direction of gradient for single input-output pair and the direction of the average gradient. 

Redo the training with what mentioned above, analyze the role of time of adiabatic evolution in ground state preparation, and write a new report. 
