# This set up the training tasks for figures in the thesis.

We will focus on QEP with finite shot and the effect of shot numbers. We have already shown that the training is more sensitive on the shot number used for obtaining the gradient $N_g$ (as seen in ```Adaptive_bn/test_richardson.ipynb```)because of the amplification of $\beta$. So here we will study this effect quantitatively. 

Do the following with searching for antiferromagnetic (AFM) phase on a 1d chain with open boundary condition for $N=5, 9$. When training, always use $\beta=0.1$ and $\eta=0.01$. During the training, we fix $N_g$ and do not use adaptive methods. 

1. Show that the effect is real by plotting how the distribution of gradients on $\Omega$ and $\Delta$ changes as one have smaller $\beta$ and larger $N_g$. I expect that when $\beta$ is fixed one will see the distribution of gradient components converges to a deterministic value with square variance ~ $1/N_g$ or standard variance ~ $1/\sqrt{N_g}$, and when $N_g$ is fixed one will see that as $\beta$ become smaller the variance of the distribution of gradient components becomes smaller and then become bigger. 
2. Select 3 randomly chosen initial parameter $(\Omega, \Delta)$, evolve with different $N_g=10, 100, 1000, 10000$. Plot single figures for each $N_g$. In each figure, plot 3-5 sample paths of parameters during the training with multiple different seeds for each initial parameters. I expect that I will see as one have larger $N_g$ the paths will change from random work to smooth deterministic curves. Also plot the path on the phase diagram. 
3. Plot figures showing the performance of training measured by how $\langle AFM^2\rangle$ changes as one have more training epochs, more real shot times ($N_g \times 2 \times N_{epoch}$, the multiplication comes from the fact that $\sigma^x_i$ and $\sigma^z_i$ do not commute and measuring the gradient requires 2 sets of experiments). See what is shown in ```/zeropoint/u/qinwang/notebooks/QEP_sparse/Fixed_TN/physical_time.pdf``` and the relevant code. 
4. Set a budget for total number of experiments $N_{g, budget}$. For different $N_{g, budget}$, show how the performance changes when one have larger $N_g$ per epoch. See what is shown in ```/zeropoint/u/qinwang/notebooks/QEP_sparse/Fixed_TN/time_budget.pdf``` and ```/zeropoint/u/qinwang/notebooks/QEP_sparse/Fixed_TN/time_budget_heat.pdf```
5. For a typical single training case, draw how the expectation and standard variance of gradient components changes during training. Draw how $\sigma(g_\theta)/\langle g_\theta \rangle$ evolves as the training going on ($\theta=\Omega, \sigma$). I expect to see that with relatively large $N_g$, $\sigma(g_\Omega)$ converges to nonzero constant while $\sigma(g_\Delta)$, $\langle g_\Omega \rangle$ and $\langle g_\Omega \rangle$ converge to 0. Which shows that one will need more shots for $\Omega$, and it leads to the exploration of adaptive methods. 

When you are doing the training, you should: 

1. Use ```Adaptive_bn/qep_shot.afm.Ising_Chain_AM2```. Training with $AFM^2$. Also make sure the switch on of nudging phase is smooth. 
2. Use ```Adaptive_bn/qep_shot.grad.Shot_QEP_grad``` to get the gradient. 
3. Write python scripts that can run without ```Adaptive_bn/qep_shot``` and use them to run your tasks. You can do it by setting it as a package of the workspace. 
4. Do necessary modification if there are bugs and if the code does not satisfy what I ask. 
5. Use cluster raven for training after testing everything locally.
6. Download the data. 
7. Document the structure of code and research. 
8. Store the log in folder ```QEP_multiple_shot/AFM_searching/logs```. 

After doing everthing, write a latex report in ```QEP_multiple_shot/AFM_searching/reports/2026-09-11```
