# Plan to plot figure for AE+QEP for AFM searching

With the code and the data we have, I want to plot the following figures.

1. In QEP_GPT-12.pdf there is a theory about $O(1)$ bias. I would like you to plot figures showing how the gradient components change with the total time of evolution growing. See what is plotted in ```H_detecting/figs/bd_grad_vs_T.pdf```. Plot such figures for 1d-chain with N=1,3,5,7,9, and also for 2d lattices.
2. Show paths of parameter in a $\Omega-\Delta$ plane, with the phase diagram, which describes how parameters are updated during the training. Draw such figures for 1d-chain for N=5 and 9, and 2d lattices. Show both paths with long $T$ and small $T$ for smooth and stiff schedule on the same picture for specific chain/lattice. 
3. Plot the performance of training shown with $\langle AFM^2 \rangle$ against the total time used (#epochs times $T$). See ```/zeropoint/u/qinwang/notebooks/QEP_sparse/Fixed_TN/physical_time.pdf```
4. Plot the performance of training measure by how it changes with time of evolution within a time budget, see ```/zeropoint/u/qinwang/notebooks/QEP_sparse/Fixed_TN/time_budget.pdf``` and ```/zeropoint/u/qinwang/notebooks/QEP_sparse/Fixed_TN/time_budget_heat.pdf```. 

When you plot the figures, save the data used (or a copy of data) in ```AFM_searching/paper_figs/data```, figures in ```AFM_searching/paper_figs/figs```, codes in ```AFM_searching/paper_figs/code```. 