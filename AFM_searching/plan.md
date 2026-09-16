# Plan of local study on AFM searching

Antiferromagnetic (AFM) phase exploration is an interesting problem in condensed physics. In this project we will study it with quantum equilibrium propagation with finite time adiabatic evolution. There is a nice example on cluster zeropoint, which is in /zeropoint/u/qinwang/notebooks/QEP_sparse/Fixed_TN. 

Your work is to repeat all studies done on the cluster project and plot all the figures. 

However, you need to note the following: 

1. On the cluster project, what is used for training is QEP.model.Quantum_Model_DQ. This class has a different definition of interaction and its nudging phase is switched on abruptly. In the local project, you need to train with Ising_Chain. Since they have different definition on the interaction (multiplication of 2 or 0.5), you need to re-draw the phase graph. You can know what smooth nudging is in ```QEP_multiple_shot/Adaptive_bn/qep_shot.afm.QM.total_H```. You should train with ```qep_shot.afm.Ising_Chain``` and ```qep_shot.afm.Ising_Chain_AM2```. 
2. Train with exact QEP (QEP with finite $\beta$ but exact ground state) and QEP with imperfect state obtained with adiabatic evolution. Details can be found on the cluster project. 
3. On the cluster the networks are trained with nearest neighbor correlation, while in local project you should use $AFM = \sum_i (-1)^i \sigma_i^z$ and $AFM^2$. You can verify this point in the definition of the relevant project.

You are supposed to do: 

1. Read the files of the cluster project and relevant projects on the cluster.
2. Repeat the same study with same $N_{epoch}$ and $N_{budget}$ and other hyperparameters. 
3. Repeat the experiments on $N=1,3,5,7,9$.
4. From ```AFM_searching/QEP_GPT-12.pdf```, you can see a theory of $O(1)$ bias of gradient. Due to the contribution from different level and the exponentially growing state space dimension, such bias should vanish for large $N$. Verify it. 
5. Plot the same figures as in the cluster project for $N=5,9$.
6. Try also 2d systems. In this case you first should have a look at the ground state of $2\times 2$ and $3\times 3$ and $4\times 4$ lattices (you can use ```networks.Ising_Lattice``` but adjust it with smooth switch on of nudging phase as in ```qep_shot.afm.QM```). If there are no degeneracy or the ground state is the superposition of 2 AFM states, then continue the study on it. Also do the same experiments and draw the same figures.
7. Try with multiple initial parameters. 

When you are working should note:
1. Build up scripts which does not depend on scripts in other projects. Store them in ```AFM_searching/code```
2. Download the data into ```AFM_searching/data```
3. Put figures in ```AFM_searching/figs```. Separate figures for different network sizes and different dimensions.
4. Draw your observation and write a latex report in ```AFM_searching/report```. When you are writing, use relatively plain and understandable language while keep the usage of scientific words precise. 

You can do computing on cluster raven. After verifying that the code works locally, you should move all the training onto the cluster raven or zeropoint. Try to do training of different sizes and different initial parameters in parallel to save time. You can know how to use it in ```cluster_using.md```.