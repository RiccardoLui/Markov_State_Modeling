This is an attempt at a molecular dynamics implementation. In particular, a sequence of the TAU protein was isolated from the Alphafold database and molecula dynamics simulations were performed.

The trajectory of the peptide was simulated using OpenMM and then a Markov State Model was implemented on these trajectories using Deeptime.
Finally, the same was done for a mutated peptide, to highlight differences in the dynamics.

More details and the results can be found in the report.

run_phf6_replicas_openmm_patched.py    is used to generate the trajectories
preprocess_phf6_data_joint.py          finds the implied timescales of the peptides
run_msm.py                             performs the MSM

The other scripts are for plotting and diagnostics.
