# Bipartite‑Nonclassicality — Data & Code

This repository accompanies our study of non‑classical correlations in bipartite
quantum systems. It contains

* raw coincidence‑count data for a range of isotropic states,
* the inequalities used to certify (un)steerability and locality,
* Python scripts for tomography, optimisation, and Monte‑Carlo validation.


### Data (`Counts_new/`)

* Each `.npy` file stores a **12 × 20** integer array  
  (`detector × measurement‑setting`) representing raw Poissonian counts.
* File names encode the **estimated isotropic‑state parameter \(p\)**.

### Inequality files (`Inequality/`)

| file | purpose |
|------|---------|
| `Inequalities.txt` | Full list of Example‑2 inequalities output by **PORTA** |
| `Ineqcleanup68.py` | Cleans & classifies the above into **4 inequivalent classes** |
| `Farka68.py`       | Re‑derives Example‑2 inequalities using **Farkas' lemma** |
| `Farka1220.py`     | Same derivation for **Example 3** |

The ready‑to‑use inequality vector for our experiment is saved at the top level
as **`y_ineq.npy`**.


### Key scripts

| script | role |
|--------|------|
| `Optimized‑tomography‑all‑monte.py` | Performs maximum‑likelihood tomography and estimates the best‑fit \(p\) for each dataset. |
| `Violation_monte_new.py`            | Propagates Poisson noise via Monte‑Carlo and checks violations of `y_ineq.npy`. |
| `CheckUnsteering.py`                | Searches for an unsteerable decomposition (SDP‑based). |

