# Bipartite‑Nonclassicality — Data & Code

This repository accompanies our study of non‑classical correlations in bipartite
quantum systems. It contains

* raw coincidence‑count data for a range of isotropic states,
* the inequalities used to certify (un)steerability and locality,
* Python scripts for tomography, optimisation, and Monte‑Carlo validation.


### Data (`Counts_new/`)

* Each `.npy` file stores a **12 × 20** integer array  
  (`detector × measurement‑setting`) representing raw counts.
* File names encode the ** isotropic‑state parameter \(p\)**.

### Inequality files (`Inequality/`)

| file | purpose |
|------|---------|
| `Inequalities.txt` | Full list of Example‑2 inequalities output by **PORTA** |
| `Inequality17.py`     | Noncontextual inequalities Eq(17) for **Example 3** |
| `LP_example1234.py`     | Linear programming on  for **Example 1,2,3,4** and **TableIII** |
| `TableII.py` | Cleans & classifies inequalities in  `Inequalities.txt` into **4 inequivalent classes** |
| `TableIV.py` | Linear programming on entanglement certification with randomized measurement for  **TableIV**|
| `TableV.py` | Linear programming on entanglement certification with arbitrary entangled states for  **TableV**|
| `TableIX.py` | Linear programming on steering certification for  **TableVII**|
| `y_ineq.npy` | The ready‑to‑use inequality vector for our experiment|


### Key scripts

| script | role |
|--------|------|
| `Optimized‑tomography‑all‑monte.py` | Performs quantum state tomography and estimates the best‑fit \(p\) for each dataset. |
| `Violation_monte_new.py`            | Propagates Poisson noise via Monte‑Carlo and checks violations of `y_ineq.npy`. (Fig 5 and Table IX) |
| `Violation_monte_w.o.ineq.py`       | Propagates Poisson noise via Monte‑Carlo and checks violations without inequality. (Table VI)|
| `CheckUnsteering.py`                | Searches for an unsteerable decomposition (SDP‑based). |

