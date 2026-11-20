# Finite-field digital nets for diffusive torsional energy landscapes

Code and data for the article

> **Abraham Rebairo J.**, *Sampling diffusive torsional energy landscapes with finite-field digital nets: theory, scope and limits*, submitted to the *Journal of Mathematical Chemistry* (2026).

The repository reproduces every numerical result, table and figure of the article and its Supplementary Information. It analyzes digital nets conformational sampling (DNCS), introduced in [Phys. Chem. Chem. Phys. **26**, 22640–22655 (2024)](https://doi.org/10.1039/d4cp01891e). DNCS generates the torsion angles of polypeptides from digital nets over the finite field GF(2). The calculations cover six topics:

- **Fractional relaxation:** Mittag-Leffler functions, their exponential-sum approximations, and the Grünwald–Letnikov memory recurrence.
- **Galois analysis:** Galois groups and Chebotarev statistics of the memory polynomials modulo primes.
- **Shift registers:** primitive-polynomial shift registers over GF(2) acting on the torsion circle as a discretized doubling map.
- **Point sets:** discrepancy of Frobenius-coupled constructions, digital (Sobol) nets and Korobov lattices; Ramachandran-map coverage.
- **Sampler comparison on smooth landscapes:** quadrature versus Metropolis and Langevin (MALA) chains on chain landscapes with exact transfer-matrix references, and multistart minimization.
- **Heavy-tailed trap landscapes:** memory, autocorrelation, the variance law of time averages, and independence Metropolis chains with net-based or CUD-driven proposals.

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.26
- SciPy ≥ 1.11 (uses `scipy.stats.qmc`)
- Matplotlib ≥ 3.8

```bash
pip install -r requirements.txt
```

The published results were produced with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11. All random numbers come from fixed seeds, so every result is reproducible exactly.

## Repository layout

```
code/   Python scripts (run them from inside this folder)
data/   raw numerical results written by the scripts (JSON and NumPy files)
fig/    figures of the main text (PDF and PNG)
```

The `data/` and `fig/` folders already contain the published results. Re-running the scripts overwrites them with identical values.

## Reproducing the results

```bash
cd code
python r31_ml.py          # and so on, in the order below
```

| Step | Script | What it computes | Results in the article |
|---|---|---|---|
| 1 | `r31_ml.py` | Mittag-Leffler functions, exponential-sum fits | Fig. 1; Tables S4, S5 |
| 2 | `r32_gl.py`, `r32b_cheb.py` | Grünwald–Letnikov recurrence; memory polynomials mod *p*, Galois tests | Fig. 2, Table B2; Tables S6, S7 |
| 3 | `r33_lfsr.py` | primitive shift registers on the circle | Fig. 3, Tables 1, A1 |
| 4 | `r34_disc.py` | discrepancy of all point sets | Fig. 4, Table 2; Tables S8, S9 |
| 5 | `r35_rama.py` | Ramachandran coverage, *t*-values | Figs. 5, 6; Table S11 |
| 6 | `r36_conv.py` | quadrature on chain landscapes | Figs. 7, 8, Table 4; Table S12 |
| 7 | `r37_lat_irr.py` | Korobov lattices; primitive vs irreducible nets | Fig. 4, Tables 2, 3; Table S10 |
| 8 | `r38_dyn.py` | quadrature vs Metropolis / MALA chains | Fig. 9; Figs. S2, S3, Table S13 |
| 9 | `r39_multistart.py` | net-seeded vs random-seeded MALA chains | Table S15 |
| 10 | `r40_basins.py` | multistart minimization | Table 6; Table S16 |
| 11 | `r41_ci.py` | bootstrap confidence intervals | Table 4, Fig. 9; Table S14 |
| 12 | `r42_trap.py` | trap landscapes: all samplers, autocorrelation | Fig. 10, Table 5; Tables S17, S18 |
| 13 | `r43_variance.py` | variance law of the time average | Fig. 10; Table S18 |
| 14 | `r44_qmcprop.py` | independence Metropolis with net proposals | Fig. 10, Table 5; Tables S17, S18 |
| 15 | `figs.py`, `fig_embed.py` | figures of the main text → `fig/` | Figs. 1–10 |
| 16 | `si_build.py` | tables and figures of the Supplementary Information → `si/` | Tables S2–S16, Figs. S1–S3 |

Most steps take between a few seconds and about 15 minutes on a laptop. Step 12 is the longest and takes tens of minutes. Steps 15 and 16 need only the files in `data/`, so the figures can be regenerated without re-running the calculations.

## Shared modules

| Module | Contents |
|---|---|
| `samplers.py` | pseudo-random points, Sobol nets (scrambled and unscrambled), Frobenius-conjugate and shifted-copy constructions, primitive-polynomial table |
| `landscape.py` | chain landscapes on the torus, gradients, exact ln *Z*, ⟨*U*⟩ and marginals from transfer matrices |
| `polyutil.py` | Grünwald–Letnikov weights, arithmetic over GF(*p*), distinct-degree factorization, primitivity test |
| `rama_model.py` | model Ramachandran distribution (three bivariate Gaussian basins) |
| `trap.py` | Bouchaud-type trap landscapes (narrow Gaussian wells with stratified exponential depths) and their exact averages |

## Citation

If you use this code, please cite the article above; citation details will be updated on publication. Machine-readable metadata are in `CITATION.cff`.

## License

Released under the MIT License (see `LICENSE`).

## Contact

Dr. Abraham Rebairo J., Department of Physics, Madras Christian College, Chennai, India — abraham.rebairo@mcc.edu.in
