# CLD-MaOEA

Reference implementation, experiment drivers, and raw results for the paper:

> **Automated and Reproducible Per-Problem Algorithm Configuration for Many-Objective
> Optimization via a Closed-Loop Dominance–Criterion Chain**
> Yucheng Cai, Xin Chen\*, Gang Luo, Jun Huang, Yu Deng, Fuqiang Peng
> Air Force Early Warning Academy, Wuhan, China

This repository is the artifact behind the paper's availability statement — *"code, seeds,
and table-generation scripts will be released upon acceptance."* Concretely:

| Promised artifact | Location in this repository |
|---|---|
| **Code** — the CLD-MaOEA algorithm and every experiment driver | `src/`, `experiments/*.py` |
| **Seeds** — the complete four-batch seed protocol | documented in [Seed protocol](#seed-protocol); declared as a module-level constant in each driver |
| **Table-generation scripts** — CSV → LaTeX table / figure | `experiments/make_paper_assets.py`, `experiments/gen_t20.py`–`gen_t23.py` |
| **Raw results** — the CSVs those scripts consume | `experiments/results/` |

Raw results are included on purpose: twelve of the manuscript's tables then regenerate in
seconds instead of re-running the full benchmark (~100 CPU-hours), while the drivers remain
available for anyone who wants to reproduce the numbers from scratch. The remaining tables
are typeset from the output of the analysis scripts — see
[Table-generation coverage](#table-generation-coverage).

## Algorithm

CLD-MaOEA targets many-objective optimization problems (MaOPs), where dominance-based
selection pressure collapses as the objective count grows. It couples four mechanisms:

1. **Closed-loop dominance reinforcement** — a PID-style controller drives the CDAS dominance
   strength parameter `phi` from the population's online dominance state, so selection pressure
   responds to the run instead of being fixed a priori.
2. **Directional differentiation** — dominance strength is individualized per reference
   direction according to that direction's lag, focusing effort on under-covered regions.
3. **R2 greedy truncation with champion protection** — preserves distribution uniformity in
   high dimensions without discarding the best individuals.
4. **Per-problem activation criterion chain** — a three-stage decision procedure
   (S1 gap profile → S2 three-condition validation → S3 activation) that reproduces the
   archived per-instance manual configuration on all 32 benchmark instances (32/32) and
   rejects the false positives a naive gap-rule would activate.

## Requirements

- Python 3.14 (the paper reports Python 3.14; the reproduction commands below were last
  exercised on 3.14.7)
- Timings in the paper were measured on an Intel Core Ultra 9 275HX; all algorithms run
  **single-threaded**, so core count does not affect the comparison

```bash
pip install -r requirements.txt
```

> **Do not upgrade the pinned versions individually.** The results in the paper were produced
> with exactly this environment. This matters most for `pymoo==0.6.2`: the external comparison
> table and the generalization protocol both depend on its cone directions and algorithm
> implementations, and the analytic Pareto-front handling for the inverted/scaled variant
> family is patched for that version. Upgrading `pymoo` breaks same-round comparability.

## Repository layout

```
cld-maoea/
├── src/              # algorithm core: CLDMaOEA, controller, R2 truncation, archives, variants
├── configs/          # four predefined version JSONs (multi_modal.json = K.26 final baseline)
├── experiments/      # experiment drivers + analysis scripts
│   └── results/      # raw CSV results (the inputs to the table-generation scripts)
└── paper/            # output targets, created empty
    ├── tables/       # LaTeX tables written by make_paper_assets.py / gen_t2*.py
    ├── figures/      # convergence and phi-rho figures
    └── figures_spie/ # framework and criterion-chain diagrams (draw_fig1.py, draw_fig3.py)
```

Every driver resolves the project root from its own location, so the repository can be cloned
or renamed anywhere:

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

## Seed protocol

The seed sets are disjoint by construction — this is what makes the criterion chain's
reproduction claim meaningful rather than circular. Each driver declares its set as a
module-level constant; the name differs per driver (`SEEDS`, `SEEDS10`, `SEEDS_10`,
`SEEDS_ALL`, `SEEDS_S1`/`SEEDS_S2`, `TUNE_SEEDS`), so take the set from the table below
rather than assuming the name.

| Batch | Seeds | Purpose | Declared in |
|---|---|---|---|
| **Main evaluation (S2)** | `42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640` | 32 instances × 10 seeds external comparison | `m3_benchmark.py` |
| **S1 profiling** | `2001, 2002, 2003` | gap profiling; disjoint from S2 | `gap_profiler.py`, `criterion_replay.py` |
| **Held-out re-validation** | `1001`–`1010` | independent check of the chain decision | `s2_disjoint.py` |
| **Configurator tuning** | `11, 22` | disjoint from all of the above | `auto_config.py` |
| **Small diagnostic / ablation** | `42, 7, 2024` | mechanism-wise probes | `phi_rho_trace.py`, `escape_ablate.py`, `k18_*.py`, … |
| **M2 ablation** | `42, 7, 2024, 314, 99` | five-seed ablation | `m2_ablation.py` |

S1 must not be a subset of S2; `gap_profiler.py` records that invariant in a comment beside
its `SEEDS` declaration, but nothing enforces it — if you change either set, check the
disjointness yourself. Stage A of the generalization benchmark screens with the first 3 of the
S2 set (`--seeds 3`); Stage B runs all 10 (`--seeds 10`). The gate between them is an analyst
decision, not an automatic check.

## Reproduction

All drivers support **resume-on-interrupt**: results are appended incrementally to CSV, so a
partial run can be continued. Run commands from the repository root.

```bash
# 1. External comparison (32 instances x 10 seeds x 5 algorithms; --smoke for a fast check)
python experiments/m3_benchmark.py
python experiments/m3_benchmark.py --smoke

# 2. Criterion-chain replay (S1 -> S2 -> decide in one shot; reports agreement with K.26)
python experiments/criterion_replay.py all

# 3. Gap profiler (criterion chain stage S1; --problems/--ms extend the dimension sweep)
python experiments/gap_profiler.py

# 4. Generalization benchmark (18 instances, zero tuning; Stage A 3 seeds -> Stage B 10 seeds)
python experiments/gen_benchmark.py --check
python experiments/gen_benchmark.py --seeds 3 --workers 8
python experiments/m3_analyze.py --csv experiments/results/gen_results.csv
python experiments/gen_benchmark.py --workers 8

# 5. Convergence-curve data (3 instances x 5 algorithms x 3 seeds x 40 checkpoints)
python experiments/convergence_history.py --workers 8

# 6. Paper tables and figures (CSV -> paper/tables/*.tex + paper/figures/*.pdf, with assertions)
python experiments/make_paper_assets.py
```

Step 6 runs entirely off the shipped CSVs and needs no re-computation.

## Results data ↔ paper elements

| Paper element | Data source |
|---|---|
| External comparison main table, win/tie/loss, wall-clock | `experiments/results/m3_results.csv` |
| Generalization (18 instances) | `experiments/results/gen_results.csv` |
| Criterion-chain decision table | `experiments/results/criterion_replay_decision.md`, `gap_profile.csv`, `route1_pairs*.csv` |
| Output-mode ablation | `experiments/results/dtlz6_union_ablation.csv` |
| Convergence curves | `experiments/results/history/convergence_history.csv` |
| Equal-wall-clock budget | `experiments/results/equal_budget.csv` |
| Module ablation / transplant | `module_ablation*.csv`, `transplant_chain.csv`, `transplant_summary.csv` |
| Phi-rho controller trace | `experiments/results/history/phi_rho_trace.csv` |

`m3_results.csv` is the authoritative external-comparison table. `m3_results_M3era.csv` is a
superseded earlier-phase run kept only for provenance; no script reads it.

### Table-generation coverage

Twelve of the manuscript's tables are written directly to LaTeX and regenerate byte-for-byte
from the shipped CSVs:

| Tables | Written by |
|---|---|
| T4–T11 | `experiments/make_paper_assets.py` — prints the win/tie/loss counts and the 32/32 chain agreement alongside their expected values so you can compare by eye; it reports, it does not fail |
| T20–T23 | `experiments/gen_t20.py` … `gen_t23.py` |

The remaining tables were typeset by hand from the console output of the analysis scripts.
Their data and computation are fully reproducible here — only the final LaTeX wrapping was
manual — and each script is included and runnable:

| Table | Analysis script |
|---|---|
| T12 (disjoint seeds) | `experiments/s2_disjoint.py` |
| T14 (held-out re-validation) | `experiments/gen_heldout.py` |
| T15 (profiling decomposition) | `experiments/cprofile_decompose.py` |
| T16 (union rescue) | `experiments/union_mode_adaptive.py` |
| T17 (plant gain) | `experiments/identify_plant_gain.py`, `estimate_plant_gain.py`, `summarize_plant_gain.py` |
| T18 (PD vs bandit) | `experiments/analyze_bandit.py` |
| T19 (criterion chain vs configurators) | `experiments/analyze_tuner.py`, `experiments/auto_config.py` |
| T_controller (controller parameters) | constants in `src/controller.py`, `src/algorithm.py` |

As a spot check, running `analyze_bandit.py` on the shipped CSV reproduces the exact figures
quoted in T18: `1/27/4` and `A12 mean 0.441, median 0.470`.

## Scope

The four mechanisms, the activation criterion chain, the seed protocol and the statistical
analysis (Wilcoxon signed-rank with Benjamini–Hochberg FDR correction, Vargha–Delaney effect
sizes) are all in this repository. Two things are deliberately not:

- **Manuscript assembly.** The scripts that build the Word/PDF manuscript from the LaTeX
  sources depend on a local Microsoft Office installation (`MML2OMML.XSL`) and a private
  presentation toolchain, so they are not portable and are not published here.
- **Baseline re-tuning.** As disclosed in the paper, the pymoo baselines run on library
  defaults rather than being re-tuned for high dimensions.

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper above. A machine-readable citation will be added
once the proceedings entry receives its DOI.
