# Communication-Efficient Federated Fine-Tuning of Medical Foundation Models under Cross-Specialty Heterogeneity

<!-- TODO(author): fill in names/affiliations. This is a simulation-based, software-only study;
     no real patient data is used anywhere (see Setup). -->

## Abstract

Federated fine-tuning lets multiple hospitals collaboratively adapt a shared medical imaging
backbone without pooling patient data, but cross-specialty heterogeneity -- where each hospital's
data distribution reflects a different medical specialty rather than a random partition of one
distribution -- breaks the implicit i.i.d. assumption behind naive aggregation schemes like
FedAvg. We study this setting with a fully simulated, CPU-only pipeline: five MedMNIST specialty
subsets (pathology, dermatology, blood, retina, kidney tissue) stand in for five hospitals, and
each client fine-tunes a frozen pretrained image backbone via LoRA adapters, communicating only
the low-rank adapter factors. We propose **FedSAA** (Similarity-Attended Aggregation), which
forms each client's merged LoRA delta $U_i = B_iA_i$, computes a softmax-attention aggregate over
the other clients' deltas weighted by cosine similarity, blends it with the uniform mean via a
$\lambda$ parameter, and re-factorizes the result to a personalized rank-$r$ adapter per client via
truncated SVD. FedSAA reduces exactly to FedAvg-on-merged-deltas at $\lambda=1$ (proved in
`tests/test_fedsaa.py`) and to pure similarity-attended aggregation at $\lambda=0$. On our
`local_debug` smoke configuration (3 clients, small subsampled data, few rounds -- see
Limitations), FedSAA improves global accuracy, macro-F1, AUC, and worst-client accuracy over
FedAvg and FedProx at identical communication cost.
<!-- TODO(author): replace/extend this abstract once full_run.yaml results (5 clients, 50 rounds,
     3 seeds) are in -- the current numbers are from the reduced debug pipeline, not the real run. -->

---

## 1. Introduction

<!-- TODO(author): motivate the clinical/practical framing in your own words -- e.g. why
     cross-specialty (rather than cross-institution-same-specialty) heterogeneity is
     under-studied, why parameter-efficient fine-tuning (LoRA) matters for hospital-side
     compute/communication budgets, and what gap FedSAA fills relative to FedAvg/FedProx. -->

Contributions (as implemented in this codebase):

1. A CPU-only, fully reproducible simulation harness (`src/data`, `src/models`, `src/federated`)
   that maps five MedMNIST specialty subsets to five simulated hospitals under a shared config
   schema (`configs/local_debug.yaml`, `configs/full_run.yaml`).
2. **FedSAA**, a similarity-attended LoRA aggregation rule (`src/federated/fedsaa.py`) that
   personalizes each client's adapter via softmax attention over cosine similarity of merged
   LoRA deltas, with a provable reduction to FedAvg at $\lambda=1$.
3. An evaluation suite (`src/metrics`) covering accuracy/macro-F1/AUC, communication accounting,
   and linear-CKA-based representation-collapse tracking, plus an ablation and table/figure
   generation pipeline (`experiments/`, `src/viz/`) producing the results in Section 5.

<!-- TODO(author): add 1-2 more paragraphs of narrative framing / related-work teaser here. -->

---

## 2. Related Work

<!-- TODO(author): this section is a structural skeleton with placeholder citation keys
     (\cite{...} in the .tex version) -- fill in actual citations and 1-2 sentences of
     comparison/positioning per subarea. -->

### 2.1 Federated Learning
FedAvg [TODO: cite McMahan et al. 2017] established sample-weighted parameter averaging as the
default aggregation rule; FedProx [TODO: cite Li et al. 2020] adds a proximal term to stabilize
local updates under statistical heterogeneity. Both are implemented here as baselines
(`src/federated/loop.py`, `client.py`) for direct, same-codebase comparison against FedSAA.

### 2.2 LoRA / Parameter-Efficient Fine-Tuning (PEFT)
LoRA [TODO: cite Hu et al. 2021] freezes a pretrained backbone and injects trainable low-rank
factors $(A, B)$ into target layers, drastically reducing trainable/communicated parameters --
in our setup, 148,309 trainable parameters (1.31% of the 11.3M-parameter ResNet-18 + head) per
client, and a 0.548 MB adapter payload (Section 5, Table T3). Naively averaging LoRA factors
$(A_i, B_i)$ directly (rather than their product $B_iA_i$) is known to be a biased approximation
of averaging the effective weight update; FedSAA's $\lambda=1$ case is explicitly defined at the
level of merged deltas $U_i = B_iA_i$ to sidestep this (Section 4, Section 4.3).

### 2.3 Federated Learning in Medical Imaging
<!-- TODO(author): cite medical FL surveys / cross-silo hospital FL work; discuss how those
     setups typically assume a shared task/label space, unlike our cross-specialty setting
     where each client has a DIFFERENT specialty and DIFFERENT #classes (Table T2). -->

### 2.4 Personalized Federated Learning
FedSAA produces a genuinely personalized adapter per client (unlike FedAvg/FedProx's single
global adapter) via attention over client similarity, related in spirit to clustered/personalized
FL methods. <!-- TODO(author): cite specific personalized-FL / clustered-FL baselines and
     position FedSAA's per-layer, per-round attention mechanism relative to them. -->

---

## 3. Method: FedSAA (Similarity-Attended Aggregation)

Let client $i \in \{1, \dots, n\}$ hold LoRA factors $(A_i, B_i)$ for a given target layer after
local training (Section 5.1: `conv1` and every Conv2d inside `layer1`-`layer4` of a frozen
ResNet-18). Per round, per layer, the server (`src/federated/fedsaa.py::fedsaa_aggregate`)
computes:

1. **Merged delta.** $U_i = B_i A_i$, flattened to a vector (implementation: reshape $A_i$ to
   $(r, \text{in\_flat})$, $B_i$ to $(\text{out\_ch}, r)$, so $U_i \in \mathbb{R}^{\text{out\_ch}
   \times \text{in\_flat}}$).
2. **Similarity.** $S_{ij} = \cos(U_i, U_j)$, an $n \times n$ cosine similarity matrix (saved
   every round for every layer -- Figure F4).
3. **Attention weights.** $w_{ij} = \mathrm{softmax}_j(S_{ij} / \tau)$, row-normalized.
4. **Attended aggregate + global mean.**
   $\bar{U}_i = \sum_j w_{ij} U_j$, $\quad G = \frac{1}{n}\sum_j U_j$.
5. **Blend.** $\tilde{U}_i = (1-\lambda)\,\bar{U}_i + \lambda\, G$.
6. **Re-factorization.** Truncated rank-$r$ SVD of $\tilde{U}_i$ gives back a personalized
   $(A_i, B_i)$ for client $i$: $B_{i,\text{new}} = U_{\text{svd}}[:, :r]\sqrt{S_{\text{svd}}[:r]}$,
   $A_{i,\text{new}} = \sqrt{S_{\text{svd}}[:r]}\, V_{\text{svd}}^\top[:r, :]$, reshaped back to
   conv-kernel shape.

Unlike FedAvg/FedProx, **the result is personalized**: every client gets back its own
$(A_i, B_i)$, not one broadcast global adapter.

### 4.1 Client-local training
Each client trains its LoRA adapter + local classifier head for `local_epochs` epochs of SGD
(`src/federated/client.py::local_train`); the head is never aggregated or communicated (Setup,
Table T2), since specialties have different numbers of classes.

### 4.2 Communication accounting
Only the LoRA $(A, B)$ factors are exchanged (`get_adapter_state`/`set_adapter_state`,
`src/models/lora_model.py`) -- 0.548 MB per client per round at rank $r=4$ (Table T3), independent
of the frozen backbone's 11.3M parameters.

### 4.3 FedAvg as a special case
At $\lambda=1$, $\tilde{U}_i = G$ for **every** client, independent of $\tau$ or $w_{ij}$ -- i.e.
FedSAA collapses to the rank-$r$ truncated SVD of the plain mean of merged deltas
$\frac{1}{n}\sum_j B_jA_j$. This is exactly "FedAvg on merged LoRA deltas" (as opposed to naively
averaging $A$ and $B$ factors separately, Section 2.2), and is verified to floating-point
precision across multiple $\tau$ values in `tests/test_fedsaa.py::test_lambda1_matches_fedavg_on_merged_deltas`.
At $\lambda=0$, clients receive purely self-attended, personalized aggregates that are (by
construction, `test_lambda0_is_pure_self_attention_and_differs_across_clients`) generally distinct
across clients.

<!-- TODO(author): add a short paragraph interpreting *why* similarity-attention should help
     under cross-specialty heterogeneity -- e.g. clients with more-similar specialties (by
     merged-delta cosine similarity) reinforce each other's updates while dissimilar clients
     contribute less, which uniform FedAvg cannot express. -->

---

## 4. Setup

### 5.1 Datasets (cross-specialty non-IID simulation)
Five MedMNIST subsets, each standing in for one simulated hospital (client $i$ gets
`data.specialties[i]`): PathMNIST (pathology), DermaMNIST (dermatology), BloodMNIST (blood),
OCTMNIST (retina), TissueMNIST (kidney tissue) -- verified accessible in `data_access_check.py`
(Step 1). **100% simulation; no real patient data is used anywhere in this pipeline.** Images are
converted grayscale->3-channel RGB and loaded at 28x28 (`data.image_size`); a per-client
subsample cap keeps peak RAM low on an 8 GB laptop (`src/data/specialty_dataset.py`, mmap-loaded
so full arrays are never materialized).

### 5.2 Backbone and adapters
Frozen pretrained **ResNet-18** (`model.backbone`, `model.pretrained=True`,
`model.freeze_backbone=True`); LoRA attached to `conv1` and every Conv2d inside
`layer1`-`layer4` (`model.lora.target_modules`), rank $r=4$, $\alpha=8$, dropout$=0.05$
(`model.lora`). Each client keeps its own classifier head, never aggregated.

### 5.3 Federated protocol
`federated.num_clients=3`, `federated.rounds=5`, `federated.local_epochs=1`,
`federated.client_fraction=1.0`; local optimizer SGD, lr$=0.01$, momentum$=0.9$,
weight\_decay$=10^{-4}$ (`federated.*`). FedSAA hyperparameters: $\tau=0.1$, $\lambda=0.5$,
`svd_rank=4` (`method.fedsaa.*`); FedProx $\mu=0.01$ (`method.fedprox.mu`).

**Full hyperparameter dump:** Table T2 (`paper/tables/T2_hyperparameters.tex`), generated
directly from the active config (`configs/local_debug.yaml` for the numbers in this draft;
`configs/full_run.yaml` uses 5 clients / 50 rounds / 3 seeds / no subsample cap and is the
intended real experiment -- see Limitations).

### 5.4 Baselines
Centralized (upper bound: one shared LoRA adapter trained jointly on pooled data,
`src/federated/centralized.py`), Local-only (lower bound: no communication ever,
`src/federated/local_only.py`), FedAvg and FedProx (`src/federated/loop.py`), all logging the
identical `metrics.csv` schema as FedSAA for direct comparison.

---

## 5. Results

<!-- TODO(author): the numbers below come from configs/local_debug.yaml (3 clients, <=3 rounds,
     1 seed, 2000-image subsample cap) -- a pipeline smoke-test config, NOT the intended
     full-scale experiment (configs/full_run.yaml: 5 clients, 50 rounds, 3 seeds, no cap). Treat
     every number in this section as "the pipeline works end-to-end and produces sane output",
     not as a claim about FedSAA's true effectiveness. Re-run
     `python -m experiments.runner --config configs/full_run.yaml` and regenerate figures/tables
     before drawing any real conclusions. -->

### 6.1 Main results (Table T1, Figures F1-F3)

| Method | Global Acc. | Macro-F1 | AUC | Worst-client Acc. | Total Comm. (MB) |
|---|---|---|---|---|---|
| Centralized | 0.319 | 0.058 | 0.499 | 0.090 | **0.00** |
| Local-only | 0.567 | 0.375 | 0.868 | 0.448 | **0.00** |
| FedAvg | 0.451 | 0.210 | 0.776 | 0.285 | 9.87 |
| FedProx | 0.469 | 0.226 | 0.768 | 0.264 | 9.87 |
| **FedSAA (ours)** | **0.685** | **0.443** | **0.878** | **0.678** | 9.87 |

(Full table with std-dev columns: `paper/tables/T1_main_results.tex`/`.csv`. Std is 0 here since
this draft uses `seeds=[0]` only -- Table T1 is written to support multi-seed mean$\pm$std once
`full_run.yaml`'s 3 seeds are run.)

**Figure F1** (`paper/figures/F1_convergence.png`/`.pdf`) shows global accuracy vs. round for all
five methods; FedSAA is the only method that improves monotonically after round 2 in this run.
**Figure F2** (`F2_accuracy_vs_comm.*`) plots accuracy against cumulative communication --
Centralized/Local-only appear as horizontal reference lines since they never communicate over a
network. **Figure F3** (`F3_per_specialty_bars.*`) breaks accuracy out per specialty; FedSAA's
worst-client accuracy (0.678) is far above FedAvg's (0.285) and FedProx's (0.264), suggesting the
similarity-attention mechanism is not simply averaging away the weakest client.

<!-- TODO(author): interpret the Centralized baseline's poor performance (0.319 acc, close to a
     macro_f1 collapse toward 0.058) -- this reflects an optimization instability (single shared
     LoRA adapter, unscheduled lr=0.01 SGD on pooled heterogeneous batches) flagged during Step 6,
     not a fundamental ceiling; revisit with a tuned/decayed LR before treating Centralized as a
     true upper bound. -->

### 6.2 Efficiency (Table T3)

| Method | Trainable Params | MB/Round | Rounds to 90% of own best | Total Comm. (MB) |
|---|---|---|---|---|
| Centralized | 148,309 | 0.000 | 1 | 0.00 |
| Local-only | 148,309 | 0.000 | 1 | 0.00 |
| FedAvg | 148,309 | 3.289 | 1 | 9.87 |
| FedProx | 148,309 | 3.289 | 1 | 9.87 |
| FedSAA | 148,309 | 3.289 | 3 | 9.87 |

All methods share the same 148,309-parameter (1.31% of 11,324,821 total) trainable footprint per
client model, since only `model.lora.r` and the backbone differ trainable-param count, and all
methods here use the same config. FedSAA and the baselines communicate identically (3.289
MB/round for 3 clients at rank 4) -- FedSAA's accuracy gains in Table T1 come at **no additional
communication cost**.
<!-- TODO(author): "rounds to 90% of own best" is relative to each method's OWN final accuracy,
     not a shared absolute target -- useful for convergence speed, not for cross-method accuracy
     comparison (that's Table T1). Clarify/replace this metric if you want an absolute-target
     version (src/metrics/communication.py::rounds_to_target_accuracy supports both). -->

### 6.3 Ablations (Table T4, Figures F5-F7)

**Lambda ($\lambda$)**, fixed $\tau=0.1$, $r=4$, $C=1.0$:

| $\lambda$ | Global Acc. | Macro-F1 | AUC |
|---|---|---|---|
| 0 | 0.619 | 0.415 | 0.834 |
| 0.5 | 0.605 | 0.404 | 0.838 |
| 1 (= FedAvg-on-deltas) | 0.581 | 0.364 | 0.815 |

Accuracy decreases monotonically as $\lambda \to 1$ in this run (Figure F5), i.e. pure
similarity-attention ($\lambda=0$) outperforms both the blend and the FedAvg-equivalent extreme.

**Rank ($r$)**, fixed $\lambda=0.5$, $\tau=0.1$, $C=1.0$:

| $r$ | Global Acc. | Macro-F1 | AUC | Cumulative MB |
|---|---|---|---|---|
| 2 | 0.319 | 0.126 | 0.766 | 3.29 |
| 4 | 0.605 | 0.404 | 0.838 | 3.29 |
| 8 | **0.639** | **0.458** | **0.881** | 13.16 |
| 16 | 0.605 | 0.437 | 0.859 | 26.31 |

Accuracy rises sharply from $r=2$ to $r=8$ then plateaus/dips slightly at $r=16$ (Figure F6),
while communication cost grows linearly with $r$ -- $r=8$ looks like the best accuracy/cost
tradeoff point in this run, not $r=16$.

**Figure F4** (`F4_similarity_heatmap.*`) shows the client similarity matrix $S$ (mean over 20
LoRA layers, final round): off-diagonal cosine similarities of 0.29-0.51 between the three
clients, i.e. clients' merged deltas are meaningfully correlated but far from identical --
exactly the regime where attention-weighted aggregation (rather than uniform FedAvg) should have
room to help.

**Figure F7** (`F7_collapse_cka.*`) tracks mean pairwise linear CKA between clients' backbone
features on a fixed probe batch, per round, for FedAvg vs. FedSAA -- a representation-collapse
indicator.
<!-- TODO(author): report the actual F7 trend once full_run.yaml (50 rounds) gives enough rounds
     to see a real trajectory; 1-3 rounds is too few to characterize collapse behavior. Insert
     your own reading of whether FedSAA's personalization measurably slows collapse vs FedAvg. -->

<!-- TODO(author): with only 2-3 points per ablation axis and 1 seed, none of Section 6.3's
     trends are statistically meaningful yet -- they demonstrate the ablation *pipeline* works
     (experiments/runner.py's FULL_ABLATION_GRID: lambda in {0,0.25,0.5,0.75,1}, tau in
     {0.1,0.5,1}, rank in {2,4,8,16}, client_fraction in {0.5,1.0} = 120 configs) but the full
     grid x 3 seeds under full_run.yaml has not been run. Do that before drawing conclusions. -->

---

## 6. Limitations

- **All numbers in this draft come from `configs/local_debug.yaml`**: 3 clients (not 5), a
  2000-image-per-client subsample cap, and at most 3 rounds with a single seed. This was
  deliberately kept tiny to prove the pipeline end-to-end on an 8 GB CPU-only laptop (no CUDA
  GPU); it is not the intended experiment. `configs/full_run.yaml` (5 clients, 50 rounds, 3
  seeds, no/relaxed subsample cap) is the real configuration and has not yet been run.
- **Single seed.** Every std-dev in Table T1/T3/T4 is 0.0 because `local_debug.yaml` uses
  `seeds=[0]` only -- no variance estimate exists yet.
- **Centralized baseline instability.** Flagged in Step 6: the shared-LoRA centralized baseline
  shows optimization collapse (macro-F1 -> ~0.06) under this config's unscheduled SGD lr=0.01 on
  pooled heterogeneous batches. This is a hyperparameter/optimization issue with the baseline,
  not a property of the data or of FedSAA, but it means Table T1's "Centralized" row should not
  yet be read as a true upper bound.
- **Ablation grid coverage.** Table T4/Figures F5-F6 currently cover 3 $\lambda$ points and 4
  rank points at a single $(\tau, C)$ setting each -- far short of the full
  $5\times3\times4\times2=120$-config grid `experiments/runner.py` supports.
- **CPU-only, small backbone.** ResNet-18 at 28x28 was chosen for laptop feasibility
  (`runtime_estimate.py`); results may not transfer to larger medical backbones/resolutions.
- **MedMNIST as a specialty-heterogeneity proxy.** Using different MedMNIST subsets as "different
  hospitals" simulates cross-specialty non-IID-ness via label-space and image-domain shift, but
  does not capture other real-world heterogeneity sources (device/scanner variation, class
  imbalance within a real hospital, demographic shift).
<!-- TODO(author): add any additional limitations specific to your own reading of the results
     (e.g. AUC computation caveats for small per-class batches, evaluate_global's per-sample head
     routing assumption, etc.). -->

---

## 7. Conclusion

<!-- TODO(author): write the actual conclusion once full_run.yaml results are available. Draft
     talking points based on the current (smoke-scale) evidence:
     - FedSAA's pipeline (Steps 1-11) runs end-to-end on CPU and produces personalized,
       communication-matched LoRA adapters via similarity-attended aggregation.
     - lambda=1 is verified to exactly reduce to FedAvg-on-merged-deltas (unit-tested), giving
       a principled way to interpolate between personalization and global averaging.
     - Early (small-scale) signal favors lower lambda and rank ~8 over the tested range, and
       shows meaningfully higher worst-client accuracy than FedAvg/FedProx at equal communication
       cost -- but this needs confirmation at full_run.yaml scale with multiple seeds before any
       claim beyond "the pipeline and method are implemented correctly." -->

---

## Appendix: Reproducibility

- Code layout, config schema, and CLI: see `README.md`.
- To reproduce this draft's numbers: `python -m experiments.runner --config
  configs/local_debug.yaml --reduced --rounds-override 3`, then
  `python -m src.viz.generate_figures --config configs/local_debug.yaml` and
  `python -m src.viz.generate_tables --config configs/local_debug.yaml`.
- To run the real experiment: `python -m experiments.runner --config configs/full_run.yaml`
  (see `experiments/runtime_estimate.py` for a per-config CPU runtime warning before committing
  to a run).
