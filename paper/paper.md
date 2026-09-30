# Communication-Efficient Federated Fine-Tuning of Medical Foundation Models under Cross-Specialty Heterogeneity

<!-- STATUS: this draft covers Abstract/Introduction/Related Work/Method/Setup only. Results,
     Ablations, Limitations, and Conclusion are intentionally left as [TODO: results] -- no
     run of configs/full_run.yaml has completed at time of writing (see README.md /
     experiments/estimate.py for the current runtime estimate). Do not fill in numbers by hand;
     regenerate via python -m experiments.runner --config configs/full_run.yaml --dataset A
     (and --dataset B), then src/viz/generate_figures.py / generate_tables.py, once real
     full_run_* results exist and pass experiments/validation.py::assert_full_run. -->

<!-- TODO(author): fill in names/affiliations. This is a simulation-based, software-only study;
     no real patient data is used anywhere (see Setup). -->

## Abstract

Federated fine-tuning lets multiple hospitals collaboratively adapt a shared medical imaging
backbone without pooling patient data, but cross-specialty heterogeneity -- where each hospital's
data distribution reflects a different medical specialty rather than a random partition of one
distribution -- breaks the implicit i.i.d. assumption behind naive aggregation schemes like
FedAvg. We study this setting with a fully simulated, CPU-compatible pipeline across **two**
independent specialty mixes drawn from MedMNIST: Dataset A (pathology, dermatology, blood,
retina, kidney tissue) and Dataset B (abdominal organ, chest pneumonia, retina fundus, breast
ultrasound, abdominal organ-C), each mapped to five simulated hospitals. Each client fine-tunes a
frozen pretrained image backbone via LoRA adapters, communicating only the low-rank adapter
factors. We propose **FedSAA** (Similarity-Attended Aggregation), which forms each client's
merged LoRA delta $U_i = B_iA_i$, computes a softmax-attention aggregate over the other clients'
deltas weighted by cosine similarity, blends it with the uniform mean via a $\lambda$ parameter,
re-factorizes the result to a personalized rank-$r$ adapter per client via truncated SVD, and
then blends that server-computed adapter back toward the client's own pre-aggregation adapter via
a retention coefficient $\alpha$, followed by a brief additional local fine-tuning pass. FedSAA
reduces exactly to FedAvg-on-merged-deltas at $\lambda=1$ (proved in `tests/test_fedsaa.py`).
[TODO: results -- global accuracy, macro-F1, AUC, worst-client accuracy, and communication cost
for FedSAA vs. Centralized/Local-only/FedAvg/FedProx on both datasets, with statistical
significance across 5 seeds, once `configs/full_run.yaml` has been run to completion.]
[TODO: results -- ablation findings for $\lambda$, $\tau$, rank $r$, and retention $\alpha$.]

---

## 1. Introduction

<!-- TODO(author): motivate the clinical/practical framing in your own words -- e.g. why
     cross-specialty (rather than cross-institution-same-specialty) heterogeneity is
     under-studied, why parameter-efficient fine-tuning (LoRA) matters for hospital-side
     compute/communication budgets, and what gap FedSAA fills relative to FedAvg/FedProx. -->

Contributions (as implemented in this codebase):

1. A CPU-compatible, fully reproducible simulation harness (`src/data`, `src/models`,
   `src/federated`) that maps MedMNIST specialty subsets to simulated hospitals under a shared
   config schema, with **two independently selectable specialty mixes** (Dataset A / Dataset B,
   `configs/full_run.yaml`'s `data.dataset_presets`) so results can be checked for consistency
   across two different cross-specialty non-IID partitions rather than one.
2. **FedSAA**, a similarity-attended LoRA aggregation rule (`src/federated/fedsaa.py`) that
   personalizes each client's adapter via softmax attention over cosine similarity of merged
   LoRA deltas, with a provable reduction to FedAvg at $\lambda=1$, plus a **local adapter
   retention** mechanism (retention coefficient $\alpha$) and a brief **post-aggregation local
   adaptation** step so personalization happens on the client side too, not only at the server.
3. An evaluation suite (`src/metrics`) covering accuracy/macro-F1/AUC, communication accounting,
   and linear-CKA-based representation-collapse tracking, plus an ablation and table/figure
   generation pipeline (`experiments/`, `src/viz/`) that refuses (loudly) to summarize anything
   but complete, full-scale runs (`experiments/validation.py`).

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
factors $(A, B)$ into target layers, drastically reducing trainable/communicated parameters
relative to full fine-tuning. Naively averaging LoRA factors $(A_i, B_i)$ directly (rather than
their product $B_iA_i$) is known to be a biased approximation of averaging the effective weight
update; FedSAA's $\lambda=1$ case is explicitly defined at the level of merged deltas
$U_i = B_iA_i$ to sidestep this (Section 3.3).

### 2.3 Federated Learning in Medical Imaging
<!-- TODO(author): cite medical FL surveys / cross-silo hospital FL work; discuss how those
     setups typically assume a shared task/label space, unlike our cross-specialty setting
     where each client has a DIFFERENT specialty and a DIFFERENT number of classes (Section 4). -->

### 2.4 Personalized Federated Learning
FedSAA produces a genuinely personalized adapter per client (unlike FedAvg/FedProx's single
global adapter) via attention over client similarity, further personalized by the local
retention blend and post-aggregation local adaptation described in Section 3.4-3.5 -- related in
spirit to clustered/personalized FL methods. <!-- TODO(author): cite specific personalized-FL /
clustered-FL baselines and position FedSAA's mechanism relative to them. -->

---

## 3. Method: FedSAA (Similarity-Attended Aggregation)

Let client $i \in \{1, \dots, n\}$ hold LoRA factors $(A_i, B_i)$ for a given target layer after
local training (Section 4.2: `conv1` and every Conv2d inside `layer1`-`layer4` of a frozen
ResNet-18). Per round, per layer, the server (`src/federated/fedsaa.py::fedsaa_aggregate`)
computes:

1. **Merged delta.** $U_i = B_i A_i$, flattened to a vector (implementation: reshape $A_i$ to
   $(r, \text{in\_flat})$, $B_i$ to $(\text{out\_ch}, r)$, so $U_i \in \mathbb{R}^{\text{out\_ch}
   \times \text{in\_flat}}$).
2. **Similarity.** $S_{ij} = \cos(U_i, U_j)$, an $n \times n$ cosine similarity matrix (saved
   every round for every layer -- figure F4).
3. **Attention weights.** $w_{ij} = \mathrm{softmax}_j(S_{ij} / \tau)$, row-normalized.
4. **Attended aggregate + global mean.**
   $\bar{U}_i = \sum_j w_{ij} U_j$, $\quad G = \frac{1}{n}\sum_j U_j$.
5. **Blend.** $\tilde{U}_i = (1-\lambda)\,\bar{U}_i + \lambda\, G$.
6. **Re-factorization.** Truncated rank-$r$ SVD of $\tilde{U}_i$ gives a server-computed
   personalized $(A_i, B_i)_{\text{global}}$ for client $i$:
   $B_{\text{global}} = U_{\text{svd}}[:, :r]\sqrt{S_{\text{svd}}[:r]}$,
   $A_{\text{global}} = \sqrt{S_{\text{svd}}[:r]}\, V_{\text{svd}}^\top[:r, :]$, reshaped back to
   conv-kernel shape.

Unlike FedAvg, the result at this point is already **personalized**: every client would get back
its own $(A_i, B_i)$, not one broadcast global adapter.

### 3.1 Client-local training
Each client trains its LoRA adapter + local classifier head for `local_epochs` epochs of SGD
(`src/federated/client.py::local_train`); the head is never aggregated or communicated (Section
4, Table T2), since specialties have different numbers of classes.

### 3.2 Communication accounting
Only the LoRA $(A, B)$ factors are exchanged (`get_adapter_state`/`set_adapter_state`,
`src/models/lora_model.py`), independent of the frozen backbone's parameters, which never move.

### 3.3 FedAvg as a special case
At $\lambda=1$, $\tilde{U}_i = G$ for **every** client, independent of $\tau$ or $w_{ij}$ -- i.e.
FedSAA's steps 1-6 collapse to the rank-$r$ truncated SVD of the plain mean of merged deltas
$\frac{1}{n}\sum_j B_jA_j$. This is exactly "FedAvg on merged LoRA deltas" (as opposed to naively
averaging $A$ and $B$ factors separately, Section 2.2), and is verified to floating-point
precision across multiple $\tau$ values in `tests/test_fedsaa.py::test_lambda1_matches_fedavg_on_merged_deltas`.
At $\lambda=0$, clients receive purely self-attended, personalized aggregates that are (by
construction, `test_lambda0_is_pure_self_attention_and_differs_across_clients`) generally distinct
across clients.

### 3.4 Local adapter retention
The server-computed personalized adapter $(A_i, B_i)_{\text{global}}$ from step 6 is then blended
with client $i$'s **own pre-aggregation local adapter** -- the one it sent into this round's
aggregation, before steps 1-6 touched it, denoted $(A_i, B_i)_{\text{local}}$ -- via a retention
coefficient $\alpha \in [0, 1]$:

$$(A_i, B_i)_{\text{final}} = (1 - \alpha)\,(A_i, B_i)_{\text{global}} + \alpha\,(A_i, B_i)_{\text{local}}$$

applied per LoRA tensor ($A$ and $B$ separately, same shapes, elementwise). $\alpha = 0$ is pure
FedSAA (Sections 3-3.3, no retention); $\alpha \to 1$ increasingly ignores the server's
aggregation and keeps the client's own just-trained adapter. Implemented in
`src/federated/fedsaa.py::fedsaa_aggregate` (the `alpha` parameter).

### 3.5 Post-aggregation local adaptation
After retention-blending, each participating client runs a brief additional local fine-tuning
pass -- `post_agg_adapt_steps` SGD mini-batches (not a full epoch) on its own data
(`src/federated/client.py::local_adapt_steps`) -- before evaluation. Without this step,
"personalization" would only ever happen at the server; the client's adapter would never actually
see its own data again after being overwritten by the retention-blended result.

<!-- TODO(author): add a paragraph interpreting *why* similarity-attention plus local retention
     should help under cross-specialty heterogeneity -- e.g. clients with more-similar
     specialties (by merged-delta cosine similarity) reinforce each other's updates while
     dissimilar clients contribute less, and retention + post-aggregation adaptation prevents
     the server-side aggregation step from ever fully overwriting a client's own signal. -->

---

## 4. Experimental Setup

<!-- Source of truth for every value below: configs/full_run.yaml. Do not hand-edit numbers
     here without also updating that file -- Table T2 (once generated) is produced directly
     from it. -->

### 4.1 Datasets (two independent cross-specialty non-IID partitions)
All experiments are simulation-based; **no real patient data is used anywhere in this
pipeline.** Two independently selectable specialty mixes are drawn from MedMNIST, each mapping
five specialty subsets to five simulated hospitals (`data.dataset_presets` in
`configs/full_run.yaml`):

| | Dataset A | Dataset B |
|---|---|---|
| Client 0 | PathMNIST (pathology, 9 classes) | OrganAMNIST (abdominal organ, 11 classes) |
| Client 1 | DermaMNIST (dermatology, 7 classes) | PneumoniaMNIST (chest pneumonia, 2 classes) |
| Client 2 | BloodMNIST (blood, 8 classes) | RetinaMNIST (retina fundus, 5 classes) |
| Client 3 | OCTMNIST (retina OCT, 4 classes) | BreastMNIST (breast ultrasound, 2 classes) |
| Client 4 | TissueMNIST (kidney tissue, 8 classes) | OrganCMNIST (abdominal organ, 11 classes) |

Both mixes draw exclusively from the MedMNIST v2 suite (no new external data source). Images are
converted grayscale$\to$3-channel RGB and loaded at 28$\times$28 (`data.image_size`); no
per-client subsample cap is applied for the full run (`data.subsample_cap: null`), unlike the
`configs/local_debug.yaml` smoke-test profile which caps at 2000 images/client.

### 4.2 Backbone and adapters
Frozen pretrained **ResNet-18** (`model.backbone`, `model.pretrained: true`,
`model.freeze_backbone: true`); LoRA attached to `conv1` and every Conv2d inside
`layer1`-`layer4` (`model.lora.target_modules`), rank $r=8$, $\alpha_{\text{LoRA}}=8$,
dropout$=0.05$ (`model.lora`; note this $\alpha_{\text{LoRA}}$ is the standard LoRA scaling
factor, unrelated to FedSAA's retention $\alpha$ in Section 3.4). Each client keeps its own
classifier head, never aggregated.

### 4.3 Federated protocol
`federated.num_clients=5`, `federated.rounds=60`, `federated.local_epochs=2`,
`federated.client_fraction=1.0` (full participation every round); local optimizer SGD,
lr$=0.01$, momentum$=0.9$, weight\_decay$=10^{-4}$ (`federated.*`).

### 4.4 FedSAA hyperparameters
$\tau=0.5$, $\lambda=0.5$, `svd_rank=8` (`method.fedsaa.tau/lambda/svd_rank`); local adapter
retention $\alpha=0.3$ (`method.fedsaa.alpha`, Section 3.4); post-aggregation local adaptation
`post_agg_adapt_steps=5` (Section 3.5). FedProx baseline: $\mu=0.01$
(`method.fedprox.mu`).

### 4.5 Seeds and reproducibility
5 seeds (`seeds: [0, 1, 2, 3, 4]`) for every method on every dataset, deterministic seeding
(`reproducibility.deterministic: true`) via `src/utils/seed.py`.

### 4.6 FedSAA ablation grid
One-at-a-time sweeps around the Section 4.4 defaults (`experiments/runner.py`):

| Axis | Values | Seeds/config |
|---|---|---|
| $\lambda$ | $\{0, 0.25, 0.5, 0.75, 1\}$ | 1 |
| $\tau$ | $\{0.1, 0.5, 1\}$ | 1 |
| rank $r$ | $\{2, 4, 8, 16\}$ | 1 |
| retention $\alpha$ | $\{0, 0.1, 0.3, 0.5\}$ | 3 |

The retention axis runs more seeds than the others because it directly affects the paper's
personalization claim (Sections 3.4-3.5). The grid is one-at-a-time (each axis varied
independently around the shared default), not a full product, to keep the sweep tractable on
commodity hardware (`experiments/estimate.py` gives a per-config runtime estimate).

### 4.7 Baselines
Centralized (upper bound: one shared LoRA adapter trained jointly on pooled data,
`src/federated/centralized.py`), Local-only (lower bound: no communication ever,
`src/federated/local_only.py`), FedAvg and FedProx (`src/federated/loop.py`), all logging the
identical `metrics.csv` schema as FedSAA for direct comparison, on both Dataset A and Dataset B.

### 4.8 Hardware and environment
Developed and smoke-tested on a CPU-only laptop (no CUDA GPU); `configs/full_run.yaml`'s
`device: cpu` reflects that -- flip to `cuda` locally to run on a GPU machine (see
`experiments/runtime_estimate.py` for the CPU-calibrated per-batch cost, and
`experiments/estimate.py` for a GPU estimate derived from an explicitly-flagged, unverified
speedup heuristic pending real GPU calibration).

---

## 5. Results

[TODO: results -- this section, all of Section 6 (Ablations), Section 7 (Limitations), and
Section 8 (Conclusion) require a completed run of `configs/full_run.yaml` (both datasets, 5
seeds, 60 rounds, full ablation grid). No such run has completed at time of writing. Do not
fill in numbers here by hand. Once `experiments/runner.py` has produced `full_run_*` results
that pass `experiments/validation.py::assert_full_run`, regenerate:
  - Tables T1 (main results), T3 (efficiency), T4 (ablations) via `src/viz/generate_tables.py`
  - Figures F1-F7 via `src/viz/generate_figures.py`
  - Statistical significance (paired tests across the 5 seeds, FedSAA vs. each baseline, on
    global accuracy and macro-F1, for both datasets)
and write this section from those outputs.]

### 5.1 Main results
[TODO: results -- Table T1, Figures F1-F3, both datasets.]

### 5.2 Efficiency
[TODO: results -- Table T3.]

### 5.3 Ablations
[TODO: results -- Table T4, Figures F4-F7, including the retention $\alpha$ ablation from
Section 4.6.]

---

## 6. Limitations

[TODO: results -- write from the actual completed-run outcome. Known structural limitations to
include regardless of results (do not remove these when filling in the section):]

- **CPU-only development.** ResNet-18 at 28$\times$28 was chosen for feasibility on a CPU-only
  laptop; results may not transfer to larger medical backbones/resolutions.
- **MedMNIST as a specialty-heterogeneity proxy.** Using different MedMNIST subsets as "different
  hospitals" simulates cross-specialty non-IID-ness via label-space and image-domain shift, but
  does not capture other real-world heterogeneity sources (device/scanner variation, class
  imbalance within a real hospital, demographic shift). Testing on two independent specialty
  mixes (Dataset A / B) partially addresses generalization of any finding across partitions, but
  both are still MedMNIST-derived.
- **GPU estimate is an unverified heuristic** (Section 4.8) -- not measured on real GPU hardware.

---

## 7. Conclusion

[TODO: results -- write once Section 5 exists. Do not draft talking points from unrun
experiments.]

---

## Appendix: Reproducibility

- Code layout, config schema, and CLI: see `README.md`.
- To reproduce this draft's Setup section exactly: `configs/full_run.yaml` as committed.
- To run the real experiment: `python -m experiments.runner --config configs/full_run.yaml
  --dataset A --resume` and `python -m experiments.runner --config configs/full_run.yaml
  --dataset B --resume` (see `experiments/estimate.py` for a runtime estimate before
  committing to a run).
