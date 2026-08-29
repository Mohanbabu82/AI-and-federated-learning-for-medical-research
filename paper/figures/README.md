# Figures

Generated from `configs/local_debug.yaml` (seed=0). Each figure is exported as both `.pdf` (vector, for the paper) and `.png` (300 dpi, for slides/previews). Colors follow the colorblind-safe Okabe-Ito palette, consistent across all figures.

## F1_convergence

**F1 -- Convergence.** Global test accuracy vs. federated round for each method. FedSAA's per-client similarity-attended aggregation is compared against the Centralized upper bound, Local-only lower bound, and the FedAvg/FedProx baselines.

## F2_accuracy_vs_comm

**F2 -- Accuracy vs. communication cost.** Global test accuracy plotted against cumulative communication (MB of LoRA adapter payload exchanged). Centralized and Local-only never communicate over a network, so they are shown as horizontal reference lines at their final accuracy rather than a curve.

## F3_per_specialty_bars

**F3 -- Per-specialty accuracy.** Final-round test accuracy broken out by specialty (simulated hospital), one bar group per specialty, one bar per method. Highlights which methods under-serve which specialties under cross-specialty heterogeneity.

## F4_similarity_heatmap

**F4 -- FedSAA client similarity matrix S.** Cosine similarity between clients' merged LoRA deltas U_i = B_i @ A_i (mean across LoRA layers, from the final logged round). This is the raw signal the softmax attention (Step 5 of FedSAA) acts on.

## F5_lambda_ablation

**F5 -- FedSAA lambda ablation.** Global test accuracy (mean +/- std across seeds) as lambda sweeps from 0 (pure similarity-attended aggregation) to 1 (equivalent to FedAvg on merged LoRA deltas -- see tests/test_fedsaa.py), at fixed tau/rank/client_fraction.

## F6_rank_ablation

**F6 -- FedSAA rank ablation.** Global test accuracy (left axis) and cumulative communication cost (right axis) vs. the LoRA rank r used for both local adapters and the server's truncated-SVD re-factorization, at fixed lambda/tau/client_fraction. Shows the accuracy/communication trade-off of the rank choice.

## F7_collapse_cka

**F7 -- Representation collapse (CKA).** Mean pairwise linear CKA between clients' backbone features on a small fixed probe batch, per round. Higher values indicate clients' representations have homogenized (collapsed) toward each other; FedSAA is compared against FedAvg to see whether attention-based aggregation preserves more cross-specialty representational diversity.

