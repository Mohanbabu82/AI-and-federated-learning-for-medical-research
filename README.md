# Communication-Efficient Federated Fine-Tuning of Medical Foundation Models under Cross-Specialty Heterogeneity

100% software, simulation-based research project. No real patient data. MedMNIST specialty
subsets stand in for hospitals (cross-specialty non-IID). LoRA adapters are attached to a
frozen pretrained image backbone; only the adapters are trained and communicated between
clients and server.

Novel algorithm: **FedSAA** (Similarity-Attended Aggregation), with FedAvg as the
special case (lambda=1, uniform weights). Baselines: Centralized, Local-only, FedAvg, FedProx.

## Hardware constraint

Runs entirely on CPU. Developed on an Intel i5-1235U laptop, 8 GB RAM, no CUDA GPU
(Intel UHD integrated graphics only). All code must work without CUDA and stay memory-frugal.

## Layout

```
configs/            local_debug.yaml, full_run.yaml -- one shared schema, two profiles
src/
  data/              MedMNIST loading, per-client (per-specialty) dataset construction
  models/            frozen backbone + LoRA adapter wiring
  federated/         client/server loops, FedSAA / FedAvg / FedProx / Local-only / Centralized
  metrics/           evaluation metrics, communication cost accounting
  viz/               plotting utilities for results
  utils/             seed.py (determinism), config.py (load + validate YAML)
experiments/         experiment driver scripts
results/             run outputs (metrics, logs, checkpoints if enabled)
paper/               writeup materials
run.py               CLI entrypoint: python run.py --config <cfg> --method <name>
data_access_check.py Step 1 standalone script confirming MedMNIST access on this laptop
```

## Setup

```bash
pip install -r requirements.txt
```

(CPU-only PyTorch is pinned via the `--index-url` at the top of requirements.txt.)

## Running

Fast sanity/debug profile (3 clients, 5 rounds, ~2000 images/client, seed 0 -- should
finish quickly on this laptop):

```bash
python run.py --config configs/local_debug.yaml --method fedsaa
```

Full experiment (5 clients, ~50 rounds, 3 seeds, no/relaxed subsample cap -- run later,
same code, heavier settings):

```bash
python run.py --config configs/full_run.yaml --method fedsaa
```

`--method` selects among `fedsaa | fedavg | fedprox | local_only | centralized` and
overrides `method.name` in the config. Both config files share one schema (see
`src/utils/config.py` for validation rules) so the same code path drives debug and full runs.

## Status

Step 1: data access verified. Step 2: project scaffold (this step) -- no training logic yet.
