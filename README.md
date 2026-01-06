# PP-HCFL+ Project

This repository contains our course project on privacy-preserving clustered federated learning (PP-HCFL+) built on OpenFGL. The focus is protecting cluster statistics (sample weights, membership weights, prototypes, and label counts) with differential privacy.

## Run Table 9 Experiments

```bash
# from the repo root
python run_table9.py
```

Notes:
- Update your data root in the script or set it via `args.root` in your config before running.
- The script will create/update files under `results/` (e.g., `results/table9.csv`).
- Runs can take a long time depending on GPU and dataset size.

## HCFL+ DP Flags (Project Defaults)

```python
args.hcfl_dp_stats = True
args.hcfl_proto_clip = 1.0
args.hcfl_proto_noise = 0.1
args.hcfl_count_clip = 10.0
args.hcfl_count_noise = 0.5

args.hcfl_dp_membership = True
args.hcfl_membership_clip = 1.0
args.hcfl_membership_noise = 0.1

# optional entropy regularization on client-wise cluster weights
args.hcfl_entropy_reg = 0.05

# optional DP on sample-wise responsibilities (omega_{i,j;k})
args.hcfl_dp_sample_weights = True
args.hcfl_sample_clip = 1.0
args.hcfl_sample_noise = 0.1

# optional DP accounting logs (approximate eps with basic composition)
args.hcfl_dp_accounting = True
```

## Threat Model (HCFL+ Privacy Extensions)
- Honest-but-curious server may inspect client statistics and clustering metadata.
- DP is applied to gradients (OpenFGL baseline) and HCFL+ clustering stats (prototypes, counts, membership, sample weights).
- Secure aggregation is provided as a placeholder hook for clustering stats; it is not a cryptographic implementation and does not prevent server access without additional protocol work.
