
![1301717130101_ pic](https://github.com/zyl24/OpenFGL/assets/59046279/e21b410f-2b5d-4515-8ab5-a176f98805a7)


# Open Federated Graph Learning (OpenFGL)
OpenFGL is a comprehensive, user-friendly algorithm library, complemented by an integrated evaluation platform, designed specifically for researchers in the field of federated graph learning (FGL).

<p align="center">
  <a href="https://arxiv.org/abs/2408.16288">Paper</a> •
  <a href="#Library Highlights">Highlights</a> •
  <a href="https://pypi.org/project/openfgl-lib/">Installation</a> •
  <a href="https://openfgl.readthedocs.io/en/latest/">Docs</a> •
  <a href="#Citation">Citation</a> 
</p>



[![Stars](https://img.shields.io/github/stars/zyl24/OpenFGL.svg?color=orange)](https://github.com/zyl24/OpenFGL/stargazers) ![](https://img.shields.io/github/last-commit/zyl24/OpenFGL) 
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992) -->

 



## Highlights

- 2 FGL Scenarios: Graph-FL and Subgraph-FL
- 10+ FGL Algorithms
- 34 FGL Datasets
- 12 GNN Models
- 5 Downstream Tasks
- Comprehensive FGL Data Property Analysis

## Get Started

```python
import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "your_data_root"


args.dataset = ["Cora"]
args.simulation_mode = "subgraph_fl_louvain"
args.num_clients = 10


if True:
    args.fl_algorithm = "fedavg"
    args.model = ["gcn"]
else:
    args.fl_algorithm = "fedproto"
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = ["accuracy"]



trainer = FGLTrainer(args)

trainer.train()
```

## HCFL+ DP Flags
Enable DP on HCFL+ clustering statistics (prototypes/counts) and membership weights:

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

## Project: Run Table 9 Experiments
Use the helper script in the repo root to reproduce the Table 9 runs and write results to `results/`.

```bash
# from the repo root
python run_table9.py
```

Notes:
- Update your data root in the script or set it via `args.root` in your config before running.
- The script will create/update files under `results/` (e.g., `results/table9.csv`).
- Runs can take a long time depending on GPU and dataset size.

## Threat Model (HCFL+ Privacy Extensions)
- Honest-but-curious server may inspect client statistics and clustering metadata.
- DP is applied to gradients (OpenFGL baseline) and HCFL+ clustering stats (prototypes, counts, membership, sample weights).
- Secure aggregation is provided as a placeholder hook for clustering stats; it is not a cryptographic implementation and does not prevent server access without additional protocol work.


## Citation
Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:
```
@misc{li2024openfglcomprehensivebenchmarksfederated,
      title={OpenFGL: A Comprehensive Benchmarks for Federated Graph Learning}, 
      author={Xunkai Li and Yinlin Zhu and Boyang Pang and Guochen Yan and Yeyu Yan and Zening Li and Zhengyu Wu and Wentao Zhang and Rong-Hua Li and Guoren Wang},
      year={2024},
      eprint={2408.16288},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.16288}, 
}
```
