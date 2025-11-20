import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# You can check the other datasets from config file
dataset = "Cora"

# All of the FL Algorithms
# supported_fl_algorithm = ["isolate", "fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedtgp", "fedpub", "fedstar", "fedgta", "fedtad", "gcfl_plus", "fedsage_plus", "adafgl", "feddep", "fggp", "fgssl", "fedgl", "hcfl_plus"]
algorithms = ["fedavg", "fedproto", "fedgta", "hcfl_plus"]

model = "gcn"

results = {}

for algo in algorithms:
    args = config.args
    args.root = "./dataset"
    args.dataset = [dataset]
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 10

    args.fl_algorithm = algo
    args.model = [model]
    args.metrics = ["accuracy"]

    print(f"\n===== Running {algo} on {dataset} =====\n")

    trainer = FGLTrainer(args)
    trainer.train()

    best_metrics = trainer.evaluation_result
    results[algo] = best_metrics

print("\n===== ALGORITHM COMPARISON ON DATASET:", dataset, "=====\n")
for algo, acc in results.items():
    print(f"{algo}: {acc}")