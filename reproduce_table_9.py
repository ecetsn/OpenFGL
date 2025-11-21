import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# You can check the other datasets from config file
dataset = "Physics"

# All of the FL Algorithms
# supported_fl_algorithm = ["isolate", "fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedtgp", "fedpub", "fedstar", "fedgta", "fedtad", "gcfl_plus", "fedsage_plus", "adafgl", "feddep", "fggp", "fgssl", "fedgl", "hcfl_plus"]
algorithms = ["fedavg"]

model = "gcn"

args = config.args
args.root = "./dataset"
args.dataset = [dataset]
args.simulation_mode = "subgraph_fl_metis_plus"
args.num_clients = 10
args.model = [model]
args.metrics = ["accuracy"]
args.dp_eps = 5
# “The parameter δ represents the maximum permissible failure probability and is usually chosen to be much smaller 
# than the inverse of the number of data records.”
# If a dataset has N samples, they imply: δ << 1 / N
args.dp_delta = 10
results = {}

for algo in algorithms:

    args.fl_algorithm = algo

    print(f"\n===== Running {algo} on {dataset} =====\n")

    trainer = FGLTrainer(args)
    trainer.train()

    best_acc = trainer.evaluation_result["best_test_accuracy"]
    results[algo] = best_acc

print("\n===== ALGORITHM COMPARISON ON DATASET:", dataset, "=====\n")
for algo, acc in results.items():
    print(f"{algo}: {acc}")