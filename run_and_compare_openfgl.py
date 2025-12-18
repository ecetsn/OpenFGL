import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import numpy as np



# All of the FL Algorithms
supported_fl_algorithms = ["isolate", "fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedtgp", "fedpub", "fedstar", "fedgta", "fedtad", "gcfl_plus", "fedsage_plus", "adafgl", "feddep", "fggp", "fgssl", "fedgl", "hcfl_plus"]
##algorithms = ["fedavg", "fedproto", "fedgta", "hcfl_plus"]

datasets = ["Cora","CiteSeer","PubMed","Photo","Computers","Products","Chameleon","Actor","Ratings"]

supported_subgraph_fl_simulations = ["subgraph_fl_label_skew", "subgraph_fl_louvain_plus", "subgraph_fl_metis_plus", "subgraph_fl_louvain", "subgraph_fl_metis"]

model = "gcn"

results = {}

for dataset in datasets:
    results[dataset] = {}

    for sim in supported_subgraph_fl_simulations:
        print(f"\n### DATASET: {dataset} | SIMULATION: {sim} ###\n")
        results[dataset][sim] = {}

        for algo in supported_fl_algorithms:
            args = config.args
            args.root = "./dataset"
            args.dataset = [dataset]
            args.simulation_mode = sim
            args.num_clients = 10
            args.fl_algorithm = algo
            args.model = [model]
            args.metrics = ["accuracy"]

            # 🔹 skew_alpha FIX: define it only when needed
            if "skew" in sim or "plus" in sim:
                args.skew_alpha = 0.5     # try different values later
            else:
                args.skew_alpha = 0.0     # prevent crash; not used

            print(f"\n===== Running {algo} on {dataset} with {sim} =====\n")

            # store results from 3 runs
            run_metrics = []
            for _ in range(3):
                trainer = FGLTrainer(args)
                trainer.train()
                run_metrics.append(trainer.evaluation_result)

            # average the metrics
            avg_metrics = {}
            for key in run_metrics[0].keys():
                values = [rm[key] for rm in run_metrics]
                avg_metrics[key] = float(np.mean(values))

            # save inside nested dict
            results[dataset][sim][algo] = avg_metrics

# 🔹 Print all results
print("\n===== EXPERIMENT RESULTS (nested) =====\n")
for dataset, sim_res in results.items():
    print(f"\n=== DATASET: {dataset} ===")
    for sim_mode, algo_res in sim_res.items():
        print(f"\n  SIMULATION: {sim_mode}")
        for algo, metrics in algo_res.items():
            print(f"    {algo}: {metrics}")

# 🔹 OPTIONAL — Export to CSV
all_rows = []
for dataset, sim_res in results.items():
    for sim_mode, algo_res in sim_res.items():
        for algo, metrics in algo_res.items():
            row = {"dataset": dataset, "simulation": sim_mode, "algorithm": algo}
            row.update(metrics)
            all_rows.append(row)

df = pd.DataFrame(all_rows)
df.to_csv("all_experiment_results.csv", index=False)
print("\nSaved as: all_experiment_results.csv")