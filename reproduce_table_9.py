import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# You can check the other datasets from config file
dataset = "Physics"
model = "gcn"

args = config.args
args.root = "./dataset"
args.dataset = [dataset]
args.simulation_mode = "subgraph_fl_metis_plus"
args.num_clients = 10
args.model = [model]
args.metrics = ["accuracy"]
args.dp_mech = "gaussian"
args.dp_delta = 1e-5

# hcfl+ dp settings (shared)
hcfl_proto_clip = 1.0
hcfl_proto_noise = 0.1
hcfl_count_clip = 10.0
hcfl_count_noise = 0.5
hcfl_membership_clip = 1.0
hcfl_membership_noise = 0.1
hcfl_sample_clip = 1.0
hcfl_sample_noise = 0.1

epsilons = [5, 10, 20]

experiments = [
    {"name": "fedavg", "algo": "fedavg", "pp_hcfl": False},
    {"name": "hcfl_plus", "algo": "hcfl_plus", "pp_hcfl": False},
    {"name": "hcfl_plus_pp", "algo": "hcfl_plus", "pp_hcfl": True},
]

results = {}

for eps in epsilons:
    args.dp_eps = eps
    results[eps] = {}
    for exp in experiments:
        args.fl_algorithm = exp["algo"]

        if exp["algo"] == "hcfl_plus":
            if exp["pp_hcfl"]:
                args.hcfl_dp_stats = True
                args.hcfl_dp_membership = True
                args.hcfl_dp_sample_weights = True
                args.hcfl_proto_clip = hcfl_proto_clip
                args.hcfl_proto_noise = hcfl_proto_noise
                args.hcfl_count_clip = hcfl_count_clip
                args.hcfl_count_noise = hcfl_count_noise
                args.hcfl_membership_clip = hcfl_membership_clip
                args.hcfl_membership_noise = hcfl_membership_noise
                args.hcfl_sample_clip = hcfl_sample_clip
                args.hcfl_sample_noise = hcfl_sample_noise
            else:
                args.hcfl_dp_stats = False
                args.hcfl_dp_membership = False
                args.hcfl_dp_sample_weights = False

        print(f"\n===== Running {exp['name']} on {dataset} (eps={eps}) =====\n")

        trainer = FGLTrainer(args)
        trainer.train()

        best_acc = trainer.evaluation_result["best_test_accuracy"]
        results[eps][exp["name"]] = best_acc

print("\n===== ALGORITHM COMPARISON ON DATASET:", dataset, "=====\n")
for eps, res in results.items():
    print(f"\n--- eps={eps} ---")
    for algo, acc in res.items():
        print(f"{algo}: {acc}")
