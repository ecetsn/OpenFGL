import csv
import json
import logging
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import warnings

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from tqdm import tqdm


SAFE_DATASETS = ["Cora", "CiteSeer", "PubMed", "Physics", "Photo", "Computers"]
EPSILONS = [5, 10, 20]
CLIENT_COUNTS = [5, 10, 20]
SEEDS = [0, 1, 2]
RUNS_DIR = "runs"
JSON_PATH = os.path.join(RUNS_DIR, "table9_safe_ablation.json")
CSV_PATH = os.path.join(RUNS_DIR, "table9_safe_ablation.csv")
LOG_PATH = os.path.join(RUNS_DIR, "table9_safe_ablation.log")
QUIET_TRAINER = True


def reset_hcfl_flags(args):
    args.hcfl_dp_stats = False
    args.hcfl_dp_membership = False
    args.hcfl_dp_sample_weights = False
    args.hcfl_entropy_reg = 0.0
    args.hcfl_dp_accounting = False
    args.secure_agg_stats = False
    args.secure_agg_mask_scale = 1.0


def apply_hcfl_dp_params(args):
    args.hcfl_proto_clip = 1.0
    args.hcfl_proto_noise = 0.1
    args.hcfl_count_clip = 10.0
    args.hcfl_count_noise = 0.5
    args.hcfl_membership_clip = 1.0
    args.hcfl_membership_noise = 0.1
    args.hcfl_sample_clip = 1.0
    args.hcfl_sample_noise = 0.1


VARIANTS = [
    {"name": "fedavg", "algo": "fedavg"},
    {"name": "hcfl_base", "algo": "hcfl_plus"},
    {"name": "hcfl_dp_stats", "algo": "hcfl_plus", "hcfl_dp_stats": True},
    {"name": "hcfl_dp_stats_membership", "algo": "hcfl_plus", "hcfl_dp_stats": True, "hcfl_dp_membership": True},
    {
        "name": "hcfl_dp_stats_membership_sample",
        "algo": "hcfl_plus",
        "hcfl_dp_stats": True,
        "hcfl_dp_membership": True,
        "hcfl_dp_sample_weights": True,
    },
    {
        "name": "hcfl_dp_stats_membership_sample_entropy",
        "algo": "hcfl_plus",
        "hcfl_dp_stats": True,
        "hcfl_dp_membership": True,
        "hcfl_dp_sample_weights": True,
        "hcfl_entropy_reg": 0.05,
    },
    {
        "name": "hcfl_pp_all",
        "algo": "hcfl_plus",
        "hcfl_dp_stats": True,
        "hcfl_dp_membership": True,
        "hcfl_dp_sample_weights": True,
        "hcfl_entropy_reg": 0.05,
        "secure_agg_stats": True,
    },
]


def run():
    os.makedirs(RUNS_DIR, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Starting table9 safe ablation run")
    warnings.filterwarnings(
        "ignore",
        message="It is not recommended to directly access the internal storage format `data`",
        category=UserWarning,
        module="torch_geometric",
    )

    args = config.args
    args.root = "./dataset"
    args.simulation_mode = "subgraph_fl_metis_plus"
    args.num_clients = 10
    args.model = ["gcn"]
    args.metrics = ["accuracy"]
    args.use_cuda = True
    args.gpuid = 0
    args.dp_mech = "gaussian"
    args.dp_delta = 1e-5

    apply_hcfl_dp_params(args)

    results = {}
    total_runs = len(SAFE_DATASETS) * len(EPSILONS) * len(CLIENT_COUNTS) * len(VARIANTS) * len(SEEDS)
    run_idx = 0

    with tqdm(total=total_runs, desc="Table9 Safe Ablation", unit="run") as pbar:
        for dataset in SAFE_DATASETS:
            args.dataset = [dataset]
            results.setdefault(dataset, {})

        for client_count in CLIENT_COUNTS:
            args.num_clients = client_count
            results[dataset].setdefault(client_count, {})

            for eps in EPSILONS:
                args.dp_eps = eps
                results[dataset][client_count].setdefault(eps, {})

                for variant in VARIANTS:
                    reset_hcfl_flags(args)
                    args.fl_algorithm = variant["algo"]

                    if variant["algo"] == "hcfl_plus":
                        args.hcfl_dp_stats = variant.get("hcfl_dp_stats", False)
                        args.hcfl_dp_membership = variant.get("hcfl_dp_membership", False)
                        args.hcfl_dp_sample_weights = variant.get("hcfl_dp_sample_weights", False)
                        args.hcfl_entropy_reg = variant.get("hcfl_entropy_reg", 0.0)
                        args.secure_agg_stats = variant.get("secure_agg_stats", False)

                    name = variant["name"]
                    accs = []
                    durations = []
                    for seed in SEEDS:
                        args.seed = seed
                        run_idx += 1
                        start = time.time()
                        pbar.set_description(f"{dataset} | clients={client_count} | eps={eps} | {name}")
                        pbar.set_postfix_str(f"run {run_idx}/{total_runs}")
                        logging.info(
                            "Run %d/%d start: %s %s clients=%s eps=%s seed=%s",
                            run_idx,
                            total_runs,
                            name,
                            dataset,
                            client_count,
                            eps,
                            seed,
                        )
                        trainer = FGLTrainer(args)
                        if QUIET_TRAINER:
                            with open(LOG_PATH, "a", encoding="utf-8") as log_file:
                                with redirect_stdout(log_file), redirect_stderr(log_file):
                                    trainer.train()
                        else:
                            trainer.train()
                        best_acc = trainer.evaluation_result["best_test_accuracy"]
                        accs.append(best_acc)
                        duration = time.time() - start
                        durations.append(duration)
                        logging.info(
                            "Run %d/%d end: %s %s clients=%s eps=%s seed=%s acc=%.6f time=%.2fs",
                            run_idx,
                            total_runs,
                            name,
                            dataset,
                            client_count,
                            eps,
                            seed,
                            best_acc,
                            duration,
                        )
                        _append_csv_row(dataset, eps, client_count, name, seed, best_acc, duration)
                        pbar.update(1)

                    mean_acc, std_acc = _mean_std(accs)
                    results[dataset][client_count][eps][name] = {
                        "mean": mean_acc,
                        "std": std_acc,
                        "seeds": accs,
                    }
                    _save_json(results)

    logging.info("Completed table9 safe ablation run")


def _save_json(results):
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _append_csv_row(dataset, eps, client_count, variant, seed, acc, duration):
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["dataset", "epsilon", "num_clients", "variant", "seed", "best_test_accuracy", "duration_sec"])
        writer.writerow([dataset, eps, client_count, variant, seed, f"{acc:.6f}", f"{duration:.2f}"])


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return mean, var ** 0.5


if __name__ == "__main__":
    run()
