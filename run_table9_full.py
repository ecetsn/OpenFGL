import csv
import json
import logging
import os
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from tqdm import tqdm


RUNS_DIR = "runs"
CSV_PATH = os.path.join(RUNS_DIR, "table9_full.csv")
JSON_PATH = os.path.join(RUNS_DIR, "table9_full.json")
LOG_PATH = os.path.join(RUNS_DIR, "table9_full.log")
QUIET_TRAINER = True
SEEDS = [0, 1, 2]


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


def load_completed():
    completed = set()
    if not os.path.exists(CSV_PATH):
        return completed
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["scenario"],
                row["dataset"],
                row["simulation"],
                row["algorithm"],
                row["variant"],
                row["epsilon"],
                row["seed"],
            )
            completed.add(key)
    return completed


def append_csv(row):
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "scenario",
                    "dataset",
                    "simulation",
                    "algorithm",
                    "variant",
                    "epsilon",
                    "seed",
                    "best_test_accuracy",
                    "duration_sec",
                ]
            )
        writer.writerow(row)


def save_json(results):
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return mean, var ** 0.5


def build_experiments():
    experiments = []

    # Graph-FL section (AIDS, NCI1) with Label and Topology splits.
    graph_datasets = ["AIDS", "NCI1"]
    graph_sims = [
        ("Label", "graph_fl_label_skew"),
        ("Topology", "graph_fl_topology_skew"),
    ]
    graph_algos = ["fedavg", "gcfl_plus"]
    graph_eps = [None, 5, 20]

    for dataset in graph_datasets:
        for sim_name, sim_mode in graph_sims:
            for algo in graph_algos:
                for eps in graph_eps:
                    variant = "base" if eps is None else f"dp_eps_{eps}"
                    experiments.append(
                        {
                            "scenario": "graph_fl",
                            "dataset": dataset,
                            "simulation_mode": sim_mode,
                            "simulation_label": sim_name,
                            "algorithm": algo,
                            "variant": variant,
                            "epsilon": eps,
                        }
                    )

    # Subgraph-FL section (CS, Physics) with Louvain+ and Metis+.
    subgraph_datasets = ["CS", "Physics"]
    subgraph_sims = [
        ("Louvain+", "subgraph_fl_louvain_plus"),
        ("Metis+", "subgraph_fl_metis_plus"),
    ]
    subgraph_algos = ["fedavg", "fedgta", "fedsage_plus", "feddep"]
    subgraph_eps = [None, 10]

    for dataset in subgraph_datasets:
        for sim_name, sim_mode in subgraph_sims:
            for algo in subgraph_algos:
                for eps in subgraph_eps:
                    variant = "base" if eps is None else f"dp_eps_{eps}"
                    experiments.append(
                        {
                            "scenario": "subgraph_fl",
                            "dataset": dataset,
                            "simulation_mode": sim_mode,
                            "simulation_label": sim_name,
                            "algorithm": algo,
                            "variant": variant,
                            "epsilon": eps,
                        }
                    )

    # HCFL+ variants (Subgraph-FL, eps=10)
    hcfl_variants = [
        {"name": "hcfl_base", "dp": False},
        {"name": "hcfl_dp_stats_membership_sample", "dp": True, "entropy": 0.0, "secure": False},
        {"name": "hcfl_pp_all", "dp": True, "entropy": 0.05, "secure": True},
    ]
    for dataset in subgraph_datasets:
        for sim_name, sim_mode in subgraph_sims:
            for hcfl in hcfl_variants:
                experiments.append(
                    {
                        "scenario": "subgraph_fl",
                        "dataset": dataset,
                        "simulation_mode": sim_mode,
                        "simulation_label": sim_name,
                        "algorithm": "hcfl_plus",
                        "variant": hcfl["name"],
                        "epsilon": 10 if hcfl["dp"] else None,
                        "hcfl_dp": hcfl["dp"],
                        "hcfl_entropy": hcfl.get("entropy", 0.0),
                        "hcfl_secure": hcfl.get("secure", False),
                    }
                )
    return experiments


def run():
    os.makedirs(RUNS_DIR, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Starting Table 9 full runner")
    warnings.filterwarnings(
        "ignore",
        message="It is not recommended to directly access the internal storage format `data`",
        category=UserWarning,
        module="torch_geometric",
    )

    args = config.args
    args.root = "./dataset"
    args.model = ["gcn"]
    args.metrics = ["accuracy"]
    args.use_cuda = True
    args.gpuid = 0
    args.dp_mech = "gaussian"
    args.dp_delta = 1e-5

    apply_hcfl_dp_params(args)
    experiments = build_experiments()
    completed = load_completed()

    total_runs = len(experiments) * len(SEEDS)
    run_idx = 0

    results = {}

    with tqdm(total=total_runs, desc="Table9 Full", unit="run") as pbar:
        for exp in experiments:
            scenario = exp["scenario"]
            dataset = exp["dataset"]
            sim_mode = exp["simulation_mode"]
            algorithm = exp["algorithm"]
            variant = exp["variant"]
            eps = exp["epsilon"]

            args.scenario = scenario
            args.dataset = [dataset]
            args.simulation_mode = sim_mode
            args.fl_algorithm = algorithm

            if scenario == "graph_fl":
                args.task = "graph_cls"
            else:
                args.task = "node_cls"

            if eps is None:
                args.dp_mech = "no_dp"
                args.dp_eps = 0.0
            else:
                args.dp_mech = "gaussian"
                args.dp_eps = eps

            if algorithm == "hcfl_plus":
                reset_hcfl_flags(args)
                if exp.get("hcfl_dp"):
                    args.hcfl_dp_stats = True
                    args.hcfl_dp_membership = True
                    args.hcfl_dp_sample_weights = True
                    args.hcfl_entropy_reg = exp.get("hcfl_entropy", 0.0)
                    args.secure_agg_stats = exp.get("hcfl_secure", False)

            accs = []
            for seed in SEEDS:
                key = (scenario, dataset, sim_mode, algorithm, variant, str(eps), str(seed))
                if key in completed:
                    run_idx += 1
                    pbar.update(1)
                    continue

                args.seed = seed
                run_idx += 1
                pbar.set_description(f"{dataset} | {sim_mode} | {algorithm} | {variant}")
                pbar.set_postfix_str(f"run {run_idx}/{total_runs}")

                logging.info(
                    "Run %d/%d start: %s %s %s %s eps=%s seed=%s",
                    run_idx,
                    total_runs,
                    scenario,
                    dataset,
                    sim_mode,
                    algorithm,
                    eps,
                    seed,
                )
                start = time.time()
                try:
                    trainer = FGLTrainer(args)
                    if QUIET_TRAINER:
                        with open(LOG_PATH, "a", encoding="utf-8") as log_file:
                            with redirect_stdout(log_file), redirect_stderr(log_file):
                                trainer.train()
                    else:
                        trainer.train()
                    best_acc = trainer.evaluation_result["best_test_accuracy"]
                except Exception as exc:
                    duration = time.time() - start
                    logging.exception(
                        "Run %d/%d failed: %s %s %s %s eps=%s seed=%s time=%.2fs",
                        run_idx,
                        total_runs,
                        scenario,
                        dataset,
                        sim_mode,
                        algorithm,
                        eps,
                        seed,
                        duration,
                    )
                    pbar.update(1)
                    continue

                duration = time.time() - start
                accs.append(best_acc)

                append_csv(
                    [
                        scenario,
                        dataset,
                        sim_mode,
                        algorithm,
                        variant,
                        "" if eps is None else eps,
                        seed,
                        f"{best_acc:.6f}",
                        f"{duration:.2f}",
                    ]
                )
                logging.info(
                    "Run %d/%d end: %s %s %s %s eps=%s seed=%s acc=%.6f time=%.2fs",
                    run_idx,
                    total_runs,
                    scenario,
                    dataset,
                    sim_mode,
                    algorithm,
                    eps,
                    seed,
                    best_acc,
                    duration,
                )
                pbar.update(1)

            # Aggregate mean/std for this experiment (from CSV so resume works).
            results.setdefault(scenario, {})
            results[scenario].setdefault(dataset, {})
            results[scenario][dataset].setdefault(sim_mode, {})
            results[scenario][dataset][sim_mode].setdefault(algorithm, {})
            results[scenario][dataset][sim_mode][algorithm].setdefault(variant, {})
            values = _load_values(scenario, dataset, sim_mode, algorithm, variant, eps)
            mean, std = mean_std(values)
            results[scenario][dataset][sim_mode][algorithm][variant] = {
                "epsilon": eps,
                "mean": mean,
                "std": std,
                "n": len(values),
            }
            save_json(results)

    logging.info("Completed Table 9 full runner")


def _load_values(scenario, dataset, sim_mode, algorithm, variant, eps):
    values = []
    if not os.path.exists(CSV_PATH):
        return values
    target_eps = "" if eps is None else str(eps)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["scenario"] == scenario
                and row["dataset"] == dataset
                and row["simulation"] == sim_mode
                and row["algorithm"] == algorithm
                and row["variant"] == variant
                and row["epsilon"] == target_eps
            ):
                try:
                    values.append(float(row["best_test_accuracy"]))
                except ValueError:
                    pass
    return values


if __name__ == "__main__":
    run()
