"""
Consolidated Table 9 runner for OpenFGL experiments.

Edit the configuration below to control which experiments to run:
- RUN_MODE: "full" | "hcfl" | "base"
- RUN_SCENARIO: "all" | "graph_fl" | "subgraph_fl"

The script automatically resumes from where it stopped by tracking
completed runs in the CSV file.
"""
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


# === CONFIGURATION (edit these to control experiments) ===
RUN_MODE = "hcfl"           # "full", "hcfl", "base"
RUN_SCENARIO = "all"        # "all", "graph_fl", "subgraph_fl"
SEEDS = [0, 1, 2]
QUIET_TRAINER = True
RUNS_DIR = "runs"

# === FILE PATHS ===
CSV_PATH = os.path.join(RUNS_DIR, "table9.csv")
JSON_PATH = os.path.join(RUNS_DIR, "table9.json")
MD_PATH = os.path.join(RUNS_DIR, "table9.md")
LOG_PATH = os.path.join(RUNS_DIR, "table9.log")

# === EXPERIMENT DEFINITIONS ===
GRAPH_FL_DATASETS = ["AIDS", "NCI1"]
GRAPH_FL_SIMS = [
    ("Label", "graph_fl_label_skew"),
    ("Topology", "graph_fl_topology_skew"),
]
GRAPH_FL_BASELINES = ["fedavg", "gcfl_plus"]
GRAPH_FL_EPS = [None, 5, 20]

SUBGRAPH_FL_DATASETS = ["CS", "Physics"]
SUBGRAPH_FL_SIMS = [
    ("Louvain+", "subgraph_fl_louvain_plus"),
    ("Metis+", "subgraph_fl_metis_plus"),
]
SUBGRAPH_FL_BASELINES = ["fedavg", "fedgta", "fedsage_plus", "feddep"]
SUBGRAPH_FL_EPS = [None, 10]

HCFL_VARIANTS = [
    {"name": "hcfl_base", "dp": False},
    {"name": "hcfl_dp_stats_membership_sample", "dp": True, "entropy": 0.0, "secure": False},
    {"name": "hcfl_pp_all", "dp": True, "entropy": 0.05, "secure": True},
]


def reset_hcfl_flags(args):
    """Reset all HCFL+ flags to defaults."""
    args.hcfl_dp_stats = False
    args.hcfl_dp_membership = False
    args.hcfl_dp_sample_weights = False
    args.hcfl_entropy_reg = 0.0
    args.hcfl_dp_accounting = False
    args.secure_agg_stats = False
    args.secure_agg_mask_scale = 1.0


def apply_hcfl_dp_params(args):
    """Apply HCFL+ DP clipping and noise parameters."""
    args.hcfl_proto_clip = 1.0
    args.hcfl_proto_noise = 0.1
    args.hcfl_count_clip = 10.0
    args.hcfl_count_noise = 0.5
    args.hcfl_membership_clip = 1.0
    args.hcfl_membership_noise = 0.1
    args.hcfl_sample_clip = 1.0
    args.hcfl_sample_noise = 0.1


def load_completed():
    """Load completed runs from CSV as set of tuples for resume."""
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
    """Append a single result row to CSV (atomic append for resume)."""
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "scenario", "dataset", "simulation", "algorithm",
                "variant", "epsilon", "seed", "best_test_accuracy", "duration_sec"
            ])
        writer.writerow(row)


def load_values(scenario, dataset, sim_mode, algorithm, variant, eps):
    """Load accuracy values from CSV for computing mean/std."""
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


def mean_std(values):
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return mean, var ** 0.5


def save_json(results):
    """Save aggregated results to JSON."""
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_mode": RUN_MODE,
        "run_scenario": RUN_SCENARIO,
        "results": results,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_markdown(results):
    """Save formatted markdown table."""
    def fmt(val):
        return f"{val * 100:.1f}"

    def fmt_pm(mean, std):
        return f"{fmt(mean)}±{fmt(std)}"

    lines = []
    lines.append("# Table 9 Results (Mean ± Std, %)")
    lines.append(f"\n**Mode:** {RUN_MODE} | **Scenario:** {RUN_SCENARIO}")
    lines.append(f"\n**Generated:** {datetime.utcnow().isoformat()}Z")
    lines.append("")

    # Graph-FL section
    graph = results.get("graph_fl", {})
    if graph:
        lines.append("## Graph-FL")
        for dataset in GRAPH_FL_DATASETS:
            if dataset not in graph:
                continue
            lines.append(f"\n### {dataset}")
            lines.append("| Algorithm | Variant | Label | Topology |")
            lines.append("|---|---|---:|---:|")

            # Collect all algorithms for this dataset
            all_algos = set()
            for sim_mode in ["graph_fl_label_skew", "graph_fl_topology_skew"]:
                all_algos.update(graph[dataset].get(sim_mode, {}).keys())

            for algo in sorted(all_algos):
                # Get all variants for this algorithm
                variants = set()
                for sim_mode in ["graph_fl_label_skew", "graph_fl_topology_skew"]:
                    variants.update(graph[dataset].get(sim_mode, {}).get(algo, {}).keys())

                for variant in sorted(variants):
                    label_data = graph[dataset].get("graph_fl_label_skew", {}).get(algo, {}).get(variant, {})
                    topo_data = graph[dataset].get("graph_fl_topology_skew", {}).get(algo, {}).get(variant, {})
                    label_str = fmt_pm(label_data.get("mean", 0), label_data.get("std", 0))
                    topo_str = fmt_pm(topo_data.get("mean", 0), topo_data.get("std", 0))
                    lines.append(f"| {algo} | {variant} | {label_str} | {topo_str} |")
        lines.append("")

    # Subgraph-FL section
    sub = results.get("subgraph_fl", {})
    if sub:
        lines.append("## Subgraph-FL")
        for dataset in SUBGRAPH_FL_DATASETS:
            if dataset not in sub:
                continue
            lines.append(f"\n### {dataset}")
            lines.append("| Algorithm | Variant | Louvain+ | Metis+ |")
            lines.append("|---|---|---:|---:|")

            all_algos = set()
            for sim_mode in ["subgraph_fl_louvain_plus", "subgraph_fl_metis_plus"]:
                all_algos.update(sub[dataset].get(sim_mode, {}).keys())

            for algo in sorted(all_algos):
                variants = set()
                for sim_mode in ["subgraph_fl_louvain_plus", "subgraph_fl_metis_plus"]:
                    variants.update(sub[dataset].get(sim_mode, {}).get(algo, {}).keys())

                for variant in sorted(variants):
                    louv_data = sub[dataset].get("subgraph_fl_louvain_plus", {}).get(algo, {}).get(variant, {})
                    metis_data = sub[dataset].get("subgraph_fl_metis_plus", {}).get(algo, {}).get(variant, {})
                    louv_str = fmt_pm(louv_data.get("mean", 0), louv_data.get("std", 0))
                    metis_str = fmt_pm(metis_data.get("mean", 0), metis_data.get("std", 0))
                    lines.append(f"| {algo} | {variant} | {louv_str} | {metis_str} |")
        lines.append("")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_experiments():
    """Build experiment list based on RUN_MODE and RUN_SCENARIO."""
    experiments = []
    include_graph = RUN_SCENARIO in ("all", "graph_fl")
    include_subgraph = RUN_SCENARIO in ("all", "subgraph_fl")
    include_base = RUN_MODE in ("full", "base")
    include_hcfl = RUN_MODE in ("full", "hcfl")

    # Graph-FL Baselines
    if include_graph and include_base:
        for dataset in GRAPH_FL_DATASETS:
            for sim_name, sim_mode in GRAPH_FL_SIMS:
                for algo in GRAPH_FL_BASELINES:
                    for eps in GRAPH_FL_EPS:
                        variant = "base" if eps is None else f"dp_eps_{eps}"
                        experiments.append({
                            "scenario": "graph_fl",
                            "dataset": dataset,
                            "simulation_mode": sim_mode,
                            "simulation_label": sim_name,
                            "algorithm": algo,
                            "variant": variant,
                            "epsilon": eps,
                        })

    # Graph-FL HCFL+ variants
    if include_graph and include_hcfl:
        for dataset in GRAPH_FL_DATASETS:
            for sim_name, sim_mode in GRAPH_FL_SIMS:
                for hcfl in HCFL_VARIANTS:
                    if hcfl["dp"]:
                        for eps in [5, 20]:  # Match Graph-FL baselines
                            experiments.append({
                                "scenario": "graph_fl",
                                "dataset": dataset,
                                "simulation_mode": sim_mode,
                                "simulation_label": sim_name,
                                "algorithm": "hcfl_plus",
                                "variant": f"{hcfl['name']}_eps{eps}",
                                "epsilon": eps,
                                "hcfl_dp": True,
                                "hcfl_entropy": hcfl.get("entropy", 0.0),
                                "hcfl_secure": hcfl.get("secure", False),
                            })
                    else:
                        experiments.append({
                            "scenario": "graph_fl",
                            "dataset": dataset,
                            "simulation_mode": sim_mode,
                            "simulation_label": sim_name,
                            "algorithm": "hcfl_plus",
                            "variant": hcfl["name"],
                            "epsilon": None,
                            "hcfl_dp": False,
                            "hcfl_entropy": 0.0,
                            "hcfl_secure": False,
                        })

    # Subgraph-FL Baselines
    if include_subgraph and include_base:
        for dataset in SUBGRAPH_FL_DATASETS:
            for sim_name, sim_mode in SUBGRAPH_FL_SIMS:
                for algo in SUBGRAPH_FL_BASELINES:
                    for eps in SUBGRAPH_FL_EPS:
                        variant = "base" if eps is None else f"dp_eps_{eps}"
                        experiments.append({
                            "scenario": "subgraph_fl",
                            "dataset": dataset,
                            "simulation_mode": sim_mode,
                            "simulation_label": sim_name,
                            "algorithm": algo,
                            "variant": variant,
                            "epsilon": eps,
                        })

    # Subgraph-FL HCFL+ variants
    if include_subgraph and include_hcfl:
        for dataset in SUBGRAPH_FL_DATASETS:
            for sim_name, sim_mode in SUBGRAPH_FL_SIMS:
                for hcfl in HCFL_VARIANTS:
                    if hcfl["dp"]:
                        for eps in [10]:  # Match Subgraph-FL baselines
                            experiments.append({
                                "scenario": "subgraph_fl",
                                "dataset": dataset,
                                "simulation_mode": sim_mode,
                                "simulation_label": sim_name,
                                "algorithm": "hcfl_plus",
                                "variant": f"{hcfl['name']}_eps{eps}",
                                "epsilon": eps,
                                "hcfl_dp": True,
                                "hcfl_entropy": hcfl.get("entropy", 0.0),
                                "hcfl_secure": hcfl.get("secure", False),
                            })
                    else:
                        experiments.append({
                            "scenario": "subgraph_fl",
                            "dataset": dataset,
                            "simulation_mode": sim_mode,
                            "simulation_label": sim_name,
                            "algorithm": "hcfl_plus",
                            "variant": hcfl["name"],
                            "epsilon": None,
                            "hcfl_dp": False,
                            "hcfl_entropy": 0.0,
                            "hcfl_secure": False,
                        })

    return experiments


def run():
    """Main execution function."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Starting Table 9 runner (mode=%s, scenario=%s)", RUN_MODE, RUN_SCENARIO)

    warnings.filterwarnings(
        "ignore",
        message="It is not recommended to directly access the internal storage format `data`",
        category=UserWarning,
        module="torch_geometric",
    )

    # Base configuration
    args = config.args
    args.root = "./dataset"
    args.metrics = ["accuracy"]
    args.use_cuda = True
    args.gpuid = 0
    args.dp_mech = "gaussian"
    args.dp_delta = 1e-5
    args.skew_alpha = 0.5

    apply_hcfl_dp_params(args)
    experiments = build_experiments()
    completed = load_completed()

    total_runs = len(experiments) * len(SEEDS)
    run_idx = 0
    results = {}

    print(f"Table 9 Runner | Mode: {RUN_MODE} | Scenario: {RUN_SCENARIO}")
    print(f"Total experiments: {len(experiments)} | Seeds: {len(SEEDS)} | Total runs: {total_runs}")
    print(f"Output: {CSV_PATH}")
    print("-" * 60)

    with tqdm(total=total_runs, desc="Table9", unit="run") as pbar:
        for exp in experiments:
            scenario = exp["scenario"]
            dataset = exp["dataset"]
            sim_mode = exp["simulation_mode"]
            algorithm = exp["algorithm"]
            variant = exp["variant"]
            eps = exp["epsilon"]

            # Configure args for this experiment
            args.scenario = scenario
            args.dataset = [dataset]
            args.simulation_mode = sim_mode
            args.fl_algorithm = algorithm

            if scenario == "graph_fl":
                args.task = "graph_cls"
                args.model = ["gin"]
            else:
                args.task = "node_cls"
                args.model = ["gcn"]

            if eps is None:
                args.dp_mech = "no_dp"
                args.dp_eps = 0.0
            else:
                args.dp_mech = "gaussian"
                args.dp_eps = eps

            # Configure HCFL+ specific settings
            if algorithm == "hcfl_plus":
                reset_hcfl_flags(args)
                if exp.get("hcfl_dp"):
                    args.hcfl_dp_stats = True
                    args.hcfl_dp_membership = True
                    args.hcfl_dp_sample_weights = True
                    args.hcfl_entropy_reg = exp.get("hcfl_entropy", 0.0)
                    args.secure_agg_stats = exp.get("hcfl_secure", False)

            for seed in SEEDS:
                key = (scenario, dataset, sim_mode, algorithm, variant,
                       "" if eps is None else str(eps), str(seed))

                if key in completed:
                    run_idx += 1
                    pbar.update(1)
                    continue

                args.seed = seed
                run_idx += 1
                pbar.set_description(f"{dataset} | {algorithm} | {variant}")
                pbar.set_postfix_str(f"seed={seed}")

                logging.info(
                    "Run %d/%d start: %s %s %s %s variant=%s eps=%s seed=%s",
                    run_idx, total_runs, scenario, dataset, sim_mode,
                    algorithm, variant, eps, seed
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
                except Exception:
                    duration = time.time() - start
                    logging.exception(
                        "Run %d/%d FAILED: %s %s %s %s eps=%s seed=%s time=%.2fs",
                        run_idx, total_runs, scenario, dataset, sim_mode,
                        algorithm, eps, seed, duration
                    )
                    pbar.update(1)
                    continue

                duration = time.time() - start
                append_csv([
                    scenario, dataset, sim_mode, algorithm, variant,
                    "" if eps is None else eps, seed,
                    f"{best_acc:.6f}", f"{duration:.2f}"
                ])

                logging.info(
                    "Run %d/%d end: %s %s %s %s eps=%s seed=%s acc=%.6f time=%.2fs",
                    run_idx, total_runs, scenario, dataset, sim_mode,
                    algorithm, eps, seed, best_acc, duration
                )
                pbar.update(1)

            # Update aggregated results after each experiment (all seeds)
            results.setdefault(scenario, {})
            results[scenario].setdefault(dataset, {})
            results[scenario][dataset].setdefault(sim_mode, {})
            results[scenario][dataset][sim_mode].setdefault(algorithm, {})

            values = load_values(scenario, dataset, sim_mode, algorithm, variant, eps)
            mean, std = mean_std(values)
            results[scenario][dataset][sim_mode][algorithm][variant] = {
                "epsilon": eps,
                "mean": mean,
                "std": std,
                "n": len(values),
            }
            save_json(results)

    # Final markdown output
    save_markdown(results)
    logging.info("Completed Table 9 runner")
    print("-" * 60)
    print(f"Done! Results saved to {CSV_PATH}")


if __name__ == "__main__":
    run()
