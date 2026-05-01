import os
import json
import pickle
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# Import from the actual (fixed) filenames
from pporandomized import train_ppo
from ppolagrandomaized import train_ppolag
from ltlppolabelledrandomized import train_labelled_ltl_ppo


OUT_DIR = "safetreasuregoal_comparison_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]

COMMON_CONFIG = dict(
    total_steps=500_000,
    rollout_steps=1024,
    update_epochs=8,
    minibatch_size=256,
)

ALGO_CONFIGS = {
    "LTL-PPO": dict(
        gamma=0.99,
        lam=0.95,
        clip_coef=0.20,
        vf_coef=0.50,
        ent_coef=0.04,
        lr=3e-4,
        progress_bonus=1.91,
        cliff_penalty=0.0,
        terminate_on_cliff=False,   # accepted by train_labelled_ltl_ppo, ignored by env
    ),
    "PPO": dict(
        gamma=0.99,
        lam=0.95,
        clip_coef=0.20,
        vf_coef=0.50,
        ent_coef=0.04,
        lr=3e-4,
    ),
    "PPO-Lag": dict(
        gamma=0.99,
        cost_gamma=0.99,
        lam=0.95,
        clip_coef=0.20,
        vf_coef=0.50,
        cost_vf_coef=0.50,
        ent_coef=0.04,
        lr=3e-4,
        lag_lr=0.01,
        cost_limit=1.0,
    ),
}

TRAINERS = {
    "LTL-PPO": train_labelled_ltl_ppo,
    "PPO": train_ppo,
    "PPO-Lag": train_ppolag,
}

# All three algos log under these exact keys — verified against each trainer.
METRICS = {
    "eval_return":   ("Evaluation Return",       "Return",    "comparison_eval_return.png",   None),
    "eval_cost":     ("Evaluation Cost",          "Cost",      "comparison_eval_cost.png",     None),
    "eval_success":  ("Goal Success Rate",        "Rate",      "comparison_success_rate.png",  (0.0, 1.05)),
    "eval_ltl":      ("LTL Satisfaction Rate",    "Rate",      "comparison_ltl_rate.png",      (0.0, 1.05)),
    "key_rate":      ("Key Rate",                 "Rate",      "comparison_key_rate.png",      (0.0, 1.05)),
    "door_rate":     ("Door Rate",                "Rate",      "comparison_door_rate.png",     (0.0, 1.05)),
    "goal_rate":     ("Goal Rate",                "Rate",      "comparison_goal_rate.png",     (0.0, 1.05)),
    "unsafe_rate":   ("Unsafe Visit Rate",        "Rate",      "comparison_unsafe_rate.png",   (0.0, 1.05)),
    "treasure_count":("Average Treasure Count",   "Treasures", "comparison_treasure_count.png",None),
}

# Metrics that only exist for a single algo
EXTRA_METRICS = {
    "lagrangian":      ("PPO-Lag Lagrange Multiplier", "Lambda",           "ppolag_lagrangian.png",    None),
    "eval_ltl_return": ("LTL-PPO Shaped Return",       "LTL-shaped return","ltlppo_shaped_return.png", None),
}


def align_metric(seed_logs, metric):
    valid_logs = [log for log in seed_logs if metric in log and len(log[metric]) > 0]

    if len(valid_logs) == 0:
        return None, None, None

    min_len = min(len(log[metric]) for log in valid_logs)
    y = np.array([log[metric][:min_len] for log in valid_logs], dtype=float)
    x = np.array(valid_logs[0]["steps"][:min_len], dtype=float)

    mean = y.mean(axis=0)
    stderr = y.std(axis=0) / np.sqrt(y.shape[0])

    return x, mean, stderr


def latest(log, key, default=np.nan):
    values = log.get(key, [])
    return values[-1] if len(values) > 0 else default


def print_latest(algo, seed, log):
    line = (
        f"{algo:<8} | seed {seed:03d} | "
        f"Steps {int(latest(log, 'steps', 0)):08d} | "
        f"EvalRet {latest(log, 'eval_return'):8.3f} | "
        f"Cost {latest(log, 'eval_cost'):7.3f} | "
        f"Succ {latest(log, 'eval_success'):.2f} | "
        f"LTL {latest(log, 'eval_ltl'):.2f} | "
        f"K {latest(log, 'key_rate'):.2f} | "
        f"D {latest(log, 'door_rate'):.2f} | "
        f"G {latest(log, 'goal_rate'):.2f} | "
        f"Unsafe {latest(log, 'unsafe_rate'):.2f} | "
        f"T {latest(log, 'treasure_count'):.2f}"
    )
    if "lagrangian" in log:
        line += f" | Lag {latest(log, 'lagrangian'):7.3f}"
    if "eval_ltl_return" in log:
        line += f" | LTLRet {latest(log, 'eval_ltl_return'):8.3f}"
    print(line)


def plot_metric(results, metric, title, ylabel, filename, ylim=None):
    plt.figure(figsize=(9, 6))
    plotted = False

    for algo, seed_logs in results.items():
        x, mean, stderr = align_metric(seed_logs, metric)
        if x is None:
            continue
        plt.plot(x, mean, label=algo)
        plt.fill_between(x, mean - stderr, mean + stderr, alpha=0.20)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(f"SafeTreasureGoal: {title}")
    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.show()
    print(f"Saved plot: {path}")


def plot_all(results):
    for metric, (title, ylabel, filename, ylim) in METRICS.items():
        plot_metric(results, metric, title, ylabel, filename, ylim)

    plot_metric(
        {"PPO-Lag": results.get("PPO-Lag", [])},
        "lagrangian",
        EXTRA_METRICS["lagrangian"][0],
        EXTRA_METRICS["lagrangian"][1],
        EXTRA_METRICS["lagrangian"][2],
        EXTRA_METRICS["lagrangian"][3],
    )

    plot_metric(
        {"LTL-PPO": results.get("LTL-PPO", [])},
        "eval_ltl_return",
        EXTRA_METRICS["eval_ltl_return"][0],
        EXTRA_METRICS["eval_ltl_return"][1],
        EXTRA_METRICS["eval_ltl_return"][2],
        EXTRA_METRICS["eval_ltl_return"][3],
    )


def print_summary(results):
    all_metrics = list(METRICS.keys()) + list(EXTRA_METRICS.keys())

    print("\n" + "=" * 120)
    print("Final 5-seed averaged summary")
    print("=" * 120)

    for algo, seed_logs in results.items():
        print(f"\n{algo}")
        print("-" * 120)

        for metric in all_metrics:
            vals = [
                log[metric][-1]
                for log in seed_logs
                if metric in log and len(log[metric]) > 0
            ]
            if not vals:
                continue

            vals = np.asarray(vals, dtype=float)
            print(
                f"{metric:<18} | "
                f"mean {vals.mean():9.4f} | "
                f"std {vals.std():9.4f} | "
                f"stderr {vals.std() / np.sqrt(len(vals)):9.4f} | "
                f"n {len(vals)}"
            )


def save_results(results):
    logs_path = os.path.join(OUT_DIR, "all_logs.pkl")
    cfg_path = os.path.join(OUT_DIR, "runner_config.json")

    with open(logs_path, "wb") as f:
        pickle.dump(results, f)

    with open(cfg_path, "w") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "common_config": COMMON_CONFIG,
                "algo_configs": ALGO_CONFIGS,
            },
            f,
            indent=4,
        )

    print(f"\nSaved logs: {logs_path}")
    print(f"Saved config: {cfg_path}")


def run_all():
    results = {algo: [] for algo in TRAINERS}

    for algo, trainer in TRAINERS.items():
        print("\n" + "=" * 120)
        print(f"Running {algo}")
        print("=" * 120)

        for seed in SEEDS:
            cfg = deepcopy(COMMON_CONFIG)
            cfg.update(ALGO_CONFIGS[algo])
            cfg["seed"] = seed

            print(f"\n{algo} | seed {seed}")
            print("-" * 120)

            _, logs = trainer(**cfg)
            results[algo].append(logs)

            print_latest(algo, seed, logs)

            per_seed_path = os.path.join(
                OUT_DIR,
                f"{algo.replace('-', '').replace(' ', '_').lower()}_seed{seed}_logs.pkl",
            )
            with open(per_seed_path, "wb") as f:
                pickle.dump(logs, f)

    save_results(results)
    print_summary(results)
    plot_all(results)

    return results


if __name__ == "__main__":
    run_all()