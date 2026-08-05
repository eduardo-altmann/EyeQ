import argparse
import json
import os
import signal
import subprocess
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=4,
                        help='Número de GPUs a usar (--nproc_per_node do torchrun)')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Número de trials do Optuna a rodar nesta execução')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Épocas por trial (não confundir com épocas do treino final)')
    parser.add_argument('--study_name', type=str, default='eyeq_tuning')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db')
    parser.add_argument('--selection_metric', type=str, default='kappa',
                        choices=['kappa', 'macro_f1', 'auc', 'loss'],
                        help='Objective maximised by Optuna (validation split only)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Fixed seed for every trial so the val split is identical and '
                             'differences reflect hyperparameters, not RNG (+ sampler seed)')
    parser.add_argument('--warmup_startup_trials', type=int, default=4,
                        help='TPE explores randomly for this many trials before modelling')
    parser.add_argument('--pruner_warmup_epochs', type=int, default=8,
                        help='Do not prune before this epoch (let warmup + early dynamics settle)')
    return parser.parse_args()


args = parse_args()
os.makedirs('./result', exist_ok=True)


def _read_metric_lines(progress_path, seen):
    if not os.path.exists(progress_path):
        return
    with open(progress_path) as f:
        lines = f.readlines()
    for line in lines[seen[0]:]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen[0] += 1
        yield int(rec['epoch']), float(rec['score'])


def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    lr_scheduler = trial.suggest_categorical('lr_scheduler', ['cosine', 'step'])
    warmup_epochs = trial.suggest_int('warmup_epochs', 2, 6)
    momentum = trial.suggest_float('momentum', 0.8, 0.99)

    tag = f"trial_{trial.number}"
    progress_path = f"./result/{tag}_progress.jsonl"
    metrics_path = f"./result/{tag}_metrics.txt"
    for p in (progress_path, metrics_path):
        if os.path.exists(p):
            os.remove(p)

    cmd = [
        "torchrun", f"--nproc_per_node={args.n_gpus}",
        "Main_EyeQuality_tuned_parallel.py",
        "--tuning",
        "--save_model", tag,
        "--seed", str(args.seed),
        "--selection_metric", args.selection_metric,
        "--progress_file", progress_path,
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--weight_decay", str(weight_decay),
        "--lr_scheduler", lr_scheduler,
        "--momentum", str(momentum),
        "--warmup_epochs", str(warmup_epochs),
        "--epochs", str(args.epochs),
    ]

    proc = subprocess.Popen(cmd, start_new_session=True)
    seen = [0]
    pruned = False
    try:
        while proc.poll() is None:
            for epoch, score in _read_metric_lines(progress_path, seen):
                trial.report(score, step=epoch)
                if trial.should_prune():
                    pruned = True
                    break
            if pruned:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                break
            time.sleep(5)
        for epoch, score in _read_metric_lines(progress_path, seen):
            trial.report(score, step=epoch)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            proc.wait(timeout=60)

    if pruned:
        raise optuna.exceptions.TrialPruned()

    if proc.returncode not in (0, None):
        raise optuna.exceptions.TrialPruned()

    if not os.path.exists(metrics_path):
        raise optuna.exceptions.TrialPruned()

    with open(metrics_path) as f:
        for line in f:
            if line.startswith("Objective"):
                return float(line.split(":")[1])
    raise optuna.exceptions.TrialPruned()

study = optuna.create_study(
    direction="maximize",
    storage=args.storage,
    study_name=args.study_name,
    load_if_exists=True,
    sampler=TPESampler(seed=args.seed, n_startup_trials=args.warmup_startup_trials),
    pruner=MedianPruner(
        n_startup_trials=args.warmup_startup_trials,
        n_warmup_steps=args.pruner_warmup_epochs,
        interval_steps=1,
    ),
)

if len(study.trials) == 0:
    study.enqueue_trial({
        'lr': 0.001,
        'batch_size': 4,
        'weight_decay': 1e-4,
        'lr_scheduler': 'cosine',
        'warmup_epochs': 5,
        'momentum': 0.9,
    })

study.optimize(objective, n_trials=args.n_trials)

print("Best params:", study.best_params)
print("Best objective ({}):".format(args.selection_metric), study.best_value)
study.trials_dataframe().to_csv("optuna_results.csv", index=False)