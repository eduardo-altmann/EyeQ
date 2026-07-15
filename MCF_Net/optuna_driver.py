import optuna, subprocess, json, os

def objective(trial):
    lr = trial.suggest_float('lr', 5e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    lr_scheduler = trial.suggest_categorical('lr_scheduler', ['cosine', 'step'])
    warmup_epochs = trial.suggest_int('warmup_epochs', 3, 7)

    tag = f"trial_{trial.number}"
    cmd = [
        "torchrun", "--nproc_per_node=4",
        "Main_EyeQuality_tuned_parallel.py",
        "--save_model", tag,
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--weight_decay", str(weight_decay),
        "--lr_scheduler", lr_scheduler,
        "--warmup_epochs", str(warmup_epochs),
        "--epochs", "25",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    metrics_path = f"./result/{tag}_metrics.txt"
    if not os.path.exists(metrics_path):
        raise optuna.exceptions.TrialPruned()

    # ler a val_loss salva pelo próprio script de treino
    with open(metrics_path) as f:
        for line in f:
            if line.startswith("Best_Val_Loss"):
                return float(line.split(":")[1])
    raise optuna.exceptions.TrialPruned()

study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study.db", study_name="eyeq_tuning")
study.optimize(objective, n_trials=12)

print(study.best_params)
print(study.best_value)
df = study.trials_dataframe()
df.to_csv("./result/optuna_results.csv")
