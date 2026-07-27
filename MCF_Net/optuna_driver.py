import argparse
import optuna, subprocess, os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=4,
                        help='Número de GPUs a usar (--nproc_per_node do torchrun)')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Número de trials do Optuna a rodar nesta execução')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Épocas por trial (não confundir com épocas do treino final)')
    parser.add_argument('--study_name', type=str, default='eyeq_tuning')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db')
    return parser.parse_args()

args = parse_args()

def objective(trial):
    lr = trial.suggest_float('lr', 5e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    lr_scheduler = trial.suggest_categorical('lr_scheduler', ['cosine', 'step'])
    warmup_epochs = trial.suggest_int('warmup_epochs', 3, 7)

    tag = f"trial_{trial.number}"
    cmd = [
        "torchrun", f"--nproc_per_node={args.n_gpus}",
        "Main_EyeQuality_tuned_parallel.py",
        "--save_model", tag,
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--weight_decay", str(weight_decay),
        "--lr_scheduler", lr_scheduler,
        "--warmup_epochs", str(warmup_epochs),
        "--epochs", str(args.epochs),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # ex: 1h de folga
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} excedeu o timeout — matando e marcando como pruned.")
        raise optuna.exceptions.TrialPruned()
    if result.returncode != 0:
        print(result.stderr)
        raise optuna.exceptions.TrialPruned()

    metrics_path = f"./result/{tag}_metrics.txt"
    if not os.path.exists(metrics_path):
        raise optuna.exceptions.TrialPruned()

    # ler todas as métricas do arquivo, não só a loss
    val_loss = None
    with open(metrics_path) as f:
        for line in f:
            if line.startswith("Best_Val_Loss"):
                val_loss = float(line.split(":")[1])
            elif line.startswith("Accuracy"):
                trial.set_user_attr("accuracy", float(line.split(":")[1]))
            elif line.startswith("Precision"):
                trial.set_user_attr("precision", float(line.split(":")[1]))
            elif line.startswith("F1"):
                trial.set_user_attr("f1", float(line.split(":")[1]))

    if val_loss is None:
        raise optuna.exceptions.TrialPruned()

    return val_loss

study = optuna.create_study(
    direction="minimize",
    storage=args.storage,
    study_name=args.study_name,
    load_if_exists=True,
)
study.optimize(objective, n_trials=args.n_trials)

df = study.trials_dataframe()
# colunas viram user_attrs_accuracy, user_attrs_precision, user_attrs_f1
print(df[['number', 'value', 'user_attrs_accuracy']].sort_values('user_attrs_accuracy', ascending=False))

if len(study.trials_dataframe(attrs=("state",)).query("state == 'COMPLETE'")) > 0:
    print(study.best_params)
    print(study.best_value)
else:
    print("Nenhum trial completou — confira os PRUNED antes de prosseguir.")
    
df.to_csv("optuna_results.csv", index=False)  # em vez de study.trials_dataframe().to_csv(...)