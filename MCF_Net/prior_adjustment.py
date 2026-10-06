"""
Ajuste de prior / logit adjustment pós-hoc para o MCF-Net (EyeQ).
Não altera pesos nem retreina: soma um vetor fixo de 3 valores aos log-probs salvos.

O MCF-Net treina com BCE e saída sigmoide independente por classe (não softmax).
Aqui a saída s é convertida em logit(s) e tratada como logit de um softmax; a
temperatura (ajustada na validação) absorve a escala. O argmax é preservado
(logit é monotônico), então "sem_ajuste" reproduz a predição original.

Entradas (geradas pelo dump_scores.py), dentro de --dir:
  val_scores.npy, val_labels.npy, test_scores.npy, test_labels.npy
Rótulos: 0 = Good, 1 = Usable, 2 = Reject

Para cada método exporta, em <dir>/<método>_*, o mesmo conjunto do evaluate.py
(_metrics.txt, _roc.pdf, _confusion_matrix.csv/.png) e acrescenta uma linha em
<dir>/metrics_history.csv. "sem_ajuste" usa a saída sigmoide original, então
reproduz os números do evaluate.py; os demais usam softmax(log-probs + ajuste).

Uso (a partir de MCF_Net/):
  python prior_adjustment.py --dir result/DenseNet121_v3_tuned_scores
"""
import argparse
import csv
import json
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # no display on a SLURM/cluster node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import (accuracy_score, auc, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

from utils.metric import compute_metric, save_confusion_matrix

CLASSES = ["Good", "Usable", "Reject"]
EPS = 1e-6  # evita logit infinito quando a sigmoide float32 satura


def prior_from_labels(y):
    counts = np.bincount(y, minlength=3)
    return counts / counts.sum()


def scores_to_logits(scores):
    s = np.clip(scores.astype(np.float64), EPS, 1 - EPS)
    return np.log(s) - np.log1p(-s)


def fit_temperature(logits, labels):
    """Temperature scaling na validação: deixa as probabilidades calibradas,
    o que importa para o EM e para a escala do ajuste."""
    def nll(t):
        lp = log_softmax(logits / t, axis=1)
        return -lp[np.arange(len(labels)), labels].mean()
    return minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x


def em_prior(probs, prior_train, iters=1000, tol=1e-8):
    """Saerens et al. (2002): estima o prior do conjunto-alvo SEM usar rótulos."""
    prior = prior_train.copy()
    for _ in range(iters):
        p = probs * (prior / prior_train)
        p /= p.sum(axis=1, keepdims=True)
        new = p.mean(axis=0)
        if np.abs(new - prior).max() < tol:
            return new
        prior = new
    return prior


def evaluate(logp, labels):
    probs = softmax(logp, axis=1)
    pred = probs.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, pred, labels=[0, 1, 2], zero_division=0)
    per_class = {}
    for k, c in enumerate(CLASSES):
        per_class[c] = {
            "acc_ovr": float(((pred == k) == (labels == k)).mean()),
            "prec": float(prec[k]),
            "sens": float(rec[k]),
            "f1": float(f1[k]),
            "auc": float(roc_auc_score(labels == k, probs[:, k])),
        }
    return {
        "acc_global": float(accuracy_score(labels, pred)),
        "acc_media_ovr": float(np.mean([v["acc_ovr"] for v in per_class.values()])),
        "f1_macro": float(f1_score(labels, pred, average="macro")),
        "kappa_quad": float(cohen_kappa_score(labels, pred, weights="quadratic")),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
    }


def export_method(name, probs, labels, out_dir, model_name):
    """Mesmas saídas do evaluate.py para um método (probs: (N, 3) por classe)."""
    slug = re.sub(r"[^A-Za-z0-9._=-]+", "_", name).strip("_")
    prefix = os.path.join(out_dir, slug)
    title = f"{model_name} [{name}]"
    report = compute_metric(labels, probs, target_names=CLASSES)

    mean_accuracy = float(report["Accuracy"])
    mean_precision = float(np.mean(report["Precision"]))
    mean_sensitivity = float(np.mean(report["Sensitivity"]))
    mean_f1 = float(report["macro-F1"])
    mean_auc_macro = float(report["AUC"])

    with open(prefix + "_metrics.txt", "w") as f:
        f.write(f"Method      : {name}\n")
        f.write(f"Accuracy    : {mean_accuracy:.4f}\n")
        f.write(f"Precision   : {mean_precision:.4f}\n")
        f.write(f"Sensitivity : {mean_sensitivity:.4f}\n")
        f.write(f"F1          : {mean_f1:.4f}\n")
        f.write(f"Kappa       : {float(report['Kappa']):.4f}\n")

    # ROC (PDF): curvas por classe vêm do próprio compute_metric
    y_bin = np.stack([(labels == i).astype(int) for i in range(len(CLASSES))], axis=1)
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    auc_per_class = report["AUC_per_class"].ravel()
    roc = report["ROC_curve"]

    plt.figure(figsize=(7, 7))
    for i, color in enumerate(["#2ca02c", "#ff7f0e", "#d62728"]):
        plt.plot(roc[f"ROC_fpr_{i}"], roc[f"ROC_tpr_{i}"], color=color, lw=2,
                 label=f"{CLASSES[i]} (AUC = {auc_per_class[i]:.3f})")
    plt.plot(fpr_micro, tpr_micro, color="deeppink", linestyle=":", lw=2,
             label=f"micro-average (AUC = {auc_micro:.3f})")
    plt.plot([], [], " ", label=f"macro-average AUC = {mean_auc_macro:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {title}")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(prefix + "_roc.pdf", format="pdf")
    plt.close()

    save_confusion_matrix(labels, probs.argmax(axis=1), CLASSES, prefix,
                          title=f"Confusion matrix — {title}")

    history = os.path.join(out_dir, "metrics_history.csv")
    new_file = not os.path.isfile(history)
    with open(history, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "model", "accuracy", "precision",
                             "recall", "f1", "auc_micro", "auc_macro"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), title,
                         f"{mean_accuracy:.4f}", f"{mean_precision:.4f}",
                         f"{mean_sensitivity:.4f}", f"{mean_f1:.4f}",
                         f"{auc_micro:.4f}", f"{mean_auc_macro:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="pasta gerada pelo dump_scores.py")
    ap.add_argument("--train-csv", default="../data/Label_EyeQ_patient_stratified_train.csv",
                    help="CSV do treino EFETIVAMENTE usado (fonte do prior de treino)")
    ap.add_argument("--label-col", default="quality")
    ap.add_argument("--no-temp", action="store_true", help="desliga temperature scaling")
    ap.add_argument("--model-name", default=None,
                    help="rótulo nos títulos/histórico (default: nome da pasta --dir)")
    ap.add_argument("--out", default=None, help="default: <dir>/prior_adjustment_results.json")
    args = ap.parse_args()

    model_name = args.model_name or os.path.basename(os.path.normpath(args.dir))
    load = lambda name: np.load(os.path.join(args.dir, name))
    val_y, test_y = load("val_labels.npy").astype(int), load("test_labels.npy").astype(int)
    test_scores = load("test_scores.npy")
    val_z, test_z = scores_to_logits(load("val_scores.npy")), scores_to_logits(test_scores)

    prior_train = prior_from_labels(pd.read_csv(args.train_csv)[args.label_col].to_numpy())
    log_pt = np.log(prior_train)

    t = 1.0 if args.no_temp else fit_temperature(val_z, val_y)
    val_lp = log_softmax(val_z / t, axis=1)
    test_lp = log_softmax(test_z / t, axis=1)

    # 1) Logit adjustment para alvo balanceado; tau escolhido na validação (F1 macro)
    taus = np.round(np.arange(0.0, 2.01, 0.05), 2)
    scores = [evaluate(val_lp - tau * log_pt, val_y)["f1_macro"] for tau in taus]
    best_tau = float(taus[int(np.argmax(scores))])

    # 2) Prior do teste estimado por EM (sem rótulos); na validação deve voltar ~ao prior do treino
    prior_em_val = em_prior(np.exp(val_lp), prior_train)
    prior_em_test = em_prior(np.exp(test_lp), prior_train)

    # 3) Oráculo: prior real do teste. SÓ referência de teto, usa rótulos do teste.
    prior_oracle = prior_from_labels(test_y)

    adjustments = {
        "sem_ajuste": np.zeros(3),
        f"logit_adj_tau={best_tau}": -best_tau * log_pt,
        "em_prior_teste": np.log(prior_em_test) - log_pt,
        "oraculo_prior_teste (ref.)": np.log(prior_oracle) - log_pt,
    }

    print(f"Temperatura: {t:.3f}")
    print("Prior treino   :", np.round(prior_train, 4))
    print("EM na validação:", np.round(prior_em_val, 4), "(sanidade: ~prior treino)")
    print("EM no teste    :", np.round(prior_em_test, 4))
    print("Prior real teste:", np.round(prior_oracle, 4))
    print()
    header = ["método", "sens U", "prec U", "F1 U", "F1 mac", "acc", "acc ovr", "kappa"]
    print(f"{header[0]:<30}" + "".join(f"{h:>9}" for h in header[1:]))

    results = {"temperature": t, "prior_train": prior_train.tolist(),
               "prior_em_test": prior_em_test.tolist(), "best_tau": best_tau, "methods": {}}
    for name, adj in adjustments.items():
        r = evaluate(test_lp + adj, test_y)
        r["adjustment"] = adj.tolist()
        results["methods"][name] = r
        probs = test_scores if name == "sem_ajuste" else softmax(test_lp + adj, axis=1)
        export_method(name, probs, test_y, args.dir, model_name)
        u = r["per_class"]["Usable"]
        row = [u["sens"], u["prec"], u["f1"], r["f1_macro"], r["acc_global"],
               r["acc_media_ovr"], r["kappa_quad"]]
        print(f"{name:<30}" + "".join(f"{v:>9.4f}" for v in row))

    out = args.out or os.path.join(args.dir, "prior_adjustment_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados completos em {out}")
    print(f"Por método (_metrics.txt, _roc.pdf, _confusion_matrix.*) e metrics_history.csv em {args.dir}")


if __name__ == "__main__":
    main()
