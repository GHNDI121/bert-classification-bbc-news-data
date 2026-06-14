"""
utils.py
Fonctions utilitaires pour le projet BERT :
  - set_seed             : reproductibilité (random, numpy, torch)
  - compute_metrics      : accuracy + F1-score macro
  - count_parameters     : nombre de paramètres entraînables
  - plot_curves          : courbes loss / accuracy par epoch
  - plot_confusion_matrix: matrice de confusion 
"""

import random
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# Reproductibilité

def set_seed(seed: int = 42) -> None:
    """
    Fixe toutes les seeds pour rendre l'entraînement reproductible.
    À appeler au tout début de train.py, avant toute opération aléatoire.

    Paramètres
    seed : int — valeur de la graine (par défaut 42)
    """
    random.seed(seed)                          # Seed Python standard
    np.random.seed(seed)                       # Seed NumPy
    torch.manual_seed(seed)                    # Seed Torch CPU
    torch.cuda.manual_seed_all(seed)           # Seed Torch GPU (multi-GPU)
    torch.backends.cudnn.deterministic = True  # Reproductibilité CUDA (plus lent)
    torch.backends.cudnn.benchmark = False     # Désactive l'optimisation auto


# Métriques

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Calcule l'accuracy et le F1-score macro.

    Paramètres
    y_true : liste des labels réels (entiers)
    y_pred : liste des labels prédits (entiers)

    Retourne
    dict avec les clés 'accuracy' et 'f1'
    """
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "f1": f1}


def count_parameters(model: torch.nn.Module) -> int:
    """
    Compte et affiche le nombre de paramètres entraînables du modèle.

    Paramètres
    model : nn.Module

    Retourne
    int — nombre total de paramètres entraînables
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres entraînables : {total:,}")
    return total


# Visualisations

def plot_curves(history: dict, save_path: str = "training_curves.png") -> None:
    """
    Trace et sauvegarde les courbes loss / accuracy par epoch.

    Paramètres
    history   : dict avec les clés train_loss, val_loss, train_acc, val_acc
    save_path : chemin de sauvegarde de l'image
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Courbe Loss ---
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss par epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Courbe Accuracy ---
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Accuracy")
    axes[1].set_title("Accuracy par epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Courbes sauvegardées : {save_path}")


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    model_name: str = "BERT",
    save_path: Optional[str] = None,
) -> None:
    """
    Génère et sauvegarde la matrice de confusion.

    Paramètres
    y_true       : labels réels
    y_pred       : labels prédits
    class_names  : noms des classes dans l'ordre des indices
    model_name   : nom du modèle (affiché dans le titre)
    save_path    : chemin de sauvegarde (optionnel)
    """
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {model_name}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Matrice de confusion sauvegardée : {save_path}")

    plt.close()
