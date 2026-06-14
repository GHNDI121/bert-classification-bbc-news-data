"""
train.py:
Pipeline principal du projet BERT — Classification de texte.

Dataset : configurable via la section CONFIG ci-dessous
  - TEXT_COL  : colonne texte  (ex. "content", "case_text")
  - LABEL_COL : colonne label  (ex. "category", "case_outcome")

Fonctions exportées :
  - train_epoch : une epoch d'entraînement (forward + backward + clip + optim)
  - eval_epoch  : une epoch de validation (torch.no_grad, mode eval)
  - main        : pipeline complet (chargement → entraînement → sauvegarde)

PERSONNALISATION — modifiez uniquement la section CONFIG :
  - DATA_PATH  : chemin vers votre fichier CSV
  - TEXT_COL   : nom de la colonne texte       (ex. "case_text", "content")
  - LABEL_COL  : nom de la colonne label       (ex. "case_outcome", "category")
  - SEP        : séparateur CSV                (",", "\\t", ";")
  - MODEL_NAME : modèle HuggingFace            (voir commentaires inline)
  - MAX_LENGTH : longueur max de tokenization  (128 ou 256)
  - BATCH_SIZE : taille du batch               (16 ou 32 selon VRAM)
  - NUM_EPOCHS : nombre d'epochs               (3 à 5 pour BERT)
  - LR         : learning rate                 (2e-5 à 5e-5)
  - SEED       : graine aléatoire
"""

import os
import json
import platform
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import wandb

from dataset import TextClassificationDataset
from model   import get_tokenizer, get_model
from utils   import set_seed, compute_metrics, count_parameters, plot_curves, plot_confusion_matrix


# ===========================================================================
# ⚙️  SECTION CONFIG — Adaptez ces valeurs à votre dataset
# ===========================================================================
DATA_PATH    = "data/bbc-news-data.csv"   # Chemin vers le CSV
TEXT_COL     = "content"                   # ← Remplacez par "case_text" si besoin
LABEL_COL    = "category"                 # ← Remplacez par "case_outcome" si besoin
SEP          = "\t"                        # Séparateur : "\t" (TSV), "," (CSV), ";" (Excel FR)

MODEL_NAME   = "bert-base-uncased"        # Anglais — remplacer par "camembert-base" (FR)
                                           #           ou "bert-base-multilingual-cased" (multi)
MAX_LENGTH   = 256                        # Justification : couvre ~95% des textes BBC (~379 mots)
BATCH_SIZE   = 16                         # Réduire à 8 si OOM (Out Of Memory)
NUM_EPOCHS   = 4                          # BERT converge vite — 3 à 5 epochs suffisent
LR           = 3e-5                       # Zone recommandée : 2e-5 – 5e-5
WEIGHT_DECAY = 0.01                       # Régularisation L2 dans AdamW
WARMUP_RATIO = 0.1                        # 10% des steps pour le warmup linéaire
SEED         = 42                         # Graine pour la reproductibilité

# Stratégie de gel du backbone :
#   False (défaut) — Fine-tuning complet : tous les paramètres BERT s'entraînent.
#                    Le LR faible (3e-5) protège les poids pré-entraînés.
#                    → Standard pour BERT, meilleures performances.
#   True           — Feature extraction : seule la tête de classification s'entraîne.
#                    Plus rapide et économe en VRAM, mais performances inférieures.
#                    → À utiliser si dataset très petit ou ressources très limitées.
FREEZE_BACKBONE = False

CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_BEST = os.path.join(CHECKPOINT_DIR, "best_model.pt")    # Meilleur modèle (val_loss)
CHECKPOINT_RESUME = os.path.join(CHECKPOINT_DIR, "resume.pt")       # Reprise d'entraînement
LABEL_MAP_PATH  = "label_map.json"        # Mapping classes → indices (utilisé par demo.py)

# Weights & Biases (logging expérimental — mettre WANDB_MODE="disabled" pour désactiver)
WANDB_PROJECT = "BERT_Classification"
WANDB_ENTITY  = None                      # Votre entité wandb (ou None)
WANDB_MODE    = "online"                  # "online" | "offline" | "disabled"
# ===========================================================================


def train_epoch(
    model: nn.Module, 
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Effectue une epoch d'entraînement complète.

    Étapes par batch :
      1. Transfert sur device
      2. Remise à zéro des gradients
      3. Forward pass → logits
      4. Calcul de la loss (CrossEntropyLoss)
      5. Backward pass → gradients
      6. Clip des gradients (max_norm=1.0) — important pour BERT
      7. Mise à jour des poids (optimizer.step)
      8. Mise à jour du scheduler
    Retourne
    (avg_loss, accuracy, f1_macro) — tuple de floats
    """
    model.train()  # Active dropout et batch norm en mode entraînement
    total_loss = 0.0
    all_preds, all_labels = [], []

    loop = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for batch in loop:
        # Transfert des tenseurs sur le périphérique cible
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)  # torch.long requis par CrossEntropyLoss

        optimizer.zero_grad()  # Remet les gradients à zéro avant chaque batch

        # Forward pass — BERT retourne un objet SequenceClassifierOutput
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits  # (batch_size, num_labels) — scores bruts avant softmax

        # CrossEntropyLoss attend des logits (pas de softmax) et des labels entiers (pas one-hot)
        loss = criterion(logits, labels)

        loss.backward()  # Calcul des gradients par rétropropagation

        # Clip des gradients pour éviter l'explosion — bonne pratique pour le fine-tuning BERT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()   # Mise à jour des poids
        scheduler.step()   # Mise à jour du learning rate

        total_loss += loss.item()

        # Prédiction : indice de la valeur maximale des logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics["accuracy"], metrics["f1"]


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, list, list]:
    """
    Effectue une epoch d'évaluation sans mise à jour des poids.

    Différences importantes par rapport à train_epoch :
      - model.eval() : désactive le dropout → prédictions déterministes
      - torch.no_grad() : pas de calcul de gradient → plus rapide, moins de VRAM
      - Pas d'appel à optimizer.step() ni scheduler.step()
    Retourne
    (avg_loss, accuracy, f1_macro, all_preds, all_labels)
    """
    model.eval()  # IMPORTANT : désactive le dropout pour des prédictions reproductibles
    total_loss = 0.0
    all_preds, all_labels = [], []

    # torch.no_grad() désactive le calcul des gradients → économie mémoire et temps
    with torch.no_grad():
        loop = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
        for batch in loop:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics["accuracy"], metrics["f1"], all_preds, all_labels


def main():
    """
    Point d'entrée principal.

    Étapes :
      1. Reproductibilité & device
      2. Chargement et inspection du dataset
      3. Encodage des labels + split train/val 80/20 stratifié
      4. Tokenizer + Datasets + DataLoaders
      5. Modèle + optimiseur AdamW + scheduler warmup linéaire + loss
      6. Boucle d'entraînement avec reprise de checkpoint
      7. Visualisations (courbes + matrice de confusion)
    """

    # 1. Reproductibilité
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    # 2. Chargement et inspection du dataset
    print("\n" + "="*60)
    print("  INSPECTION DU DATASET")
    print("="*60)

    df = pd.read_csv(DATA_PATH, sep=SEP)

    # Vérification des colonnes configurées
    assert TEXT_COL  in df.columns, f"Colonne '{TEXT_COL}' introuvable. Colonnes : {df.columns.tolist()}"
    assert LABEL_COL in df.columns, f"Colonne '{LABEL_COL}' introuvable. Colonnes : {df.columns.tolist()}"

    df = df[[TEXT_COL, LABEL_COL]].dropna()  # Supprime les lignes avec valeurs manquantes

    print(f"\n  Fichier      : {DATA_PATH}")
    print(f"  Nb exemples  : {len(df)}")
    print(f"  Distribution des classes :")
    for cls, cnt in df[LABEL_COL].value_counts().items():
        print(f"    {cls:20s} : {cnt:5d} ({cnt/len(df)*100:.1f}%)")

    # Longueur des textes (en mots) — aide à choisir max_length
    df["_n_words"] = df[TEXT_COL].str.split().str.len()
    print(f"\n  Longueur des textes (mots) :")
    print(f"    min={df['_n_words'].min()}  "
          f"moyenne={df['_n_words'].mean():.0f}  "
          f"max={df['_n_words'].max()}")
    print(f"    → max_length={MAX_LENGTH} choisi")

    # Affichage de 5 exemples
    print(f"\n  5 exemples :")
    for _, row in df.sample(5, random_state=SEED).iterrows():
        preview = str(row[TEXT_COL])[:80].replace("\n", " ")
        print(f"    [{row[LABEL_COL]}] {preview}...")

    df = df.drop(columns=["_n_words"])  # Nettoyage colonne temporaire

    # 3. Encodage des labels + split stratifié 80/20
    print("\n" + "="*60)
    print("  ENCODAGE & SPLIT")
    print("="*60)

    # LabelEncoder : chaîne → entier (tri alphabétique → reproductible)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df[LABEL_COL])
    class_names = le.classes_.tolist()
    num_labels  = len(class_names)
    print(f"\n  Classes ({num_labels}) : {class_names}")

    # Split stratifié : garantit les mêmes proportions de classes dans train et val
    X_train, X_val, y_train, y_val = train_test_split(
        df[TEXT_COL].tolist(),
        df["label_enc"].tolist(),
        test_size=0.20,
        random_state=SEED,
        stratify=df["label_enc"],  # Stratification obligatoire (consigne section 3.2)
    )
    print(f"  Train : {len(X_train)} | Val : {len(X_val)}")

    # Sauvegarde du mapping pour demo.py
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump({
            "class_names": class_names,
            "model_name":  MODEL_NAME,
            "max_length":  MAX_LENGTH,
            "num_labels":  num_labels,
        }, f, indent=2)
    print(f"  label_map.json sauvegardé → {LABEL_MAP_PATH}")

    # 4. Tokenizer + Datasets + DataLoaders
    print("\n" + "="*60)
    print("  TOKENIZER & DATALOADERS")
    print("="*60)

    tokenizer     = get_tokenizer(MODEL_NAME)
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset   = TextClassificationDataset(X_val,   y_val,   tokenizer, MAX_LENGTH)

    # num_workers=0 sur Windows pour éviter les erreurs de multiprocessing
    # Passer à 2 ou 4 sur Linux/macOS pour accélérer le chargement des batches
    nw = 0 if platform.system() == "Windows" else 2

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    print(f"  Batches — Train : {len(train_loader)} | Val : {len(val_loader)}")

   
    # 5. Modèle, optimiseur, scheduler, loss
    print("\n" + "="*60)
    print("  MODÈLE BERT")
    print("="*60)

    model = get_model(MODEL_NAME, num_labels, freeze_backbone=FREEZE_BACKBONE).to(device)
    count_parameters(model)  # Affiche le nb de paramètres entraînables

    # AdamW avec weight_decay — filter pour n'optimiser que les params entraînables
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler linéaire avec warmup : LR monte progressivement puis descend
    # Le warmup évite les mises à jour destructrices au début du fine-tuning
    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # CrossEntropyLoss : attend des logits bruts (pas de softmax) et des labels entiers
    criterion = nn.CrossEntropyLoss()

    # 6. Initialisation wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"bert_{num_labels}classes",
        config={
            "model":        MODEL_NAME,
            "max_length":   MAX_LENGTH,
            "batch_size":   BATCH_SIZE,
            "epochs":       NUM_EPOCHS,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "seed":         SEED,
            "dataset":      DATA_PATH,
            "num_labels":   num_labels,
        },
        reinit=True,
        mode=WANDB_MODE,
    )
    wandb.watch(model, log="gradients", log_freq=50)  # Surveille les gradients

    # 7. Boucle d'entraînement avec reprise de checkpoint
    print("\n" + "="*60)
    print(f"  ENTRAÎNEMENT ({NUM_EPOCHS} EPOCHS)")
    print("="*60)

    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    start_epoch   = 1

    # Reprise depuis un checkpoint de résumé si disponible
    # Permet de reprendre l'entraînement là où il s'est arrêté
    if os.path.exists(CHECKPOINT_RESUME):
        print(f"\n  ↩ Checkpoint de reprise trouvé : {CHECKPOINT_RESUME}")
        ckpt = torch.load(CHECKPOINT_RESUME, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        history       = ckpt.get("history", history)
        print(f"  ↩ Reprise depuis l'epoch {start_epoch} (meilleure val_loss={best_val_loss:.4f})")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n  Epoch {epoch}/{NUM_EPOCHS}")

        # Entraînement
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validation — IMPORTANT : basculer en mode eval + désactiver gradients
        val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(
            model, val_loader, criterion, device
        )

        # Historique pour les courbes
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        # Log wandb
        wandb.log({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "train_accuracy": train_acc,
            "train_f1_score": train_f1,
            "val_loss":       val_loss,
            "val_accuracy":   val_acc,
            "val_f1_score":   val_f1,
            "learning_rate":  current_lr,
        })

        print(
            f"    train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}  lr={current_lr:.2e}"
        )

        # Sauvegarde du checkpoint de reprise à chaque epoch (écrase le précédent)
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss":   best_val_loss,
            "history":         history,
        }, CHECKPOINT_RESUME)

        # Sauvegarde du meilleur modèle (critère : val_loss minimale)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_BEST)
            wandb.save(CHECKPOINT_BEST)
            print(f"    ✓ Meilleur modèle sauvegardé (val_loss={best_val_loss:.4f})")

    # Suppression du checkpoint de reprise : entraînement terminé
    if os.path.exists(CHECKPOINT_RESUME):
        os.remove(CHECKPOINT_RESUME)

    # 8. Visualisations post-entraînement
    print("\n" + "="*60)
    print("  VISUALISATIONS")
    print("="*60)

    curves_path = os.path.join(CHECKPOINT_DIR, "training_curves.png")
    cm_path     = os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")

    plot_curves(history, save_path=curves_path)
    plot_confusion_matrix(
        val_labels, val_preds, class_names,
        model_name="BERT fine-tuned",
        save_path=cm_path,
    )

    # Log des figures dans wandb
    wandb.log({
        "training_curves":  wandb.Image(curves_path),
        "confusion_matrix": wandb.Image(cm_path),
    })

    wandb.finish()

    print("\n" + "="*60)
    print(f"  ✓ Entraînement terminé. Meilleur val_loss = {best_val_loss:.4f}")
    print(f"  Checkpoints : {CHECKPOINT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
