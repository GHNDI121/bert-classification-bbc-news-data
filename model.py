"""
model.py
Chargement du modèle BERT pour la classification de texte.

Fonctions exportées :
  - get_tokenizer   : charge le tokenizer HuggingFace adapté au modèle
  - get_model       : charge BertForSequenceClassification pré-entraîné
  - load_checkpoint : recharge un modèle sauvegardé pour l'inférence
"""

import torch
from transformers import AutoTokenizer, BertForSequenceClassification


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Charge et retourne le tokenizer correspondant au modèle.

    AutoTokenizer s'adapte automatiquement au modèle choisi
    (BERT WordPiece, CamemBERT SentencePiece, etc.).
    
    Retourne
    Tokenizer HuggingFace pré-chargé
    """
    print(f"  Chargement du tokenizer : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_model(
    model_name: str,
    num_labels: int,
    freeze_backbone: bool = False,
) -> BertForSequenceClassification:
    """
    Charge BertForSequenceClassification depuis HuggingFace.

    Le modèle BERT est chargé avec ses poids pré-entraînés.
    Une tête de classification linéaire (hidden_size → num_labels)
    est ajoutée automatiquement et initialisée aléatoirement.

    Stratégie de gel (freeze_backbone) :
    ─────────────────────────────────────────────────────────────
    False (défaut) — Fine-tuning complet :
        Tous les paramètres BERT + tête sont entraînés.
        Protection des poids pré-entraînés assurée par le LR faible (2e-5 – 5e-5).
        → Recommandé pour BERT : meilleure performance, convergence rapide (3-5 epochs).

    True — Feature extraction :
        Le backbone BERT (bert.embeddings + bert.encoder) est gelé (requires_grad=False).
        Seule la tête de classification (classifier.*) est entraînée.
        → Plus rapide, moins de VRAM, mais performances généralement inférieures.
        → Utile si dataset très petit ou ressources très limitées.
    ─────────────────────────────────────────────────────────────

    Paramètres
    ----------
    model_name      : identifiant HuggingFace du modèle
    num_labels      : nombre de classes de sortie
    freeze_backbone : si True, gèle le backbone BERT (ne garde que la tête entraînable)

    Retourne
    --------
    BertForSequenceClassification prêt pour le fine-tuning
    """
    print(f"  Chargement du modèle : {model_name} ({num_labels} classes)")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # Ignore les poids incompatibles de la tête pré-entraînée
    )

    if freeze_backbone:
        # Gèle tous les paramètres du backbone BERT (embeddings + 12 couches encoder)
        # Seuls les paramètres du classifier (tête linéaire) resteront entraînables
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False  # Gèle ce paramètre

        # Affiche un résumé du gel
        trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total       = sum(p.numel() for p in model.parameters())
        print(f"  Backbone gelé — paramètres entraînables : {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")
    else:
        # Fine-tuning complet : tous les paramètres sont entraînables
        # Le LR faible (3e-5) protège les poids pré-entraînés du catastrophic forgetting
        print(f"  Fine-tuning complet (backbone non gelé) — LR faible requis (≤ 5e-5)")

    return model


def load_checkpoint(
    checkpoint_path: str,
    model_name: str,
    num_labels: int,
    device: torch.device,
) -> BertForSequenceClassification:
    """
    Recharge un modèle sauvegardé depuis un fichier .pt pour l'inférence.
    Retourne
    Modèle en mode évaluation (model.eval()) placé sur device
    """
    print(f"  Chargement du checkpoint : {checkpoint_path}")
    model = get_model(model_name, num_labels)

    # map_location permet de charger un modèle entraîné sur GPU vers CPU
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Mode évaluation : désactive dropout → prédictions déterministes
    return model
