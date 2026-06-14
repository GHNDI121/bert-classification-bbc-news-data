"""
dataset.py
Classe TextClassificationDataset : dataset PyTorch personnalisé pour la
classification de texte avec un tokenizer BERT.

Chaque exemple est tokenisé à la volée dans __getitem__ et retourne :
  - input_ids      : identifiants des tokens (padding inclus)
  - attention_mask : 1 pour tokens réels, 0 pour le padding
  - label          : entier représentant la classe

PERSONNALISATION :
  - MAX_LENGTH : à configurer dans train.py selon la longueur de vos textes
    (128 si textes courts, 256 si textes longs — justifier dans le README)
"""

from typing import List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TextClassificationDataset(Dataset):
    """Dataset PyTorch pour la classification de texte avec BERT.

    Paramètres
    ----------
    texts      : liste de chaînes de caractères (les textes bruts)
    labels     : liste d'entiers (les labels encodés numériquement)
    tokenizer  : tokenizer HuggingFace pré-chargé (ex. BertTokenizer)
    max_length : longueur maximale de tokenization (padding / truncation)
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ) -> None:
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Retourne le nombre d'exemples dans le dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Retourne un exemple tokenisé sous forme de tenseurs PyTorch.

        Paramètres
        ----------
        idx : index de l'exemple

        Retourne
        --------
        dict avec les clés 'input_ids', 'attention_mask', 'label'
        """
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenization avec padding et troncature automatiques
        # IMPORTANT : ne pas oublier return_attention_mask=True
        # (oublier le masque d'attention dégrade significativement les résultats)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",       # Rembourre jusqu'à max_length avec [PAD]
            truncation=True,            # Tronque si le texte dépasse max_length
            return_attention_mask=True, # 1 = token réel, 0 = padding
            return_tensors="pt",        # Retourne des tenseurs PyTorch
        )

        return {
            # Squeeze : (1, max_length) → (max_length,) pour le batching par DataLoader
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),  # CrossEntropyLoss attend torch.long
        }
