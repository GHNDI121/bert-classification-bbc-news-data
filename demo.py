"""
# demo.py — Interface de démonstration Gradio pour la classification BERT

Lance une interface web interactive qui :
  - Accepte un texte saisi par l'utilisateur
  - Affiche la classe prédite et les probabilités de chaque classe

PERSONNALISATION :
    - CHECKPOINT : chemin vers le fichier .pt sauvegardé par train.py
    - LABEL_MAP  : chemin vers label_map.json généré par train.py
    - Modifiez TITLE / DESCRIPTION pour adapter l'interface à votre dataset
    - Modifiez EXAMPLES avec 2+ exemples représentatifs de votre dataset
"""
import json
import os
import torch
import torch.nn.functional as F
import gradio as gr

from model import get_tokenizer, load_checkpoint


#  SECTION CONFIG
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT  = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
LABEL_MAP   = os.path.join(BASE_DIR, "label_map.json")
SHARE       = False

# Titre et description de l'interface (à adapter à votre dataset)
TITLE       = "Classification de texte avec BERT"
DESCRIPTION = (
    "Fine-tuning de `bert-base-uncased` sur le dataset BBC News (5 catégories).\n\n"
    "Saisissez un texte en anglais et le modèle prédit sa catégorie "
    "parmi : **Business, Entertainment, Politics, Sport, Tech**."
)

# Exemples pré-remplis (minimum 2 requis)
# ← Remplacez par des exemples de VOTRE dataset
EXAMPLES = [
    ["The stock market saw significant gains today as technology companies reported "
     "strong quarterly earnings, boosting investor confidence across all sectors."],
    ["The national football team secured a dramatic victory in the championship "
     "final, with the winning goal scored in the last minute of extra time."],
    ["A new artificial intelligence system has been developed that can diagnose "
     "rare diseases from medical images with greater accuracy than human doctors."],
    ["Parliament voted to approve the new budget proposal, with members debating "
     "spending cuts to healthcare and education funding."],
]



def load_resources():
    """Charge le modèle, le tokenizer et le mapping des classes depuis le disque.

    Returns:
        Tuple (model, tokenizer, class_names, max_length, device).
    """
    # Lecture du label_map généré à la fin de train.py
    with open(LABEL_MAP, "r") as f:
        cfg = json.load(f)

    class_names = cfg["class_names"]
    model_name  = cfg["model_name"]
    max_length  = cfg["max_length"]
    num_labels  = cfg["num_labels"]

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(model_name)
    model     = load_checkpoint(CHECKPOINT, model_name, num_labels, device)

    print(f"[demo] Modèle chargé sur {device} — {num_labels} classes : {class_names}")
    return model, tokenizer, class_names, max_length, device


def predict(text: str, model, tokenizer, class_names: list,
            max_length: int, device: torch.device) -> dict:
    """Prédit la classe d'un texte et retourne les probabilités.

    Args:
        text        : Texte brut saisi par l'utilisateur.
        model       : Modèle BERT chargé.
        tokenizer   : Tokenizer correspondant.
        class_names : Liste des noms de classes.
        max_length  : Longueur max de tokenization.
        device      : Périphérique.

    Returns:
        Dictionnaire {classe: probabilité} compatible avec gr.Label.
    """
    if not text or not text.strip():
        return {c: 0.0 for c in class_names}

    # Tokenization du texte d'entrée
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inférence sans calcul de gradients
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits  # (1, num_labels)

    # Softmax pour obtenir des probabilités entre 0 et 1
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    # Format attendu par gr.Label : {nom_classe: probabilité}
    return {class_names[i]: round(probs[i], 4) for i in range(len(class_names))}


def build_interface():
    """Construit et lance l'interface Gradio.

    L'interface affiche :
    - Un champ texte pour la saisie
    - Un composant Label montrant la classe prédite et les probabilités
    """
    # Chargement des ressources une seule fois au démarrage
    model, tokenizer, class_names, max_length, device = load_resources()

    # Fonction wrapper pour Gradio (ne prend qu'un seul argument : le texte)
    def classify(text: str) -> dict:
        """Wrapper appelé par Gradio à chaque soumission."""
        return predict(text, model, tokenizer, class_names, max_length, device)

    # Construction de l'interface
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=2):
                # Zone de saisie du texte
                text_input = gr.Textbox(
                    label="Entrez votre texte ici",
                    placeholder="Collez ou tapez un texte à classifier...",
                    lines=6,
                )
                submit_btn = gr.Button("Classifier", variant="primary")

            with gr.Column(scale=1):
                # Affiche la classe prédite + les probabilités sous forme de barres
                label_output = gr.Label(
                    label="Prédiction (probabilités par classe)",
                    num_top_classes=len(class_names),
                )

        # Exemples pré-remplis (au moins 2 requis par le sujet)
        gr.Examples(
            examples=EXAMPLES,
            inputs=text_input,
            outputs=label_output,
            fn=classify,
            cache_examples=False,  # Ne pas mettre en cache pour éviter les erreurs
            label="Exemples",
        )

        # Déclencheurs : bouton ou touche Entrée
        submit_btn.click(fn=classify, inputs=text_input, outputs=label_output)
        text_input.submit(fn=classify, inputs=text_input, outputs=label_output)

    return demo


if __name__ == "__main__":
    # Lance le serveur Gradio sur http://localhost:7860
    # share=True génère un lien public (utile sur Google Colab)
    interface = build_interface()
    interface.launch(share=SHARE)
