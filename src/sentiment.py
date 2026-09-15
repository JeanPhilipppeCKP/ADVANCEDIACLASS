import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
# nécessite : nltk.download('vader_lexicon')
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

sia = SentimentIntensityAnalyzer()


def load_and_clean_reviews(path: str) -> pd.DataFrame:
    """Charge le dataset de critiques et retire les entrées inexploitables.

    Fondement : contrairement au preprocessing NLP classique (stopwords,
    stemming), VADER a besoin du texte proche de sa forme naturelle
    (ponctuation, négations, casse intactes) pour bien fonctionner. Le seul
    nettoyage nécessaire ici est structurel : éliminer les lignes qui n'ont
    rien à analyser (valeurs manquantes ou blanches).
    """
    df = pd.read_csv(path, sep="\t")
    before = len(df)

    df = df.dropna(subset=["review"])
    df = df[df["review"].str.strip() != ""]

    after = len(df)
    print(f"Lignes retirées : {before - after} (sur {before} au total)")

    return df.reset_index(drop=True)

def predict_sentiment_bert(review: str) -> str:
    """Classe une critique via un modèle BERT fine-tuné pour le sentiment.

    Fondement : contrairement à VADER (score de mots additionnés), BERT
    traite la phrase entière avec son contexte complet grâce au mécanisme
    d'attention — il peut en théorie mieux gérer la négation, l'ironie et
    les tournures indirectes typiques des critiques longues.
    """
    # BERT a une limite de ~512 tokens : on tronque les critiques trop longues
    result = sentiment_model(review[:512])[0]
    label = result["label"]  # "POSITIVE" ou "NEGATIVE"
    return "pos" if label == "POSITIVE" else "neg"

def evaluate_predictions_bert(df: pd.DataFrame) -> pd.DataFrame:
    """Applique BERT à chaque critique et compare la prédiction au vrai label."""
    df = df.copy()
    df["prediction_bert"] = df["review"].apply(predict_sentiment_bert)
    df["correct_bert"] = df["prediction_bert"] == df["label"]
    return df

def predict_sentiment(review: str) -> str:
    """Calcule le score de polarité VADER d'une critique et le convertit
    en prédiction positive/négative selon le score composé.

    Fondement : le score 'compound' de VADER est normalisé entre -1 et +1.
    Un seuil à 0 (plutôt que d'exiger un score franchement positif) reste
    la convention standard pour une classification binaire pos/neg.
    """
    score = sia.polarity_scores(review)["compound"]
    return "pos" if score > 0.05 else "neg"


def evaluate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Applique VADER à chaque critique et compare la prédiction au vrai
    label déjà fourni dans le dataset.

    Fondement : les labels du dataset sont la vérité terrain (établie par
    des humains). Comparer la prédiction de VADER à cette vérité permet de
    mesurer objectivement sa fiabilité, plutôt que de produire des scores
    sans savoir s'ils sont justes.
    """
    df = df.copy()
    df["prediction"] = df["review"].apply(predict_sentiment)
    df["correct"] = df["prediction"] == df["label"]
    return df


if __name__ == "__main__":
    df = load_and_clean_reviews("data/raw/moviereviews.tsv")

    # --- VADER sur tout le dataset ---
    df = evaluate_predictions(df)
    accuracy = df["correct"].mean()
    print(f"\nPrécision de VADER sur {len(df)} critiques : {accuracy:.2%}")
    print("\nDétail par vrai label (VADER) :")
    print(df.groupby("label")["correct"].mean())

    # --- BERT, d'abord sur un échantillon pour vérifier que ça tourne ---
    """df_sample = df.head(50).copy()
    df_sample = evaluate_predictions_bert(df_sample)
    print(f"\nPrécision de BERT sur l'échantillon (50 critiques) : {df_sample['correct_bert'].mean():.2%}")"""

     # --- BERT sur le dataset complet ---
    df = evaluate_predictions_bert(df)
    accuracy_bert = df["correct_bert"].mean()
    print(f"\nPrécision de BERT sur {len(df)} critiques : {accuracy_bert:.2%}")
    print("\nDétail par vrai label (BERT) :")
    print(df.groupby("label")["correct_bert"].mean())

   

    df.to_csv("data/processed/reviews_with_predictions.csv", index=False)
    print("\nRésultats sauvegardés : data/processed/reviews_with_predictions.csv")