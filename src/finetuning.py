import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"  # modèle léger, pas encore fine-tuné pour le sentiment


def inspect_dataset(df: pd.DataFrame) -> None:
    """Affiche un diagnostic du dataset avant nettoyage."""
    print("=== Diagnostic avant nettoyage ===")
    print(f"Lignes totales : {len(df)}")
    print(f"\nValeurs manquantes par colonne :\n{df.isnull().sum()}")
    print(f"\nDoublons exacts sur 'text' : {df.duplicated(subset=['text']).sum()}")
    non_null_text = df["text"].dropna()
    print(f"\nTextes de moins de 50 caractères : {(non_null_text.str.len() < 50).sum()}")
    print("=" * 35)


def load_and_clean_fake_news(path: str, min_length: int = 50) -> pd.DataFrame:
    """Charge et nettoie le dataset fake news (voir fondement détaillé
    dans les échanges précédents : pas de stemming/lemmatization/stopwords,
    uniquement du nettoyage structurel)."""
    df = pd.read_excel(path)
    before = len(df)

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() >= min_length]

    after = len(df)
    print(f"\nLignes retirées : {before - after} (sur {before} au total)")
    print(f"Répartition finale des labels :\n{df['label'].value_counts()}")

    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Sépare en train/test en préservant la proportion de chaque classe."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    print(f"Train : {len(train_df)} lignes ({train_df['label'].value_counts().to_dict()})")
    print(f"Test  : {len(test_df)} lignes ({test_df['label'].value_counts().to_dict()})")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Convertit un DataFrame pandas en Dataset Hugging Face tokenisé.

    Fondement : BERT n'accepte pas du texte brut directement — il faut le
    convertir en identifiants numériques (tokens) via le même tokenizer
    que celui utilisé pendant son pré-entraînement, pour que le vocabulaire
    corresponde exactement à ce que le modèle a appris à comprendre.
    Truncation à 128 tokens : la limite technique du modèle (voir échange
    précédent sur la stratégie de troncature).
    """
    dataset = Dataset.from_pandas(df[["text", "label"]])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    return dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    """Calcule l'accuracy pendant l'évaluation, appelée automatiquement
    par le Trainer à chaque étape d'évaluation."""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def plot_confusion_matrix(y_true, y_pred, title: str = "BERT fine-tuné") -> None:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real", "fake"])
    disp.plot(cmap="Blues")
    plt.title(f"Matrice de confusion — {title}")
    plt.savefig(f"data/processed/confusion_matrix_{title.replace(' ', '_').lower()}.png")
    print(cm)


if __name__ == "__main__":
   if __name__ == "__main__":
    # --- 1. Chargement et nettoyage ---
    df_raw = pd.read_excel("data/raw/fake_news.xlsx")
    inspect_dataset(df_raw)
    df_clean = load_and_clean_fake_news("data/raw/fake_news.xlsx", min_length=50)

    # --- TEST : réduction à un petit échantillon pour valider le pipeline ---
        # --- TEST : réduction à un petit échantillon pour valider le pipeline ---
    fake_sample = df_clean[df_clean["label"] == 1].sample(100, random_state=42)
    real_sample = df_clean[df_clean["label"] == 0].sample(100, random_state=42)
    df_clean = pd.concat([fake_sample, real_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n[MODE TEST] Échantillon réduit : {len(df_clean)} lignes")
    print(df_clean["label"].value_counts())

    # --- 2. Split train/test ---
    train_df, test_df = split_dataset(df_clean)

    # --- 3. Tokenisation ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = tokenize_dataset(train_df, tokenizer)
    test_dataset = tokenize_dataset(test_df, tokenizer)

    # --- 4. Chargement du modèle (pas encore entraîné pour cette tâche) ---
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # --- 5. Configuration de l'entraînement ---
    training_args = TrainingArguments(
        output_dir="data/processed/finetuned_model",
        num_train_epochs=1,              # 2 passages sur tout le dataset
        per_device_train_batch_size=2,   # petit batch, adapté à un CPU/GPU limité
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
    )

    # --- 6. Entraînement ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nDémarrage du fine-tuning...")
    trainer.train()

    # --- 7. Évaluation finale ---
    results = trainer.evaluate()
    print(f"\nRésultats finaux : {results}")

    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(axis=-1)
    y_true = test_df["label"].tolist()

    plot_confusion_matrix(y_true, y_pred)

    # --- 8. Sauvegarde du modèle fine-tuné ---
    trainer.save_model("data/processed/finetuned_model_final")
    print("\nModèle sauvegardé dans data/processed/finetuned_model_final")