from pathlib import Path

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

import spacy

# --- Setup NLTK ---
STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# --- Setup spaCy ---
nlp = spacy.load("en_core_web_sm")


def remove_stopwords(text: str) -> str:
    """Retire les mots vides (articles, conjonctions, pronoms...) du texte.

    Fondement : ces mots sont extrêmement fréquents mais portent peu de sens
    sémantique distinctif ('the', 'is', 'and'...). Pour du retrieval basé sur
    la fréquence des mots (TF-IDF, BM25), ils polluent le signal. Les retirer
    réduit aussi la taille du vocabulaire à indexer.
    """
    words = word_tokenize(text)
    filtered = [w for w in words if w.lower() not in STOP_WORDS]
    return " \n".join(filtered)


def apply_stemming(text: str) -> str:
    """Réduit chaque mot à sa racine grossière en coupant les suffixes.

    Fondement : approche mécanique et rapide (règles de troncature), mais
    grossière : le résultat n'est pas forcément un vrai mot ('amaz' n'existe
    pas en anglais).
    """
    words = word_tokenize(text)
    stemmed = [stemmer.stem(w) for w in words]
    return " \n".join(stemmed)


def get_wordnet_pos(treebank_tag: str) -> str:
    """Convertit un tag POS détaillé (Penn Treebank, ex: 'VBG') vers le
    format simplifié attendu par WordNetLemmatizer ('v', 'n', 'a', 'r').

    Fondement : pos_tag() et lemmatize() viennent de deux ressources
    linguistiques différentes qui ne parlent pas le même "langage" de tags.
    Sans cette conversion, lemmatize() ignorerait la nature grammaticale
    réelle du mot et supposerait "nom" par défaut, donnant des résultats
    faux sur les verbes/adjectifs/adverbes.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def apply_lemmatization_nltk(text: str) -> str:
    """Lemmatise avec NLTK, en détectant automatiquement la nature
    grammaticale de chaque mot (POS tagging) plutôt que de la supposer fixe.

    Fondement : le lemme correct d'un mot dépend de sa fonction dans la
    phrase ('meeting' = nom ou verbe donnent des lemmes différents). Le POS
    tagging automatique adapte la lemmatization mot par mot, contrairement
    à un pos fixe qui ne serait juste qu'approximatif.
    """
    words = word_tokenize(text)
    tagged = pos_tag(words)
    lemmatized = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in tagged
    ]
    return " \n".join(lemmatized)


def apply_lemmatization_spacy(text: str) -> str:
    """Lemmatise avec spaCy, qui gère nativement le contexte grammatical
    en un seul appel, sans conversion de tag manuelle.

    Fondement : spaCy analyse tokenisation + POS tagging + lemmatization
    ensemble dans le même pipeline, avec un modèle pré-entraîné plus
    sophistiqué que les règles de NLTK — généralement plus précis, au prix
    d'un modèle à charger en mémoire.
    """
    doc = nlp(text)
    return " \n".join(token.lemma_ for token in doc)

def extract_entities_spacy(text: str) -> list[tuple[str, str]]:
    """Détecte et catégorise les entités nommées présentes dans le texte :
    personnes, organisations, lieux, dates, pourcentages, quantités...

    Fondement général : contrairement au stopwords/stemming/lemmatization qui
    normalisent le texte (réduire le bruit, regrouper les variantes d'un même
    mot), le NER repère des unités de sens porteuses d'une information
    factuelle précise et les classe par catégorie. C'est un changement de
    nature de traitement : on ne simplifie plus le texte, on l'annote.

    Utilité pour le RAG : ces entités peuvent servir de métadonnées
    structurées associées à un chunk (en plus de son contenu textuel brut),
    utiles pour filtrer ou prioriser le retrieval selon les entités
    mentionnées dans la question de l'utilisateur.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


"""if __name__ == "__main__":
    input_path = Path("sample.txt")
    text = input_path.read_text(encoding="utf-8")

    no_stopwords = remove_stopwords(text)
    stemmed = apply_stemming(text)
    lemmatized_nltk = apply_lemmatization_nltk(text)
    lemmatized_spacy = apply_lemmatization_spacy(text)
    entities = extract_entities_spacy(text)
    Path("sample_no_stopwords.txt").write_text(no_stopwords, encoding="utf-8")
    Path("sample_stemming.txt").write_text(stemmed, encoding="utf-8")
    Path("sample_lemmatization_nltk.txt").write_text(lemmatized_nltk, encoding="utf-8")
    Path("sample_lemmatization_spacy.txt").write_text(lemmatized_spacy, encoding="utf-8")

   
    entities_output = "\n".join(f"{ent_text} -> {ent_label}" for ent_text, ent_label in entities)
    Path("sample_entities.txt").write_text(entities_output, encoding="utf-8")

print("Terminé : 5 fichiers générés.")"""

if __name__ == "__main__":
    input_path = Path("data/corpus.txt")
    text = input_path.read_text(encoding="utf-8")

    # spaCy limite nlp() à ~1M caractères par défaut (protection mémoire).
    # On augmente la limite si le corpus la dépasse.
    if len(text) > nlp.max_length:
        nlp.max_length = len(text) + 1000
        print(f"nlp.max_length augmenté à {nlp.max_length} (corpus de {len(text)} caractères)")

    print("Traitement en cours...")

    no_stopwords = remove_stopwords(text)
    stemmed = apply_stemming(text)
    lemmatized_nltk = apply_lemmatization_nltk(text)
    lemmatized_spacy = apply_lemmatization_spacy(text)
    entities = extract_entities_spacy(text)

    Path("corpus_no_stopwords.txt").write_text(no_stopwords, encoding="utf-8")
    Path("corpus_stemming.txt").write_text(stemmed, encoding="utf-8")
    Path("corpus_lemmatization_nltk.txt").write_text(lemmatized_nltk, encoding="utf-8")
    Path("corpus_lemmatization_spacy.txt").write_text(lemmatized_spacy, encoding="utf-8")

    entities_output = "\n".join(f"{ent_text} -> {ent_label}" for ent_text, ent_label in entities)
    Path("corpus_entities.txt").write_text(entities_output, encoding="utf-8")

    print("Terminé : 5 fichiers générés.")

