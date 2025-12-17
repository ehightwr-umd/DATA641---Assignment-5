## Libraries & Imports:
import re
import spacy
import pandas as pd
from pathlib import Path
from .utils import timeit, save_csv, load_csv

## Paths:
INPUT_CSV = Path("processed/articles.csv")
OUTPUT_VICTIM_CSV = Path("processed/coref_victim_sentences.csv")
OUTPUT_SHOOTER_CSV = Path("processed/coref_shooter_sentences.csv")

## Load Model: spaCy
nlp = spacy.load("en_core_web_sm")

## Define Lexicons:
VICTIM_KEYWORDS = ["child"]
SHOOTER_KEYWORDS = ["shooter", "suspect", "gunman", "perpetrator"]

HARM_VERBS = ["kill", "dead", "injure", "wound", "shoot", "die"]
VICTIM_PATTERNS = [
    r"\b\d+\s+(?:children|people|students|men|women)\b",
    r"\bdead\b", r"\binjured\b", r"\bwounded\b", r"\bkilled\b"
]

## Function Utilties
def is_victim_sentence(sentence: str) -> bool:
    for pat in VICTIM_PATTERNS:
        if re.search(pat, sentence, re.IGNORECASE):
            return True

    doc = nlp(sentence.lower())
    for token in doc:
        if token.lemma_ in VICTIM_KEYWORDS and any(v in sentence.lower() for v in HARM_VERBS):
            return True
    return False


def is_shooter_sentence(sentence: str) -> bool:
    sentence_lower = sentence.lower()
    return any(word in sentence_lower for word in SHOOTER_KEYWORDS)

## Main Pipeline:
@timeit
def main():
    cached_victim = load_csv(OUTPUT_VICTIM_CSV)
    cached_shooter = load_csv(OUTPUT_SHOOTER_CSV)
    if cached_victim is not None and cached_shooter is not None:
        print("Using cached coref outputs.")
        return

    df = pd.read_csv(INPUT_CSV)

    victim_records = []
    shooter_records = []

    grouped = df.groupby(["journal", "article_number"])

    for (_, _), group in grouped:
        group = group.sort_values("sentence_id").reset_index(drop=True)

        for i, row in group.iterrows():
            sentence = row["text"]

            context = []
            if i > 0:
                context.append(group.loc[i - 1, "text"])
            context.append(sentence)
            if i < len(group) - 1:
                context.append(group.loc[i + 1, "text"])

            context_span = " ".join(context)

            if is_victim_sentence(sentence):
                victim_records.append({
                    "journal": row["journal"],
                    "article_number": row["article_number"],
                    "date": row["date"],
                    "sentence": sentence,
                    "context_span": context_span
                })

            elif is_shooter_sentence(sentence):
                shooter_records.append({
                    "journal": row["journal"],
                    "article_number": row["article_number"],
                    "date": row["date"],
                    "sentence": sentence,
                    "context_span": context_span
                })

    save_csv(pd.DataFrame(victim_records), OUTPUT_VICTIM_CSV)
    save_csv(pd.DataFrame(shooter_records), OUTPUT_SHOOTER_CSV)

    print(f"Saved {len(victim_records)} victim contexts")
    print(f"Saved {len(shooter_records)} shooter contexts")

if __name__ == "__main__":
    main()
