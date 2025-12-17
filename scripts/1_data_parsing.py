## Libraries & Imports:
import re
import unicodedata
from pathlib import Path
import pandas as pd
import spacy
from .utils import save_csv

## Paths:
DATA_DIR = Path("data")
OUTPUT_CSV = Path("processed/articles.csv")

FOLDERS = {
    "CNN": DATA_DIR / "cnn_five_para",
    "FOX": DATA_DIR / "FOX_five_para",
    "NYT": DATA_DIR / "NYT_five_para",
    "WSJ": DATA_DIR / "WSJ_five_para"
}

## Load Model: Spacy
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer", before="parser")

## Function Utilities:
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "‚Äì": "–",
        "‚Äî": "—",
        "‚Äô": "'",
        "‚Äú": '"',
        "‚Äù": '"',
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "–",
        "â€”": "—"
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)  # 18yo → 18 yo
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)  # Grade3 → Grade 3
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def split_sentences(text: str):
    text = normalize_text(text)
    text = re.sub(r"\n+", ". ", text)

    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) < 15 or s.count(" ") < 3:
            continue
        sentences.append(s)

    return sentences

def parse_filename(file_path: Path, default_journal: str):
    stem = file_path.stem.strip()
    match = re.match(r"([A-Za-z]+)_(\d+)_([A-Za-z0-9]+)$", stem)
    if match:
        journal, article_num, date = match.groups()
        return journal.upper(), int(article_num), date

    match = re.match(r"([A-Za-z]+)_(\d+)$", stem)
    if match:
        journal, article_num = match.groups()
        return journal.upper(), int(article_num), "UNKNOWN"

    match = re.search(r"(\d+)", stem)
    if match:
        return default_journal, int(match.group(1)), "UNKNOWN"

    raise ValueError(f"Unparseable filename: {file_path.name}")

## Main Pipeline:
def main():
    records = []
    fallback_files = []

    for journal, folder_path in FOLDERS.items():
        if not folder_path.exists():
            continue

        for file in folder_path.iterdir():
            if not file.is_file() or file.suffix.lower() not in {".txt", ".text"}:
                continue

            parsed_journal, article_number, date = parse_filename(file, journal)

            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

            sentences = split_sentences(raw_text)

            for i, sentence_text in enumerate(sentences):
                records.append({
                    "journal": parsed_journal,
                    "article_number": article_number,
                    "date": date,
                    "file_name": file.name,
                    "sentence_id": i,
                    "text": sentence_text
                })

    df = pd.DataFrame(records)
    df = df.sort_values(
        by=["journal", "article_number", "sentence_id"]
    ).reset_index(drop=True)

    save_csv(df, OUTPUT_CSV)
    print(f"Total sentences compiled: {len(df)}")


if __name__ == "__main__":
    main()