## Libraries & Imports:
import spacy
import re
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from .utils import timeit, save_csv, load_csv
import json

## Paths: 
INPUT_VICTIM_CSV = Path("processed/coref_victim_sentences.csv")
INPUT_SHOOTER_CSV = Path("processed/coref_shooter_sentences.csv")

OUTPUT_VICTIM_CSV = Path("processed/descriptions_victims.csv")
OUTPUT_SHOOTER_CSV = Path("processed/descriptions_shooters.csv")

## Load Models: SpaCy
nlp = spacy.load("en_core_web_sm")

## Lexicons:
HARM_VERBS = {"kill", "shoot", "injure", "wound", "strike", "hit", "murder", "die"}
SHOOTER_NOUNS = {"shooter", "gunman", "suspect", "perpetrator", "attacker"}
VICTIM_NOUNS = {"child", "children", "student", "students", "teacher", "person", "people", "grandmother", "adult", "massacre", "shooting"}
WEAPON_NOUNS = {"gun", "rifle", "handgun", "weapon", "ammunition", "rounds"}
BIO_ATTR_VERBS = {"be", "become", "remain", "know", "have"}
PURCHASE_VERBS = {"buy", "purchase", "acquire"}

VICTIM_NUMBER_PATTERN = re.compile(
    r"\b(?:at least|more than|over|approximately)?\s*(\d+|a dozen|dozen)\s*(children|students|adults|people|others)\b",
    flags=re.IGNORECASE
)

## Functions to Help Extraction:
def normalize(text: str) -> str:
    return text.lower().strip()

def clean_phrase(tokens):
    """Return text of a token subtree without punctuation."""
    return " ".join(t.text for t in tokens if not t.is_punct)

## Rule Based Extraction: Victims
def extract_victim_rule(doc):
    phrases = set()

    # Numeric + Victim Noun
    for token in doc:
        if token.lemma_ in VICTIM_NOUNS:
            nums = [c.text for c in token.children if c.dep_ == "nummod"]
            phrases.add(" ".join(nums + [token.text]).strip())

    # Appositions / Direct Objects of Harm Verbs
    for token in doc:
        if token.dep_ == "appos" and token.head.lemma_ in VICTIM_NOUNS:
            phrases.add(clean_phrase(token.subtree))
        if token.lemma_ in HARM_VERBS and token.pos_ == "VERB":
            for obj in token.children:
                if obj.dep_ in ("dobj", "pobj"):
                    nums = [c.text for c in obj.children if c.dep_ == "nummod"]
                    conj = [c.text for c in obj.children if c.dep_ == "conj"]
                    phrases.add(" ".join(nums + [obj.text] + conj + [token.lemma_]).strip())

    # Regex Numeric Patterns
    for match in VICTIM_NUMBER_PATTERN.finditer(doc.text):
        phrases.add(match.group(0))

    return list(phrases)

## Rule Based Extraction: Shooters
def extract_shooter_rule(doc):
    phrases = set()

    # Shooter Nouns + Adjectives
    for token in doc:
        if token.lemma_ in SHOOTER_NOUNS:
            adj = [c.text for c in token.children if c.dep_ == "amod"]
            phrases.add(" ".join(adj + [token.text]))

    # Copula + Attribute
    for token in doc:
        if token.lemma_ in BIO_ATTR_VERBS and token.pos_ == "VERB":
            for attr in token.children:
                if attr.dep_ in ("attr", "acomp"):
                    phrases.add(clean_phrase(attr.subtree))

    # Weapon Acquisition
    for token in doc:
        if token.lemma_ in PURCHASE_VERBS or token.lemma_ == "have":
            for obj in token.children:
                if obj.dep_ in ("dobj", "pobj") and obj.lemma_ in WEAPON_NOUNS:
                    nums = [c.text for c in obj.children if c.dep_ == "nummod"]
                    phrases.add(f"{token.lemma_} {' '.join(nums + [obj.text])}".strip())

    return list(phrases)

## Optional: LLM-based extraction (local)
def extract_llm(sentence, llm_model=None):
    """Extract descriptive phrases using a local LLM. Returns JSON dict."""
    if llm_model is None:
        return {"victim_descriptions": [], "shooter_descriptions": []}

    prompt = f"""
    Extract descriptive phrases from the following sentence.
    Classify them into 'victim_descriptions' and 'shooter_descriptions'.
    Return strictly valid JSON with keys:
    victim_descriptions (list of strings)
    shooter_descriptions (list of strings)

    Sentence: {sentence}
    """
    try:
        response = llm_model.chat(prompt)  # adapt to your local LLM wrapper
        return json.loads(response)
    except Exception as e:
        print("LLM extraction failed:", e)
        return {"victim_descriptions": [], "shooter_descriptions": []}


## Combine Rule-Based + LLM Extraction
@timeit
def extract_descriptions(sentence, llm_model=None):
    doc = nlp(sentence)
    
    victim_rb = extract_victim_rule(doc)
    shooter_rb = extract_shooter_rule(doc)
    
    llm_res = extract_llm(sentence, llm_model)
    victim_llm = llm_res.get("victim_descriptions", [])
    shooter_llm = llm_res.get("shooter_descriptions", [])
    
    # Merge and Deduplicate
    victim_final = sorted(set(victim_rb + victim_llm))
    shooter_final = sorted(set(shooter_rb + shooter_llm))
    
    # Sentiment
    sentiment = TextBlob(sentence).sentiment.polarity

    return {
        "victim_descriptions": " | ".join(victim_final),
        "shooter_descriptions": " | ".join(shooter_final),
        "sentiment": sentiment
    }

## CSV Processing
def process_file(input_csv: Path, output_csv: Path, entity_name: str, llm_model=None):
    df = load_csv(input_csv)
    if df is None or df.empty:
        print(f"No input found for {entity_name}")
        return

    records = []
    for idx, row in df.iterrows():
        result = extract_descriptions(row.get("sentence", ""), llm_model=llm_model)
        records.append({
            "journal": row.get("journal", ""),
            "article_number": row.get("article_number", ""),
            "sentence": row.get("sentence", ""),
            "victim_descriptions": result["victim_descriptions"],
            "shooter_descriptions": result["shooter_descriptions"],
            "sentiment": result["sentiment"]
        })

    save_csv(pd.DataFrame(records), output_csv)
    print(f"Saved {entity_name} descriptions → {output_csv}")

## Main Pipeline
def main(llm_model=None):
    process_file(INPUT_VICTIM_CSV, OUTPUT_VICTIM_CSV, "Victims", llm_model=llm_model)
    process_file(INPUT_SHOOTER_CSV, OUTPUT_SHOOTER_CSV, "Shooters", llm_model=llm_model)

if __name__ == "__main__":
    # Optionally initialize a local LLM here, e.g., LLaMA-2 7B or Qwen-7B
    llm_model = None
    # from local_llm import LLMWrapper
    # llm_model = LLMWrapper(model, tokenizer)
    
    main(llm_model=llm_model)
