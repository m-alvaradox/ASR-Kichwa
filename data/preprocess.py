import re
import unicodedata
from pathlib import Path

import pandas as pd


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def clean_text(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = strip_accents(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in ["transcription", "sentence", "text"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(clean_text)

    return cleaned


def clean_metadata_csv(path):
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)
    df.to_csv(csv_path, index=False)
    return df


def clean_metadata_directory(directory):
    root = Path(directory)
    if not root.exists():
        return []

    cleaned_files = []
    for csv_path in sorted(root.glob("*.csv")):
        if "metadata" in csv_path.name.lower():
            clean_metadata_csv(csv_path)
            cleaned_files.append(csv_path)

    return cleaned_files


if __name__ == "__main__":
    for folder in ["csv", "dataset"]:
        cleaned = clean_metadata_directory(folder)
        if cleaned:
            print(f"Procesados {len(cleaned)} CSVs en {folder}")
