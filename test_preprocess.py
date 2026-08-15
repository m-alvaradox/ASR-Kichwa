import pandas as pd

from data.preprocess import clean_text, clean_metadata_csv


def test_clean_text_rules():
    text = "¡Qué chulla! Cuna, quimsa, ñukaka."
    assert clean_text(text) == "ke chulla kuna kimsa nukaka"


def test_clean_metadata_csv_updates_transcription(tmp_path):
    path = tmp_path / "metadata.csv"
    df = pd.DataFrame({
        "filename": ["a.wav"],
        "transcription": ["¡Qué chulla! Cuna, quimsa, ñukaka."],
    })
    df.to_csv(path, index=False)

    clean_metadata_csv(path)
    result = pd.read_csv(path)

    assert result.loc[0, "transcription"] == "ke chulla kuna kimsa nukaka"
