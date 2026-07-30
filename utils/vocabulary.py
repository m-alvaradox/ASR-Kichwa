import re
import pandas as pd


class Vocabulary:
    """
    Construye un vocabulario a partir de las transcripciones
    y permite convertir texto <-> índices.
    """

    def __init__(self):

        self.char2idx = {}
        self.idx2char = {}

    def _clean_sentence(self, sentence: str) -> str:
        """
        Limpieza básica de la transcripción.
        """

        # Eliminar etiquetas de anotación
        sentence = re.sub(r"\[SPANISH\]", "", sentence)
        sentence = re.sub(r"\[UNK\]", "", sentence)

        # Eliminar espacios repetidos
        sentence = re.sub(r"\s+", " ", sentence)

        return sentence.strip()

    def build(self, metadata_path):

        df = pd.read_csv(metadata_path)

        # Eliminar filas sin transcripción
        df = df.dropna(subset=["sentence"])

        characters = set()

        for sentence in df["sentence"]:

            sentence = str(sentence)

            sentence = self._clean_sentence(sentence)

            if sentence == "":
                continue

            characters.update(sentence)

        # CTC requiere BLANK en la posición 0
        self.char2idx = {"<blank>": 0}

        for index, char in enumerate(sorted(characters), start=1):
            self.char2idx[char] = index

        self.idx2char = {
            idx: char
            for char, idx in self.char2idx.items()
        }

    def text_to_indices(self, text):

        text = self._clean_sentence(str(text))

        return [
            self.char2idx[c]
            for c in text
            if c in self.char2idx
        ]

    def indices_to_text(self, indices):

        return "".join(
            self.idx2char[i]
            for i in indices
            if i != 0
        )

    def __len__(self):

        return len(self.char2idx)
