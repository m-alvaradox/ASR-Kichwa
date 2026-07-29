import pandas as pd

class Vocabulary:

    """
    Clase para construir un vocabulario a partir de las
    transcripciones y permite convertir texto en indices
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def build(self, metadata_path):

        df = pd.read_csv(metadata_path)

        characters = set()

        for sentence in df["sentence"]:
            characters.update(sentence)

        characters = sorted(characters)

        self.char2idx["<blank>"] == 0

        index = 1

        for c in characters:
            self.char2idx[c] = index
            index += 1

        self.idx2char = {
            v: k for k, v in self.char2idx.items()
        }

        def indices_to_text(self, indices):

            return "".join(
                self.idx2char[i]
                for i in indices
                if i!=0
            )

        def __len__(self):
            return len(self.char2idx)
