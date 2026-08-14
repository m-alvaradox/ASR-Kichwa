import re
import pandas as pd


class Vocabulary:

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def _clean_sentence(self, sentence: str) -> str:
        sentence = sentence.lower()
        sentence = re.sub(r"\[spanish\]", "", sentence)
        sentence = re.sub(r"\[unk\]", "", sentence)
        sentence = re.sub(r"[^\w\s]", "", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.strip()

    def build(self, metadata_path):
        df = pd.read_csv(metadata_path)
        df = df.dropna(subset=["transcription"])
        characters = set()
        for sentence in df["transcription"]:
            sentence = str(sentence)
            sentence = self._clean_sentence(sentence)
            if sentence == "":
                continue
            characters.update(sentence)

        self.char2idx = {"<blank>": 0}
        for index, char in enumerate(sorted(characters), start=1):
            self.char2idx[char] = index
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def text_to_indices(self, text):
        text = self._clean_sentence(str(text))
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def indices_to_text(self, indices):
        return "".join(
            self.idx2char.get(int(index), "")
            for index in indices
            if int(index) != 0
        )

    def __len__(self):
        return len(self.char2idx)
