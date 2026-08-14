import os
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

from utils.vocabulary import Vocabulary

class KillKanDataset(Dataset):

    def __init__(
            self,
            metadata_path,
            audio_dir,
            vocabulary: Vocabulary,
            sample_rate=16000
    ):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.vocabulary = vocabulary

        if isinstance(metadata_path, pd.DataFrame):
            self.data = metadata_path.copy()
        else:
            self.data = pd.read_csv(metadata_path)

        self.data = self.data.dropna(subset=["sentence"])

        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        filename = row["filename"]

        filename = filename.replace(".mp3", ".wav")

        audio_path = os.path.join(
            self.audio_dir,
            filename
        )

        waveform, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True
        )

        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        )

        sentence = row["sentence"]

        target = self.vocabulary.text_to_indices(sentence)

        target = torch.tensor(
            target,
            dtype=torch.long
        )

        return waveform, target


class KillKanEmbeddingsDataset(Dataset):

    def __init__(
            self,
            metadata_path,
            embeddings_dir,
            vocabulary: Vocabulary
    ):
        self.embeddings_dir = embeddings_dir
        self.vocabulary = vocabulary

        if isinstance(metadata_path, pd.DataFrame):
            self.data = metadata_path.copy()
        else:
            self.data = pd.read_csv(metadata_path)
        self.data = self.data.dropna(subset=["transcription"])
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        filename = row["filename"]
        emb_path = os.path.join(self.embeddings_dir, filename + ".pt")

        embedding = torch.load(emb_path).squeeze(0)
        sentence = row["transcription"]
        target = self.vocabulary.text_to_indices(sentence)
        target = torch.tensor(target, dtype=torch.long)

        return embedding, target