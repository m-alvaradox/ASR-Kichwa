from utils.vocabulary import Vocabulary
from data.dataset import KillKanDataset

vocab = Vocabulary()
vocab.build("dataset/metadata.csv")

dataset = KillKanDataset(
    metadata_path="dataset/metadata.csv",
    audio_dir="dataset/wav",
    vocabulary=vocab
)

print("Número de ejemplos:", len(dataset))

waveform, target = dataset[0]

print()

print("Forma del audio:")

print(waveform.shape)

print()

print("Primeros valores:")

print(waveform[:10])

print()

print("Target:")

print(target)

print()

print("Texto reconstruido:")

print(vocab.indices_to_text(target.tolist()))