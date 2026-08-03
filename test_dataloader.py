from torch.utils.data import DataLoader

from utils.vocabulary import Vocabulary
from data.dataset import KillKanDataset
from data.collate import collate_fn


def main():

    # ----------------------------
    # Construir vocabulario
    # ----------------------------

    vocabulary = Vocabulary()

    vocabulary.build("dataset/metadata.csv")

    # ----------------------------
    # Dataset
    # ----------------------------

    dataset = KillKanDataset(
        metadata_path="dataset/metadata.csv",
        audio_dir="dataset/wav",
        vocabulary=vocabulary
    )

    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Número de ejemplos: {len(dataset)}")

    # ----------------------------
    # DataLoader
    # ----------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Obtener únicamente el primer batch

    waveforms, targets, input_lengths, target_lengths = next(iter(dataloader))

    print()
    print("=" * 60)
    print("BATCH")
    print("=" * 60)

    print("Waveforms shape:")
    print(waveforms.shape)

    print()

    print("Targets shape:")
    print(targets.shape)

    print()

    print("Input lengths:")
    print(input_lengths)

    print()

    print("Target lengths:")
    print(target_lengths)

    print()

    print("=" * 60)
    print("PRIMER EJEMPLO")
    print("=" * 60)

    print()

    print("Audio length:")

    print(input_lengths[0].item())

    print()

    print("Target:")

    print(targets[0])

    print()

    print("Texto reconstruido:")

    text = vocabulary.indices_to_text(
        targets[0].tolist()
    )

    print(text)


if __name__ == "__main__":
    main()