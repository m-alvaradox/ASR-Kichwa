import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):

    waveforms = []
    targets = []

    input_lengths = []
    target_lengths = []

    for waveform, target in batch:

        waveforms.append(waveform)

        targets.append(target)

        input_lengths.append(len(waveform))

        target_lengths.append(len(target))

    # Padding del audio
    waveforms = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0
    )

    # Padding del texto
    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=0
    )

    input_lengths = torch.tensor(
        input_lengths,
        dtype=torch.long
    )

    target_lengths = torch.tensor(
        target_lengths,
        dtype=torch.long
    )

    return (
        waveforms,
        targets,
        input_lengths,
        target_lengths
    )