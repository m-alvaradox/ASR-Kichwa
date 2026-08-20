"""Crea checkpoints FP16 más pequeños para despliegues CPU.

Los modelos de la aplicación siguen en FP32. PyTorch convierte los tensores al
tipo del parámetro de destino al cargar el state_dict.
"""

from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "deploy_checkpoints"
CHECKPOINT_NAMES = (
    "wav2vec_small_clean.pt",
    "kichwa_decoder_conv1d.pt",
)


def to_half(value):
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.half()
    if isinstance(value, dict):
        return type(value)((key, to_half(item)) for key, item in value.items())
    if isinstance(value, list):
        return [to_half(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_half(item) for item in value)
    return value


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in CHECKPOINT_NAMES:
        source = SOURCE_DIR / name
        destination = OUTPUT_DIR / name

        if not source.exists():
            raise FileNotFoundError(f"No se encontró {source}")

        checkpoint = torch.load(source, map_location="cpu")
        torch.save(to_half(checkpoint), destination)

        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"Creado {destination.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
