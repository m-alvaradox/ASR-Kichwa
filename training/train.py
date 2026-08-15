import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import KillKanEmbeddingsDataset
from models.decoder import KichwaDecoder1D
from utils.vocabulary import Vocabulary


ROOT_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT_DIR
CHECKPOINT_DIR = WORK_DIR / "checkpoints"

TRAIN_METADATA = WORK_DIR / "dataset" / "metadata_train.csv"
VALID_METADATA = WORK_DIR / "dataset" / "metadata_valid.csv"

TRAIN_EMBEDDINGS = WORK_DIR / "embeddings" / "train"
VALID_EMBEDDINGS = WORK_DIR / "embeddings" / "valid"

DECODER_CHECKPOINT = CHECKPOINT_DIR / "kichwa_decoder_conv1d.pt"


def build_vocabulary(metadata_path: Path) -> Vocabulary:
    vocabulary = Vocabulary()
    vocabulary.build(str(metadata_path))
    return vocabulary


def load_dataloader(metadata_path: Path, embeddings_dir: Path, vocabulary: Vocabulary, batch_size: int, shuffle: bool):
    dataset = KillKanEmbeddingsDataset(
        metadata_path=str(metadata_path),
        embeddings_dir=str(embeddings_dir),
        vocabulary=vocabulary,
    )


def split_training_frame(metadata_path: Path):
    frame = pd.read_csv(metadata_path)
    validation_frame = frame.sample(frac=0.1, random_state=42)
    training_frame = frame.drop(validation_frame.index).reset_index(drop=True)
    validation_frame = validation_frame.reset_index(drop=True)
    return training_frame, validation_frame

def greedy_ctc_decode(logits, vocabulary: Vocabulary):
    pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    decoded_chars = []
    previous_id = None

    for token_id in pred_ids:
        if token_id != previous_id and token_id != 0:
            decoded_chars.append(vocabulary.idx2char.get(token_id, ""))
        previous_id = token_id

    return "".join(decoded_chars)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    if not TRAIN_METADATA.exists():
        raise FileNotFoundError(f"No existe el archivo de metadatos de entrenamiento: {TRAIN_METADATA}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vocabulary = build_vocabulary(TRAIN_METADATA)
    print(f"Vocabulario construido con exito. Tamano total: {len(vocabulary)} caracteres.")

    if VALID_METADATA.exists():
        train_source = TRAIN_METADATA
        valid_source = VALID_METADATA
        train_embeddings_dir = TRAIN_EMBEDDINGS
        valid_embeddings_dir = VALID_EMBEDDINGS
    else:
        print("No se encontro metadata_valid.csv; se hara un split automatico desde metadata_train.csv.")
        train_source, valid_source = split_training_frame(TRAIN_METADATA)
        train_embeddings_dir = TRAIN_EMBEDDINGS
        valid_embeddings_dir = TRAIN_EMBEDDINGS

    train_loader = load_dataloader(train_source, train_embeddings_dir, vocabulary, batch_size=16, shuffle=True)
    valid_loader = load_dataloader(valid_source, valid_embeddings_dir, vocabulary, batch_size=16, shuffle=False)

    model = KichwaDecoder1D(input_dim=768, hidden_dim=512, vocab_size=len(vocabulary), dropout=0.25).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    num_epochs = 25

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for embeddings, targets, input_lengths, target_lengths in train_loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(embeddings)
            log_probs = logits.log_softmax(2).transpose(0, 1)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for embeddings, targets, input_lengths, target_lengths in valid_loader:
                embeddings = embeddings.to(device)
                targets = targets.to(device)

                logits = model(embeddings)
                log_probs = logits.log_softmax(2).transpose(0, 1)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                val_loss += loss.item()

        val_loss /= max(len(valid_loader), 1)
        scheduler.step(val_loss)

        print(f"Epoca [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), DECODER_CHECKPOINT)
            print(f"  -> Nuevo mejor modelo guardado (Val Loss: {val_loss:.4f})")

    print("Entrenamiento finalizado.")


if __name__ == "__main__":
    main()