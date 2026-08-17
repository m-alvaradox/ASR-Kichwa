import os
import sys
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from jiwer import cer, wer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.dataset import KillKanEmbeddingsDataset
from models.decoder import KichwaDecoder1D
from utils.vocabulary import Vocabulary

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


def collate_fn(batch):
    embeddings, targets = zip(*batch)
    input_lengths = torch.tensor([len(emb) for emb in embeddings], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    embeddings_padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return embeddings_padded, targets_padded, input_lengths, target_lengths


def load_dataloader(metadata_path, embeddings_dir: Path, vocabulary: Vocabulary, batch_size: int, shuffle: bool):
    dataset = KillKanEmbeddingsDataset(
        metadata_path=metadata_path,
        embeddings_dir=str(embeddings_dir),
        vocabulary=vocabulary,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
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


def build_confusion_matrix(model, loader, vocabulary: Vocabulary, device):
    vocab_chars = sorted(c for c in vocabulary.char2idx.keys() if c != "<blank>")
    char_to_row = {c: i for i, c in enumerate(vocab_chars)}
    n = len(vocab_chars)
    confusion = np.zeros((n, n), dtype=int)

    model.eval()
    with torch.no_grad():
        for embeddings, targets, input_lengths, target_lengths in tqdm(loader, desc="Matriz de confusion"):
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            logits = model(embeddings)

            for i in range(logits.size(0)):
                pred_text = greedy_ctc_decode(logits[i].unsqueeze(0), vocabulary)
                real_indices = targets[i][:target_lengths[i]].cpu().tolist()
                real_text = vocabulary.indices_to_text(real_indices)

                if len(real_text.strip()) == 0 or len(pred_text.strip()) == 0:
                    continue

                output = jiwer.process_characters(real_text, pred_text)
                for alignment in output.alignments[0]:
                    if alignment.type != "substitute":
                        continue
                    for offset in range(alignment.ref_end_idx - alignment.ref_start_idx):
                        real_char = real_text[alignment.ref_start_idx + offset]
                        hyp_idx = alignment.hyp_start_idx + offset
                        if hyp_idx >= alignment.hyp_end_idx:
                            continue
                        pred_char = pred_text[hyp_idx]
                        if real_char in char_to_row and pred_char in char_to_row:
                            confusion[char_to_row[real_char], char_to_row[pred_char]] += 1

    return confusion, vocab_chars


def plot_confusion_matrix(confusion, vocab_chars, save_path: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar(label="Sustituciones")
    plt.xticks(range(len(vocab_chars)), vocab_chars, rotation=90)
    plt.yticks(range(len(vocab_chars)), vocab_chars)
    plt.xlabel("Caracter predicho")
    plt.ylabel("Caracter real")
    plt.title("Matriz de confusion (sustituciones de caracteres)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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

    model = KichwaDecoder1D(input_dim=768, hidden_dim=256, vocab_size=len(vocabulary), dropout=0.25).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=2)

    best_val_loss = float("inf")
    num_epochs = 40
    patience = 4
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []
    val_cer_history = []
    val_wer_history = []

    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=True)
            for embeddings, targets, input_lengths, target_lengths in progress:
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
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= max(len(train_loader), 1)

            model.eval()
            val_loss = 0.0
            val_cer = 0.0
            val_wer = 0.0

            with torch.no_grad():
                val_progress = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [valid]", leave=False)
                for embeddings, targets, input_lengths, target_lengths in val_progress:
                    embeddings = embeddings.to(device)
                    targets = targets.to(device)

                    logits = model(embeddings)
                    log_probs = logits.log_softmax(2).transpose(0, 1)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                    val_loss += loss.item()

                    batch_cer = 0.0
                    batch_wer = 0.0

                    for i in range(logits.size(0)):
                        pred_text = greedy_ctc_decode(logits[i].unsqueeze(0), vocabulary)
                        real_indices = targets[i][:target_lengths[i]].cpu().tolist()
                        real_text = vocabulary.indices_to_text(real_indices)

                        if len(real_text.strip()) > 0:
                            if len(pred_text.strip()) == 0:
                                batch_cer += 1.0
                                batch_wer += 1.0
                            else:
                                batch_cer += cer(real_text, pred_text)
                                batch_wer += wer(real_text, pred_text)

                    val_cer += batch_cer / logits.size(0)
                    val_wer += batch_wer / logits.size(0)
                    
                    val_progress.set_postfix(loss=f"{loss.item():.4f}")

            val_loss /= max(len(valid_loader), 1)
            val_cer /= max(len(valid_loader), 1)
            val_wer /= max(len(valid_loader), 1)

            scheduler.step(val_loss)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            val_cer_history.append(val_cer)
            val_wer_history.append(val_wer)

            print(f"Epoca [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | CER: {val_cer:.4f} | WER: {val_wer:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), DECODER_CHECKPOINT)
                print(f"  -> Nuevo mejor modelo guardado (Val Loss: {val_loss:.4f})")
            else:
                epochs_no_improve += 1                
                if epochs_no_improve >= patience:
                    print(f"  -> Early stopping en la epoca {epoch + 1}.")
                    break

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido.")

    finally:
        if len(train_loss_history) > 0:
            print("Entrenamiento finalizado.")

            plt.figure(figsize=(8, 6))
            plt.plot(train_loss_history, label='Train Loss')
            plt.plot(val_loss_history, label='Validation Loss')
            plt.title('Gráfica Perdida')
            plt.xlabel('Épocas')
            plt.ylabel('Perdida')
            plt.legend()
            plt.savefig(WORK_DIR / "perdida.png")
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(val_cer_history, label='Validation CER', color='green')
            plt.plot(val_wer_history, label='Validation WER', color='red')
            plt.title('CER and WER')
            plt.xlabel('Épocas')
            plt.ylabel('Radio Error')
            plt.legend()
            plt.savefig(WORK_DIR / "metricas.png")
            plt.close()

            print("Graficas guardadas con exito.")

            if DECODER_CHECKPOINT.exists():
                print("Generando matriz de confusion")
                best_model = KichwaDecoder1D(input_dim=768, hidden_dim=256, vocab_size=len(vocabulary), dropout=0.25).to(device)
                best_model.load_state_dict(torch.load(DECODER_CHECKPOINT, map_location=device))

                confusion, vocab_chars = build_confusion_matrix(best_model, valid_loader, vocabulary, device)
                plot_confusion_matrix(confusion, vocab_chars, WORK_DIR / "matriz_confusion.png")
                print("Matriz de confusion guardada con exito.")

if __name__ == "__main__":
    main()