from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
import librosa
import os
import shutil
from pathlib import Path

from utils.vocabulary import Vocabulary
from models.wav2vec import Wav2Vec2Partial
from models.decoder import KichwaDecoder1D

app = FastAPI(title="API Traductor Kichwa")
ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT_DIR / os.getenv("CHECKPOINT_DIR", "checkpoints")
AUDIO_EXAMPLES_DIR = ROOT_DIR / "audios prueba"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = None
extractor = None
decodificador = None

def load_partial_weights(model, checkpoint, num_layers_to_load=6):
    own_state = model.state_dict()
    loaded_keys, skipped_keys = [], []

    for name, param in checkpoint.items():
        mapped_name = None

        if name.startswith("feature_extractor.conv_layers"):
            mapped_name = name

        elif name.startswith("feature_projection.projection"):
            mapped_name = name.replace("feature_projection.projection", "feature_projection")

        elif name.startswith("encoder.pos_conv_embed"):
            mapped_name = name.replace("encoder.pos_conv_embed", "pos_conv_embed")
            mapped_name = mapped_name.replace("conv.weight_g", "conv.parametrizations.weight.original0")
            mapped_name = mapped_name.replace("conv.weight_v", "conv.parametrizations.weight.original1")

        elif name.startswith("encoder.layers"):
            layer_idx = int(name.split(".")[2])
            if layer_idx < num_layers_to_load:
                mapped_name = name.replace("encoder.layers", "layers")

        elif name in ("encoder.layer_norm.weight", "encoder.layer_norm.bias"):
            mapped_name = name.replace("encoder.layer_norm", "layer_norm")

        if mapped_name and mapped_name in own_state and own_state[mapped_name].shape == param.shape:
            own_state[mapped_name].copy_(param)
            loaded_keys.append(mapped_name)
        else:
            skipped_keys.append(name)

    model.load_state_dict(own_state)
    print(f"Cargadas {len(loaded_keys)} claves de {len(checkpoint)} totales en el checkpoint")
    print(f"Claves no cargadas: {len(skipped_keys)}")
    return model, loaded_keys, skipped_keys

# ----------------------------

@app.on_event("startup")
def cargar_modelos():
    global vocab, extractor, decodificador
    print("Iniciando servidor y cargando modelos en memoria...")

    vocab = Vocabulary()
    vocab.build(str(ROOT_DIR / "dataset" / "metadata_train.csv"))
    
    extractor = Wav2Vec2Partial(num_transformer_layers=6, dim=768).to(dispositivo)
    pesos_extractor = torch.load(CHECKPOINT_DIR / "wav2vec_small_clean.pt", map_location=dispositivo)
    
    extractor, _, _ = load_partial_weights(extractor, pesos_extractor, num_layers_to_load=6)
    extractor.eval()

    decodificador = KichwaDecoder1D(input_dim=768, hidden_dim=256, vocab_size=len(vocab)).to(dispositivo)
    pesos_decodificador = torch.load(CHECKPOINT_DIR / "kichwa_decoder_conv1d.pt", map_location=dispositivo)
    decodificador.load_state_dict(pesos_decodificador)
    decodificador.eval()

    print("¡Sistemas listos para inferencia!")

@app.post("/transcribir")
async def transcribir_audio(audio: UploadFile = File(...)):
    if not audio.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .wav o .mp3")

    temp_path = f"temp_{audio.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        waveform, _ = librosa.load(temp_path, sr=16000, mono=True)
        waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-5)
        tensor_audio = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(dispositivo)

        with torch.no_grad():
            embeddings = extractor(tensor_audio)
            logits = decodificador(embeddings)
            predicciones = torch.argmax(logits, dim=-1).squeeze(0) 

        indices_finales = []
        indice_previo = -1
        for idx in predicciones.tolist():
            if idx != indice_previo and idx != 0:
                indices_finales.append(idx)
            indice_previo = idx

        texto_final = vocab.indices_to_text(indices_finales)

    except Exception:
        raise HTTPException(status_code=500, detail="AUDIO_CORRUPTO")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "archivo": audio.filename,
        "transcripcion": texto_final
    }


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "device": str(dispositivo)}


@app.get("/api/audios-prueba", include_in_schema=False)
def listar_audios_prueba():
    if not AUDIO_EXAMPLES_DIR.exists():
        return []

    extensiones = {".wav", ".mp3"}
    return [
        {
            "nombre": archivo.name,
            "url": f"/audios-prueba/{archivo.name}",
        }
        for archivo in sorted(AUDIO_EXAMPLES_DIR.iterdir())
        if archivo.is_file() and archivo.suffix.lower() in extensiones
    ]


if AUDIO_EXAMPLES_DIR.exists():
    app.mount(
        "/audios-prueba",
        StaticFiles(directory=str(AUDIO_EXAMPLES_DIR)),
        name="audios-prueba",
    )


# El frontend y la API se publican en el mismo dominio en Railway.
app.mount(
    "/",
    StaticFiles(directory=str(ROOT_DIR / "frontend"), html=True),
    name="frontend",
)
