from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import librosa
import os
import shutil

from utils.vocabulary import Vocabulary
from models.wav2vec import Wav2Vec2Partial
from models.decoder import KichwaDecoder1D

app = FastAPI(title="API Traductor Kichwa")

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
    loaded_keys = []
    q_w, k_w, v_w = {}, {}, {}
    q_b, k_b, v_b = {}, {}, {}

    for name, param in checkpoint.items():
        clean_name = name.replace('wav2vec2.', '')

        if 'feature_extractor' in clean_name:
            mapped_name = clean_name.replace('feature_extractor', 'feature_encoder')
            if mapped_name in own_state and own_state[mapped_name].shape == param.shape:
                own_state[mapped_name].copy_(param); loaded_keys.append(mapped_name)

        elif 'feature_projection.projection' in clean_name:
            mapped_name = clean_name.replace('feature_projection.projection', 'feature_projection')
            if mapped_name in own_state and own_state[mapped_name].shape == param.shape:
                own_state[mapped_name].copy_(param); loaded_keys.append(mapped_name)

        elif 'encoder.layers' in clean_name:
            parts = clean_name.split('.')
            layer_idx = int(parts[parts.index('layers') + 1])
            
            if layer_idx >= num_layers_to_load: continue
            base_name = f'transformer_layers.{layer_idx}'

            if 'attention.q_proj' in clean_name:
                if 'weight' in clean_name: q_w[layer_idx] = param
                else: q_b[layer_idx] = param
            elif 'attention.k_proj' in clean_name:
                if 'weight' in clean_name: k_w[layer_idx] = param
                else: k_b[layer_idx] = param
            elif 'attention.v_proj' in clean_name:
                if 'weight' in clean_name: v_w[layer_idx] = param
                else: v_b[layer_idx] = param
            elif 'attention.out_proj' in clean_name:
                mapped_name = f'{base_name}.attention.out_proj.' + clean_name.split('.')[-1]
                if mapped_name in own_state: own_state[mapped_name].copy_(param); loaded_keys.append(mapped_name)
            
            elif 'layer_norm' in clean_name and 'final' not in clean_name:
                mapped_name = f'{base_name}.norm1.' + clean_name.split('.')[-1]
                if mapped_name in own_state: own_state[mapped_name].copy_(param); loaded_keys.append(mapped_name)
            elif 'final_layer_norm' in clean_name:
                mapped_name = f'{base_name}.norm2.' + clean_name.split('.')[-1]
                if mapped_name in own_state: own_state[mapped_name].copy_(param); loaded_keys.append(mapped_name)
            
            elif 'intermediate_dense' in clean_name:
                mapped_name = f'{base_name}.ff.0.' + clean_name.split('.')[-1]
                if mapped_name in own_state: own_state[mapped_name].copy_(param); loaded_keys.append(mapped_name)
            elif 'output_dense' in clean_name:
                mapped_name = f'{base_name}.ff.2.' + clean_name.split('.')[-1]
                if mapped_name in own_state: own_state[mapped_name].copy_(param); loaded_keys.append(mapped_name)

    for idx in q_w.keys():
        if idx in k_w and idx in v_w:
            in_proj_w = torch.cat([q_w[idx], k_w[idx], v_w[idx]], dim=0)
            own_state[f'transformer_layers.{idx}.attention.in_proj_weight'].copy_(in_proj_w)
            loaded_keys.append(f'transformer_layers.{idx}.attention.in_proj_weight')
        if idx in q_b and idx in k_b and idx in v_b:
            in_proj_b = torch.cat([q_b[idx], k_b[idx], v_b[idx]], dim=0)
            own_state[f'transformer_layers.{idx}.attention.in_proj_bias'].copy_(in_proj_b)
            loaded_keys.append(f'transformer_layers.{idx}.attention.in_proj_bias')

    model.load_state_dict(own_state, strict=False)
    print(f"Auditoría: {len(loaded_keys)} tensores mapeados con éxito en la arquitectura nativa.")
    return model

# ----------------------------

@app.on_event("startup")
def cargar_modelos():
    global vocab, extractor, decodificador
    print("Iniciando servidor y cargando modelos en memoria...")

    vocab = Vocabulary()
    vocab.build("dataset/metadata_train.csv")
    
    extractor = Wav2Vec2Partial(num_transformer_layers=6, dim=768).to(dispositivo)
    pesos_extractor = torch.load("checkpoints/wav2vec_small_clean.pt", map_location=dispositivo)
    
    extractor = load_partial_weights(extractor, pesos_extractor, num_layers_to_load=6) 
    extractor.eval()

    decodificador = KichwaDecoder1D(input_dim=768, hidden_dim=512, vocab_size=len(vocab)).to(dispositivo)
    pesos_decodificador = torch.load("checkpoints/kichwa_decoder_conv1d.pt", map_location=dispositivo)
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

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "archivo": audio.filename,
        "transcripcion": texto_final
    }