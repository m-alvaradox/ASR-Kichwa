import os
import librosa
import soundfile as sf
from tqdm import tqdm

INPUT_DIR = "raw"
OUTPUT_DIR = "wav"

TARGET_SR = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in tqdm(os.listdir(INPUT_DIR)):

    if not filename.endswith(".mp3"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    # Cargar audio (mono=True convierte automáticamente a un canal)
    audio, sr = librosa.load(
        input_path,
        sr=TARGET_SR,
        mono=True
    )

    output_name = filename.replace(".mp3", ".wav")
    output_path = os.path.join(OUTPUT_DIR, output_name)

    sf.write(
        output_path,
        audio,
        TARGET_SR
    )

print("Conversión completada.")