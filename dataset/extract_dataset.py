import os
import pandas as pd
import pyarrow.parquet as pq

# -----------------------
# Configuración
# -----------------------

PARQUET_FILE = "parquet/train-00000-of-00001.parquet"

OUTPUT_AUDIO = "raw"

CSV_OUTPUT = "metadata.csv"

os.makedirs(OUTPUT_AUDIO, exist_ok=True)

# -----------------------
# Leer parquet
# -----------------------

table = pq.read_table(PARQUET_FILE)

df = table.to_pandas()

metadata = []

# -----------------------
# Extraer audios
# -----------------------

for i, row in df.iterrows():

    audio = row["audio"]

    sentence = row["sentence"]

    filename = f"{i:06d}.mp3"

    filepath = os.path.join(OUTPUT_AUDIO, filename)

    with open(filepath, "wb") as f:
        f.write(audio["bytes"])

    metadata.append({
        "filename": filename,
        "sentence": sentence
    })

# -----------------------
# Guardar CSV
# -----------------------

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")

print("Proceso terminado.")

print(f"Audios extraídos: {len(metadata_df)}")

print(f"CSV generado: {CSV_OUTPUT}")