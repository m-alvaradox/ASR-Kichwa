# ASR-Kichwa

Sistema de reconocimiento automático del habla (ASR) que recibe un audio en
Kichwa y devuelve su transcripción. El proyecto usa un extractor acústico
basado en Wav2Vec2, un decodificador convolucional entrenado con CTC, una API
FastAPI y una interfaz web estática.

> Este proyecto transcribe Kichwa a texto en Kichwa; no realiza traducción a
> español.

## Arquitectura general

```text
Audio WAV/MP3
    -> conversión a mono y 16 kHz
    -> extractor Wav2Vec2 parcial (6 capas Transformer)
    -> embeddings de dimensión 768
    -> decodificador Conv1D
    -> clasificación por caracteres
    -> decodificación CTC greedy
    -> texto
```

## Requisitos

- Windows 10/11, Linux o macOS.
- Python 3.10 o 3.11 (se recomienda 3.11).
- Al menos 8 GB de RAM.
- Espacio adicional para los checkpoints y los embeddings.
- GPU NVIDIA compatible con CUDA (opcional). En CPU funciona más lentamente.
- Dos archivos de pesos exportados o descargados desde el notebook de
  entrenamiento.

No es necesario instalar Node.js: el frontend usa HTML, CSS y JavaScript sin
proceso de compilación.

## Archivos que deben obtenerse del notebook o carpetas

Los modelos no están incluidos en Git porque `checkpoints/` está excluido en
`.gitignore`. SI ES EL PROFESOR Y ESTA LA CARPETA NO ES NECESARIO AGREGAR LO SIGUIENTE:
Después de ejecutar el notebook, descargar o copiar estos
artefactos:

| Archivo generado | Ubicación en el proyecto | Uso |
| --- | --- | --- |
| `wav2vec_small_clean.pt` | `checkpoints/wav2vec_small_clean.pt` | Pesos del extractor Wav2Vec2 |
| `kichwa_decoder_conv1d.pt` | `checkpoints/kichwa_decoder_conv1d.pt` | Pesos del decodificador Kichwa |

La estructura mínima debe quedar así:

```text
ASR-Kichwa/
|-- checkpoints/
|   |-- wav2vec_small_clean.pt
|   `-- kichwa_decoder_conv1d.pt
|-- dataset/
|   `-- metadata_train.csv
|-- frontend/
|-- models/
|-- utils/
|-- main.py
`-- requirements.txt
```

### Vocabulario del notebook

El checkpoint del decodificador y el vocabulario son inseparables. El notebook
debe conservar el mismo mapeo `char2idx` usado durante el entrenamiento. Se
recomienda exportarlo desde el notebook:

```python
import json

with open("vocab.json", "w", encoding="utf-8") as archivo:
    json.dump(vocab.char2idx, archivo, ensure_ascii=False, indent=2)
```

Después debe copiarse como `checkpoints/vocab.json`. La aplicación actualmente
reconstruye el vocabulario leyendo `dataset/metadata_train.csv`; por tanto, ese
CSV debe ser exactamente la versión usada durante el entrenamiento, con la
misma limpieza de texto y los mismos caracteres.

Para verificar el número de salidas desde el notebook:

```python
import torch

checkpoint = torch.load("kichwa_decoder_conv1d.pt", map_location="cpu")
print(checkpoint["classifier.weight"].shape)
print("Tamaño del vocabulario:", len(vocab))
```

El primer número del `shape` debe coincidir con `len(vocab)`. Por ejemplo,
`torch.Size([36, 512])` requiere un vocabulario de 36 símbolos.

## Instalación del ambiente

Abrir PowerShell o una terminal en la raíz del repositorio.

### Windows (PowerShell)

```powershell
cd D:\m_alv\Github\ASR-Kichwa
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

El comando `py` no está disponible en todas las instalaciones de Windows. Si
no se reconoce, utilizar `python`, como muestran los comandos anteriores.

### Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecutar el proyecto

Se necesitan dos terminales: una para la API y otra para la interfaz web. Los
comandos deben ejecutarse desde la raíz del proyecto.

### 1. Iniciar la API

Con el ambiente virtual activado:

```powershell
python -m uvicorn main:app --reload
```

- API: <http://localhost:8000>
- Documentación interactiva: <http://localhost:8000/docs>

Durante el inicio deben aparecer mensajes indicando que se cargaron los dos
modelos y que el sistema está listo para inferencia.

### 2. Iniciar el frontend

En otra terminal:

```powershell
cd D:\m_alv\Github\ASR-Kichwa
.\.venv\Scripts\Activate.ps1
python -m http.server 5500 --directory frontend
```

Si se conserva el ambiente antiguo llamado `venv`:

```powershell
.\venv\Scripts\python.exe -m http.server 5500 --directory frontend
```

Abrir <http://localhost:5500>, seleccionar un archivo `.wav` del grupo de muestra y pulsar
el botón de transcripción. No se recomienda abrir `index.html` directamente con
doble clic.

## Probar la API sin el frontend

Desde otra ventana de PowerShell:

```powershell
curl.exe -X POST `
  -F "audio=@D:\ruta\al\audio.wav" `
  http://localhost:8000/transcribir
```

Respuesta esperada:

```json
{
  "archivo": "audio.wav",
  "transcripcion": "texto reconocido en kichwa"
}
```

También se puede probar `POST /transcribir` desde
<http://localhost:8000/docs>.

## Despliegue en una máquina o servidor

Para ejecutar la API sin recarga automática:

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Para servir el frontend en la misma máquina:

```powershell
python -m http.server 5500 --bind 0.0.0.0 --directory frontend
```

El frontend consulta actualmente la dirección fija
`http://localhost:8000/transcribir`. Si la API está en otra computadora o
dominio, cambiar esa URL en `frontend/js/app.js`.

En un despliegue público se recomienda colocar ambos servicios detrás de HTTPS
y limitar la configuración CORS de `main.py`; actualmente se permiten
solicitudes desde cualquier origen.

## Solución de problemas

### `py` no se reconoce

Usar `python`:

```powershell
python -m http.server 5500 --directory frontend
```

O llamar directamente al intérprete:

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --directory frontend
```

### No existe `checkpoints/...`

Crear la carpeta `checkpoints` y copiar los dos archivos `.pt` generados por el
notebook. Estos archivos no se descargan al clonar el repositorio.

### Error `size mismatch for classifier.weight`

Ejemplo:

```text
checkpoint: [36, 512]
modelo actual: [30, 512]
```

El decodificador fue entrenado con un vocabulario diferente del que produce
`metadata_train.csv`. La solución correcta es copiar el CSV y/o `vocab.json`
exactos del entrenamiento, o volver a entrenar el decodificador con el
vocabulario actual. Cambiar solamente `vocab_size` permite igualar dimensiones,
pero no recupera el significado de los índices y puede generar texto erróneo.

### Mensaje `Claves no cargadas` al iniciar

El proyecto carga únicamente seis capas Transformer del checkpoint Wav2Vec2.
Por eso pueden omitirse claves de capas que no forman parte del modelo parcial.
El mensaje es informativo si la carga continúa; no equivale a un `size
mismatch` del clasificador.

### La interfaz no puede conectarse

Comprobar que:

1. La API siga ejecutándose en `http://localhost:8000`.
2. Los modelos hayan terminado de cargar sin excepciones.
3. El frontend se haya abierto desde `http://localhost:5500`.
4. El archivo seleccionado sea `.wav` o `.mp3`.

## Entrenamiento y embeddings

`training/train.py` está diseñado para entrenar el decodificador a partir de
embeddings ya calculados en:

```text
embeddings/train/
embeddings/valid/
```

Tanto `embeddings/` como `checkpoints/` están excluidos de Git por su tamaño.
Para reproducir el entrenamiento hacen falta:

- Los audios originales.
- Los metadatos de entrenamiento y validación.
- El checkpoint Wav2Vec2 de origen.
- Los embeddings generados por el notebook.
- El vocabulario exacto.

El script de entrenamiento actual necesita completar la construcción del
`DataLoader` y su función de padding (`collate_fn`) antes de utilizarse de forma
independiente. Por ahora, el notebook original es una parte necesaria del flujo
reproducible de entrenamiento.

## Archivos principales

- `main.py`: API, carga de modelos e inferencia.
- `models/wav2vec.py`: extractor Wav2Vec2 parcial.
- `models/decoder.py`: decodificador convolucional Kichwa.
- `utils/vocabulary.py`: codificación y decodificación de caracteres.
- `data/dataset.py`: lectura de audios o embeddings.
- `training/train.py`: lógica de entrenamiento del decodificador.
- `frontend/`: interfaz web.
