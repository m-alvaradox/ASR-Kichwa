---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: sentence
    dtype: string
  splits:
  - name: train
    num_bytes: 184050207.062
    num_examples: 3002
  - name: valid
    num_bytes: 22788399
    num_examples: 375
  - name: test
    num_bytes: 22656433
    num_examples: 376
  download_size: 225437827
  dataset_size: 229495039.062
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: valid
    path: data/valid-*
  - split: test
    path: data/test-*
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
language:
- qu
- qvi
- qug
- qxl
- qvj
- qxr
- qud
size_categories:
- 1K<n<10K
---

# Killkan: Speech Recognition dataset for Kichwa
Killkan (Kichwa uyachkata payllatak killkak anta) is the first automatic speech recognition (ASR) dataset for the Kichwa language.
See also our paper (https://arxiv.org/abs/2404.15501).