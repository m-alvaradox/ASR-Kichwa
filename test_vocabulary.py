from utils.vocabulary import Vocabulary

vocab = Vocabulary()

vocab.build("dataset/metadata.csv")

print("Tamaño del vocabulario:", len(vocab))
print()

print(vocab.char2idx)
print()

texto = "Alli"

indices = vocab.text_to_indices(texto)

print("Texto:", texto)
print("Índices:", indices)
print("Reconstrucción:", vocab.indices_to_text(indices))