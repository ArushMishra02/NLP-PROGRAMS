# Program 13: Load GloVe embeddings and inspect vectors
# Colab: !pip install -q gensim
import gensim.downloader as api
print("Downloading/loading GloVe model; first run may take a few minutes.")
glove = api.load("glove-wiki-gigaword-50")
for word in ["computer", "language", "king"]:
    if word in glove:
        print(word, "shape:", glove[word].shape, "first 10 values:", glove[word][:10])
print("Words similar to computer:", glove.most_similar("computer", topn=5))
