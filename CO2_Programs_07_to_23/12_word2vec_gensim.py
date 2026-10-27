# Program 12: Word2Vec embeddings using gensim on a custom corpus
# Colab: !pip install -q gensim nltk
from gensim.models import Word2Vec
from nltk.tokenize import wordpunct_tokenize
corpus = ["nlp models learn words from context", "word embeddings represent words as vectors", "context helps models learn word meaning", "nlp uses embeddings for language tasks", "models learn patterns from text"]
sentences = [wordpunct_tokenize(s.lower()) for s in corpus]
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=1, sg=1, seed=42, epochs=100)
print("Vector for models:", model.wv["models"])
print("Similar words:", model.wv.most_similar("models", topn=3))
