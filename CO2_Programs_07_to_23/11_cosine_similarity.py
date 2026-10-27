# Program 11: Cosine similarity between text documents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = ["The cat sits on the mat", "A cat is sitting on a mat", "The stock market changed today"]
x = TfidfVectorizer().fit_transform(docs)
scores = cosine_similarity(x)
print(scores.round(3))
print("Similarity document 1 vs 2:", round(scores[0, 1], 3))
print("Similarity document 1 vs 3:", round(scores[0, 2], 3))
