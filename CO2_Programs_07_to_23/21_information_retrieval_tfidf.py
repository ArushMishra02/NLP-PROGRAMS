# Program 21: Information retrieval with TF-IDF ranking
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = ["Python is popular for data science and machine learning.", "NLP helps computers work with human text.", "Deep learning uses neural networks.", "Information retrieval ranks relevant documents.", "TF-IDF represents documents with weighted word frequencies."]
query = "document ranking using TF-IDF and information retrieval"
v = TfidfVectorizer(stop_words="english")
x = v.fit_transform(docs); q = v.transform([query])
scores = cosine_similarity(q, x).ravel()
for rank, i in enumerate(scores.argsort()[::-1], 1):
    print(f"{rank}. Score={scores[i]:.3f} | {docs[i]}")
