# Program 8: Bag-of-Words vectorization and representation
from sklearn.feature_extraction.text import CountVectorizer
docs = ["NLP helps computers understand language", "Computers process language using NLP", "Language processing is useful"]
v = CountVectorizer()
x = v.fit_transform(docs)
print("Vocabulary:", v.get_feature_names_out())
print("BoW matrix (documents x words):")
print(x.toarray())
