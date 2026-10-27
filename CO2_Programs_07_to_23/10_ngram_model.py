# Program 10: Generate unigram, bigram, and trigram models
from sklearn.feature_extraction.text import CountVectorizer
corpus = ["natural language processing is useful", "natural language models learn patterns", "language processing uses statistical models"]
for n, name in [(1, "Unigrams"), (2, "Bigrams"), (3, "Trigrams")]:
    v = CountVectorizer(ngram_range=(n, n))
    x = v.fit_transform(corpus)
    print("\n" + name)
    for term, count in zip(v.get_feature_names_out(), x.toarray().sum(axis=0)):
        print(f"{term}: {count}")
