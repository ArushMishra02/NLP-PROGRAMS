# Program 17: Topic modeling using Latent Dirichlet Allocation (LDA)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
docs = ["football team scored goals match", "coach trained football players", "python code uses machine learning", "developers build learning systems", "team won football game", "data science uses python machine learning"]
v = CountVectorizer(stop_words="english")
x = v.fit_transform(docs)
model = LatentDirichletAllocation(n_components=2, random_state=42, max_iter=20).fit(x)
terms = v.get_feature_names_out()
for i, weights in enumerate(model.components_, 1):
    print(f"Topic {i}:", ", ".join(terms[j] for j in weights.argsort()[-6:][::-1]))
print("Document-topic distributions:\n", model.transform(x).round(3))
