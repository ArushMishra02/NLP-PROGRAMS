# Program 18: Topic modeling using Latent Semantic Analysis (LSA)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
docs = ["football players train match", "team scored goals football", "python machine learning data science", "learning models analyze data", "football team won match", "data science uses python"]
v = TfidfVectorizer(stop_words="english")
x = v.fit_transform(docs)
model = TruncatedSVD(n_components=2, random_state=42).fit(x)
terms = v.get_feature_names_out()
for i, weights in enumerate(model.components_, 1):
    print(f"Topic {i}:", ", ".join(terms[j] for j in weights.argsort()[-6:][::-1]))
print("Document-topic representation:\n", model.transform(x).round(3))
