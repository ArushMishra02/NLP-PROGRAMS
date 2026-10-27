# Program 9: TF-IDF implementation and comparison with BoW
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
docs = ["machine learning processes text", "text mining uses machine learning", "deep learning processes images"]
bow = CountVectorizer()
b = bow.fit_transform(docs)
tf = TfidfVectorizer()
t = tf.fit_transform(docs)
print("BoW:")
print(pd.DataFrame(b.toarray(), columns=bow.get_feature_names_out()))
print("\nTF-IDF:")
print(pd.DataFrame(t.toarray().round(3), columns=tf.get_feature_names_out()))
print("\nBoW uses raw counts; TF-IDF downweights terms common across documents.")
