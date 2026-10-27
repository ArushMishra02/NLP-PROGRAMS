# Program 14: Text similarity using Word Mover's Distance (WMD)
# Colab: !pip install -q gensim nltk
import re, nltk, gensim.downloader as api
nltk.download("stopwords")
from nltk.corpus import stopwords
print("Loading GloVe vectors; first run may take a few minutes.")
vectors = api.load("glove-wiki-gigaword-50")
stop = set(stopwords.words("english"))
def tokens(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in stop and w in vectors]
a = tokens("the cat sits on the mat")
b = tokens("a cat is sitting on a rug")
c = tokens("the government announced new economic policy")
print("Doc1 tokens:", a, "\nDoc2 tokens:", b, "\nDoc3 tokens:", c)
print("WMD Doc1 vs Doc2:", vectors.wmdistance(a, b))
print("WMD Doc1 vs Doc3:", vectors.wmdistance(a, c))
