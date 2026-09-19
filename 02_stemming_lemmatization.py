# Program 2: Stemming and Lemmatization on Sample Text

# Install once in Google Colab:
# !pip install nltk
# !python -m nltk.downloader punkt wordnet omw-1.4

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

text = "The students are studying studies about running runners and easily solved problems."

words = nltk.word_tokenize(text)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Original Words:")
print(words)

print("\n========== STEMMING ==========")
for word in words:
    if word.isalpha():
        print(f"{word:15} -> {stemmer.stem(word)}")

print("\n========== LEMMATIZATION ==========")
for word in words:
    if word.isalpha():
        print(f"{word:15} -> {lemmatizer.lemmatize(word)}")
