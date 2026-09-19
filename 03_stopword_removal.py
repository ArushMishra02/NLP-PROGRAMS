# Program 3: Stop-word Removal from a Document

# Install once in Google Colab:
# !pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

document = """
Natural Language Processing is a field of Artificial Intelligence.
It helps computers process, understand, and generate human language.
"""

stop_words = set(stopwords.words("english"))
words = word_tokenize(document)

filtered_words = [
    word for word in words
    if word.isalpha() and word.lower() not in stop_words
]

print("========== ORIGINAL DOCUMENT ==========")
print(document)

print("========== TOKENS ==========")
print(words)

print("\n========== STOP WORDS REMOVED ==========")
print(filtered_words)

print("\n========== FINAL DOCUMENT ==========")
print(" ".join(filtered_words))
