# Program 1: Tokenization of Sentences and Words using NLTK and spaCy

# Install once in Google Colab:
# !pip install nltk spacy
# !python -m spacy download en_core_web_sm

import nltk
import spacy

nltk.download("punkt")
nltk.download("punkt_tab")

text = "Natural Language Processing is interesting. It helps computers understand human language!"

# ---------------- NLTK ----------------
print("========== NLTK TOKENIZATION ==========")

nltk_sentences = nltk.sent_tokenize(text)
nltk_words = nltk.word_tokenize(text)

print("Sentences:")
for s in nltk_sentences:
    print(s)

print("\nWords:")
print(nltk_words)

# ---------------- spaCy ----------------
print("\n========== spaCy TOKENIZATION ==========")

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print("Sentences:")
for sent in doc.sents:
    print(sent.text)

print("\nWords/Tokens:")
print([token.text for token in doc])
