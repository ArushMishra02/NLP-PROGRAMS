# Program 4: Part-of-Speech (POS) Tagging of a Given Sentence

# Install once in Google Colab:
# !pip install nltk

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

sentence = "The intelligent student solved the difficult problem quickly."

words = word_tokenize(sentence)
tags = pos_tag(words)

print("========== POS TAGGING ==========")
print("Sentence:", sentence)
print("\nWord -> POS Tag")

for word, tag in tags:
    print(f"{word:15} -> {tag}")

print("\nCommon POS tag examples:")
print("NN  = Noun")
print("VB  = Verb")
print("JJ  = Adjective")
print("RB  = Adverb")
print("DT  = Determiner")
