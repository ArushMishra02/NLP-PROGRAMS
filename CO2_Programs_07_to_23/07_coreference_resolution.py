# Program 7: Coreference Resolution using FastCoref (neuralcoref alternative)
# Colab: !pip install -q spacy fastcoref
# Colab: !python -m spacy download en_core_web_sm
import spacy
from fastcoref import FCoref
text = "Riya bought a laptop yesterday. She said it was fast, and she uses the laptop for classes."
nlp = spacy.load("en_core_web_sm")
model = FCoref(device="cpu")
prediction = model.predict(texts=[text])[0]
print("Text:", text)
print("Coreference clusters:")
for i, cluster in enumerate(prediction.get_clusters(as_strings=True), 1):
    print(f"Cluster {i}: {cluster}")
