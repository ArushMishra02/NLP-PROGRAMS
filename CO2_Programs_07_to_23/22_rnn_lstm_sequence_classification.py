# Program 22: Sequence classification using RNN/LSTM (TensorFlow)
# Colab: !pip install -q tensorflow
import numpy as np, tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
texts = ["love this movie","wonderful film","great acting and story","excellent film","fantastic movie","hate this movie","terrible film","boring story","awful film","horrible movie"]
y = np.array([1]*5 + [0]*5, dtype="float32")
tok = Tokenizer(num_words=1000, oov_token="<OOV>"); tok.fit_on_texts(texts)
X = pad_sequences(tok.texts_to_sequences(texts), maxlen=8, padding="post")
tf.random.set_seed(42)
model = Sequential([Embedding(1000, 16), LSTM(16), Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=12, verbose=0)
print("Training accuracy:", round(model.evaluate(X, y, verbose=0)[1], 3))
test = ["fantastic movie"]
seq = pad_sequences(tok.texts_to_sequences(test), maxlen=8, padding="post")
score = float(model.predict(seq, verbose=0)[0][0])
print(test[0], "->", "Positive" if score >= .5 else "Negative", "probability:", round(score, 3))
