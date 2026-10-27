# Program 15: Text classification using Naive Bayes/SVM with TF-IDF
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
texts = ["team won cricket match","player scored a goal","coach planned training","football player trained","cricket match starts","team lost final game","new phone camera","software update fixed laptop","computer has fast memory","laptop battery lasts long","phone has bright screen","install new software"]
labels = ["sports"]*6 + ["technology"]*6
a,b,y,z = train_test_split(texts, labels, test_size=.33, random_state=42, stratify=labels)
for name, clf in [("Naive Bayes", MultinomialNB()), ("Linear SVM", LinearSVC())]:
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf)])
    model.fit(a, y); pred = model.predict(b)
    print("\n", name, "accuracy:", round(accuracy_score(z, pred), 3))
    print(classification_report(z, pred, zero_division=0))
