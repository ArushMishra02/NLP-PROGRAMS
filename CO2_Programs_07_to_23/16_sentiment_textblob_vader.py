# Program 16: Sentiment analysis using TextBlob and VADER
# Colab: !pip install -q textblob vaderSentiment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
reviews = ["This product is excellent and easy to use!", "The service was terrible and disappointing.", "The item arrived today and works as expected."]
vader = SentimentIntensityAnalyzer()
def label(x):
    return "Positive" if x > .05 else "Negative" if x < -.05 else "Neutral"
for r in reviews:
    tb = TextBlob(r).sentiment.polarity
    vd = vader.polarity_scores(r)["compound"]
    print("\nReview:", r)
    print(f"TextBlob: {tb:.3f} ({label(tb)})")
    print(f"VADER: {vd:.3f} ({label(vd)})")
