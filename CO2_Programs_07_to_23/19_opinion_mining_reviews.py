# Program 19: Opinion mining on product/service reviews
import pandas as pd
from textblob import TextBlob
reviews = [
("Headphones", "Excellent sound quality and comfortable fit"),
("Headphones", "Battery is poor and connection drops"),
("Laptop", "Fast performance and beautiful display"),
("Laptop", "Expensive and gets very hot"),
("Delivery", "Delivery was quick and package was safe"),
("Delivery", "Parcel arrived late and support was unhelpful")]
rows = []
for product, review in reviews:
    p = TextBlob(review).sentiment.polarity
    sentiment = "Positive" if p > .05 else "Negative" if p < -.05 else "Neutral"
    rows.append({"product": product, "review": review, "polarity": round(p, 3), "sentiment": sentiment})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\nSentiment counts by product/service:\n", pd.crosstab(df.product, df.sentiment))
