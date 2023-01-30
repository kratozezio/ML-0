import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# st.write("""
#     # Audio
# """)

# st.audio("samples/Bryan_-_The_Ideal_Republic.ogg")
st.title("h")
df = pd.read_csv("../datasets/hotel_reviews.csv", encoding = "unicode_escape")
st.write("Original Dataset", df.head())
df = df.drop(['ID'], axis=1)
st.write("Modified Dataset", df.head())

X = df['Reviews']
y = df['Label']
st.write(f"""
**Independent Feature:** {X.name}\n
**Dependent Feature:** {y.name}
""")
X_train, X_test, y_train, y_test = train_test_split(X, y)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train.values.astype('U'))

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# X_test_vec = vectorizer.transform(X_test)
# predicted = model.predict(X_test_vec)
# st.write("Predicted Values", predicted)

test_reviews = ['Hotel was good', 'Hotel was trash', 'Hotel was bad']
test_reviews_vec = vectorizer.transform(test_reviews)
predicted = model.predict(test_reviews_vec)
st.write(predicted)