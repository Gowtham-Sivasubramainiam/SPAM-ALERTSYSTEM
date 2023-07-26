
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#nltk.download('stopwords')
# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
#nltk.download('stopwords')
# Preprocessing the text data
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

data['v2'] = data['v2'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

# Feature extraction using bag-of-words
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


clf = MultinomialNB()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)


new_message = "YOU HAVE WON A CASH PRIZE OF 100000"
print("YOUR INPUT:",new_message)
new_message = preprocess_text(new_message)
new_message_vector = vectorizer.transform([new_message])
prediction = clf.predict(new_message_vector)
if prediction[0] == 'spam':
    print("ALERT: Possible spam detected!")
else:
    print("Message is not spam.")