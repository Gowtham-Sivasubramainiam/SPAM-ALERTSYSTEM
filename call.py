import os
import speech_recognition as sr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define a function to transcribe an audio file using SpeechRecognition
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as audio_file:
        audio_data = recognizer.record(audio_file)
    transcription = recognizer.recognize_google(audio_data)
    return transcription

# Load the labeled dataset
dataset = pd.read_csv("transcripe.csv")

# Transcribe the input audio file
transcription = transcribe_audio("234.wav")

# Train a logistic regression model on the labeled dataset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset["transcription"])
y = dataset["label"]
model = LogisticRegression()
model.fit(X, y)

# Use the trained model to classify the input audio file
X_test = vectorizer.transform([transcription])
y_pred = model.predict(X_test)

# Print the result
if y_pred[0] == 1:
    print("Spam")
else:
    print("Not spam")
