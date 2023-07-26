# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Fri Mar 17 22:14:42 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runcell(0, 'C:/Users/GOWTHAM.S/.spyder-py3/temp.py')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runcell(0, 'C:/Users/GOWTHAM.S/.spyder-py3/temp.py')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
nltk.download('stopwords')
# Preprocessing the text data
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
text = re.sub('[^a-zA-Z]', ' ', text)
text = text.lower()
text = nltk.word_tokenize(text)
text = [word for word in text if word not in stop_words]
text = ' '.join(text)
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install nltk
conda activate yourEnvName
pip install
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install
pip instal pandas
pip install pandas

## ---(Fri Mar 17 23:17:33 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install nltk

## ---(Fri Mar 17 23:19:15 2023)---
pip install panda
install pip

## ---(Fri Mar 17 23:35:04 2023)---
pip install panda

## ---(Fri Mar 17 23:57:13 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
nltk.download('stopwords')
nltk.download()
nltk.download('stopwords')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
nltk.download('punkt')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Sat Mar 18 09:02:17 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Sat Mar 18 14:47:18 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Sun Apr  2 21:54:25 2023)---
debugfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Mon Apr  3 15:56:11 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Tue Apr 11 21:07:01 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled0.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled0.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call spam.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Wed Apr 12 09:38:25 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call spam.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Wed Apr 12 10:34:39 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call spam.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Fri Apr 21 21:16:22 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Tue May  2 21:01:23 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled0.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install google-cloud-speech
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled0.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled1.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled0.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install deepspeech
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled0.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip uninstall deepspeech
conda install -c conda-forge deepspeech
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled0.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled1.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runcell(0, 'C:/Users/GOWTHAM.S/.spyder-py3/untitled1.py')

## ---(Tue May  2 22:07:56 2023)---
conda update anaconda
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip uninstall deepspeech

## ---(Tue May  2 22:55:29 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call spam.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install deepspeech
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install deepspeech --upgrade
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip uninstall deepspeech
pip install --verbose deepspeech
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip upgrade pip
python.exe -m pip install --upgrade pip
pip install --upgrade pip
pip install deepspeech
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install deepspeech==0.9.3
pip install numpy pyaudio
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip uninstall deepspeech
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install paddlehub
pip install pyqt5==5.12 pyqtwebengine==5.12
ip install pyqt5
pip install pyqt5
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install paddlepaddle
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install --upgrade paddlehub
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
hub install deepspeech2==2.0.0
pip install paddlepaddle
hub install deepspeech2
pip install deepspeech2
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip uninstall paddlehub
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install SpeechRecognition
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/anaconda3/lib/urllib/request.py', wdir='C:/Users/GOWTHAM.S/anaconda3/lib/urllib')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Wed May  3 14:42:55 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call spam.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
pip install SpeechRecognization
pip install SpeechRecognition
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Wed May  3 16:15:26 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Thu May  4 11:11:40 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call spam.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Fri May  5 11:01:13 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Fri May  5 13:16:02 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/temp.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Tue May  9 21:36:59 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Thu May 11 15:00:14 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/untitled0.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Tue May 16 14:37:28 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call spam.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')

## ---(Fri Jun  9 16:56:43 2023)---
runfile('C:/Users/GOWTHAM.S/.spyder-py3/call spam.py', wdir='C:/Users/GOWTHAM.S/.spyder-py3')