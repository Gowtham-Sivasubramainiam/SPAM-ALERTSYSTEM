from flask import Flask,render_template,request,redirect, url_for
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import mysql.connector
app = Flask(__name__, template_folder='templates')
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='python'
)
a11 = ''
@app.route('/')
def hello_world():
    return render_template('index.html')
@app.route('/as1',methods=['POST','GET'])
def as1():
    data = pd.read_csv('spam.csv', encoding='latin-1')

    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = [word for word in text if word not in stop_words]
        text = ' '.join(text)
        print(text)
        return text

    data['v2'] = data['v2'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)


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
    global a11
    new_message = request.form['a']
    a11 = '' + new_message
    print("YOUR INPUT:", a11)
    new_message = preprocess_text(new_message)
    new_message_vector = vectorizer.transform([new_message])
    prediction = clf.predict(new_message_vector)
    if prediction[0] == 'spam':
        return render_template('spam.html')
    else:
        return render_template('notspam.html')
@app.route('/ad')
def ad():
    return render_template('admin.html')


cursor = db.cursor()
cursor1 = db.cursor()
@app.route('/admin',methods=['POST','GET'])
def admin():
        email = request.form['email']
        password = request.form['password']

        query = "SELECT * FROM user WHERE email=%s AND password=%s"
        cursor.execute(query, (email, password))
        result = cursor.fetchone()

        if result:
            response = "Login successful"

            return redirect(url_for('display_table'))
        else:
            response = "Invalid credentials"
            return render_template('admin.html',response=response)


@app.route('/fee',methods=['POST','GET'])
def fee():
    feedback = request.form['b']
    msg=a11
    query = "INSERT INTO user_feedback (message, feedback) VALUES (%s, %s)"
    a1=cursor.execute(query, (msg, feedback))
    db.commit()
    return render_template('thank.html')
@app.route('/table')
def display_table():
    cursor1.execute("Delete  FROM user_feedback where feedback=null ")
    cursor.execute("SELECT * FROM user_feedback ")
    table_data = cursor.fetchall()
    for x in table_data:
        print(x[0])
    return render_template('ab.html', table_data=table_data)

if __name__ == '__main__':
    app.run()