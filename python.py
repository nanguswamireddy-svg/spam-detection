import os
import re
import io
import base64
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevents crashing on some systems
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

app = Flask(__name__)

class SpamDetectionSystem:
    def __init__(self):
        self.data_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        # Expanded stop words for better accuracy
        self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
        self.model = None
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.conf_matrix_img = ""
        self.full_report = {}

    def text_preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        return [w for w in tokens if w not in self.stop_words and len(w) > 2]

    def initialize_engine(self):
        try:
            print("[INFO] Loading Dataset...")
            response = requests.get(self.data_url).content
            df = pd.read_csv(io.StringIO(response.decode('utf-8')), sep='\t', header=None, names=['label', 'message'])
            
            X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.25, random_state=42)
            
            self.model = Pipeline([
                ('vectorizer', CountVectorizer(analyzer=self.text_preprocess)),
                ('tfidf', TfidfTransformer()),
                ('classifier', MultinomialNB())
            ])
            
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            
            self.accuracy = round(accuracy_score(y_test, predictions) * 100, 1)
            self.precision = round(precision_score(y_test, predictions, pos_label='spam') * 100, 1)
            self.recall = round(recall_score(y_test, predictions, pos_label='spam') * 100, 1)
            self.full_report = classification_report(y_test, predictions, output_dict=True)
            
            self.generate_visuals(y_test, predictions)
            print(f"[SUCCESS] System Online - Accuracy: {self.accuracy}%")
        except Exception as e:
            print(f"[ERROR] Startup Failed: {e}")

    def generate_visuals(self, y_true, y_pred):
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Purples', 
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        self.conf_matrix_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

detector = SpamDetectionSystem()
detector.initialize_engine()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        msg = request.form.get('message_content')
        if msg:
            label = detector.model.predict([msg])[0]
            # Use predict_proba to get the confidence percentage
            prob = np.max(detector.model.predict_proba([msg])) * 100
            result = {
                "category": label.upper(),
                "confidence": round(prob, 1),
                "text": msg
            }
            
    return render_template('index.html', 
                           accuracy=detector.accuracy,
                           precision=detector.precision,
                           recall=detector.recall,
                           report=detector.full_report,
                           plot_url=detector.conf_matrix_img,
                           result=result)

if __name__ == '__main__':
    # Using port 5000 as it is standard for Flask
    app.run(debug=True, port=5000)