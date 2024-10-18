from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load the model and CountVectorizer
model = pickle.load(open('spam-sms-mnb-model.pkl', 'rb'))
cv = pickle.load(open('cv-transform.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        # Preprocess the message
        corpus = []
        for i in range(0, len(data)):
            text = re.sub("[^a-zA-Z0-9]", " ", data[i])
            text = text.lower()
            text = text.split()
            pe = PorterStemmer()
            stop_words = stopwords.words("english")  
            text = [pe.stem(word) for word in text if not word in set(stop_words)]
            text = " ".join(text)
            corpus.append(text)

        vect = cv.transform(corpus).toarray()
        my_prediction = model.predict(vect)
        
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)