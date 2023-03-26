from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
model = pickle.load(open('model.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the sentiment analysis endpoint
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Get the text input from the request body
    input_text = request.form['text']

    # Tokenize and pad the input text
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=100)

    # Make a prediction using the loaded model
    prediction = model.predict(input_seq)[0][0]

    # Convert the prediction to a label
    label = 'positive' if prediction >= 0.3 else 'negative'

    # Return the prediction as a JSON response
    response = {
        'prediction': label,
        'confidence': float(prediction)
    }
    return render_template('index.html', prediction=response['prediction'], confidence=response['confidence'])

# Define a route for the predict_api endpoint
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get the text input from the request body
    input_text = request.json['text']

    # Tokenize and pad the input text
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=100)

    # Make a prediction using the loaded model
    prediction = model.predict(input_seq)[0][0]

    # Convert the prediction to a label
    label = 'positive' if prediction >= 0.5 else 'negative'

    # Return the prediction as a JSON response
    response = {
        'prediction': label,
        'confidence': float(prediction)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
