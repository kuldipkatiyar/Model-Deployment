import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model (replace 'heart_disease_model.pkl' with your model file)
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/') 
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Create input data for prediction
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

        # Make prediction using the trained model
        prediction = model.predict(input_data)[0]

        # Return the prediction result
        if prediction == 1:
            result = "Heart disease detected."
        else:
            result = "No heart disease detected."
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)