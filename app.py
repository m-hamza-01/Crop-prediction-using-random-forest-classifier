from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved pipeline model
pipeline = joblib.load("C:/Users/Lenovo/Documents/6th semester/problem solving lab/mid/mid/classification_pipeline.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Make prediction using the pipeline
    prediction = pipeline.predict(input_data)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
