from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # frontend HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from frontend
        data = [float(x) for x in request.form.values()]
        features = np.array([data])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # Probability of class 1

        result = {
            'prediction': int(prediction),
            'probability': round(probability * 100, 2)
        }
        return render_template('index.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
