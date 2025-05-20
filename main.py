from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ---------------- Load Models ---------------- #
diabetes_model = pickle.load(open('Models/diabetes.pkl', 'rb'))
heart_model = pickle.load(open('Models/heart_disease_model.pkl', 'rb'))
cancer_model = pickle.load(open('Models/BreastCancer.pkl', 'rb'))

# ---------------- Web Routes ---------------- #

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    if request.method == 'POST':
        try:
            inputs = [float(x) for x in request.form.values()]
            prediction = diabetes_model.predict([inputs])[0]
            result = "High chance of diabetes" if prediction == 1 else "Low chance of diabetes"
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('diabetes.html', result=result)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    result = None
    if request.method == 'POST':
        try:
            # Extract all inputs
            features = [float(request.form[key]) for key in [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]]

            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features

            # Define low risk conditions
            # Define rule-based risk conditions
            low_risk_conditions = [
                (age <= 60),
                (chol <= 200),
                (trestbps <= 140),
                (thalach >= 120),
                (exang == 0),
                (oldpeak <= 2),
                (ca == 0),
                (thal != 3)
            ]

            high_risk_conditions = [
                (60 < age <= 80),
                (200 < chol <= 440),
                (trestbps > 140),
                (thalach < 120),
                (exang == 1),
                (oldpeak > 2),
                (ca >= 1),
                (thal == 3)
            ]

            # Count matched conditions
            low_risk_count = sum(low_risk_conditions)
            high_risk_count = sum(high_risk_conditions)

            # Determine result based on rule logic
            if high_risk_count >= 3:
                result = "High chance of heart disease (based on rule-based analysis)"
            elif low_risk_count == len(low_risk_conditions):
                result = "Low chance of heart disease (based on rule-based analysis)"
            else:
                prediction = heart_model.predict([np.array(features)])[0]
                if prediction == 1:
                    result = "High chance of heart disease"
                else:
                    result = "Low chance of heart disease"

        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('heart.html', result=result)

@app.route('/cancer', methods=['GET', 'POST'])
def cancer():
    result = None
    if request.method == 'POST':
        try:
            inputs = [float(x) for x in request.form.values()]
            prediction = cancer_model.predict([inputs])[0]
            result = "Possibility of breast cancer" if prediction == 1 else "Likely benign (no cancer)"
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('cancer.html', result=result)

# ---------------- API Routes ---------------- #

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes_api():
    try:
        data = request.get_json()
        features = [
            data['Pregnancies'], data['Glucose'], data['BloodPressure'],
            data['SkinThickness'], data['Insulin'], data['BMI'],
            data['DiabetesPedigreeFunction'], data['Age']
        ]
        prediction = diabetes_model.predict([np.array(features)])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_heart', methods=['POST'])
def predict_heart_api():
    try:
        data = request.get_json()
        features = [data[key] for key in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        prediction = heart_model.predict([np.array(features)])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer_api():
    try:
        data = request.get_json()
        features = [
            data['mean_radius'], data['mean_texture'],
            data['mean_perimeter'], data['mean_area'],
            data['mean_smoothness']
        ]
        prediction = cancer_model.predict([np.array(features)])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ---------------- Run Server ---------------- #
if __name__ == '__main__':
    app.run(debug=True)
