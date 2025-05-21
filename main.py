from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ---------------- Load Models ---------------- #
diabetes_model = pickle.load(open('Models/diabetes.pkl', 'rb'))
heart_model = pickle.load(open('Models/heart_disease_model.pkl', 'rb'))
cancer_model = pickle.load(open('Models/BreastCancer.pkl', 'rb'))
kidney_model=pickle.load(open('Models/kidney.pkl' , 'rb'))

# ---------------- Web Routes ---------------- #

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    if request.method == 'POST':
        try:
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            risk_score = 0
            if Pregnancies > 4: risk_score += 1
            if Glucose >= 140: risk_score += 2
            if BloodPressure >= 90: risk_score += 1
            if SkinThickness > 35: risk_score += 1
            if Insulin > 200: risk_score += 1
            if BMI >= 30: risk_score += 2
            if DiabetesPedigreeFunction >= 0.6: risk_score += 1
            if Age >= 45: risk_score += 1

            if risk_score >= 5:
                result = "High chance of diabetes"
            elif risk_score <= 2:
                result = "Low chance of diabetes"
            else:
                result = "Moderate risk — consider further testing"

        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('diabetes.html', result=result)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    result = None
    if request.method == 'POST':
        try:
            features = [float(request.form[key]) for key in [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]]

            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features

            low_risk_conditions = [
                (age < 45),
                (sex == 0),
                (cp == 0),
                (trestbps <= 120),
                (chol < 200),
                (fbs == 0),
                (restecg == 0),
                (thalach >= 150),
                (exang == 0),
                (oldpeak < 1.0),
                (slope == 2),
                (ca == 0),
                (thal == 3)
            ]

            high_risk_conditions = [
                (age >= 50),
                (sex == 1),
                (cp in [2, 3]),
                (trestbps > 140),
                (chol >= 240),
                (fbs == 1),
                (restecg in [1, 2]),
                (thalach < 100),
                (exang == 1),
                (oldpeak > 2.0),
                (slope in [0, 1]),
                (ca >= 1),
                (thal in [1, 2])
            ]

            low_risk_count = sum(low_risk_conditions)
            high_risk_count = sum(high_risk_conditions)

            if high_risk_count >= 5:
                result = "High chance of heart disease"
            elif low_risk_count >= 8:
                result = "Low chance of heart disease"
            else:
                result = "High chance of heart disease" if high_risk_count > low_risk_count else "Low chance of heart disease"

        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('heart.html', result=result)

@app.route('/cancer', methods=['GET', 'POST'])
def cancer():
    result = None
    if request.method == 'POST':
        try:
            features = [float(request.form[f"f{i}"]) for i in range(10)]

            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, \
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dim_mean = features

            high_risk_conditions = [
                radius_mean > 17,
                texture_mean > 20,
                perimeter_mean > 110,
                area_mean > 900,
                smoothness_mean > 0.10,
                compactness_mean > 0.15,
                concavity_mean > 0.12,
                concave_points_mean > 0.10,
                symmetry_mean > 0.20,
                fractal_dim_mean > 0.07
            ]

            low_risk_conditions = [
                radius_mean < 14,
                texture_mean < 17,
                perimeter_mean < 85,
                area_mean < 600,
                smoothness_mean < 0.09,
                compactness_mean < 0.12,
                concavity_mean < 0.09,
                concave_points_mean < 0.08,
                symmetry_mean < 0.18,
                fractal_dim_mean < 0.06
            ]

            high_risk_count = sum(high_risk_conditions)
            low_risk_count = sum(low_risk_conditions)

            if high_risk_count >= 5:
                result = "High risk of breast cancer (Malignant)"
            elif low_risk_count >= 7:
                result = "Low risk of breast cancer (Likely Benign)"
            else:
                result = "Moderate risk – further diagnosis recommended"

        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('cancer.html', result=result)

@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    result = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            bp = float(request.form['bp'])
            sg = float(request.form['sg'])
            al = float(request.form['al'])
            hemo = float(request.form['hemo'])
            pcv = float(request.form['pcv'])
            sc = float(request.form['sc'])
            sod = float(request.form['sod'])
            pot = float(request.form['pot'])

            risk_score = 0
            if bp >= 90: risk_score += 1
            if sg < 1.015: risk_score += 1
            if al >= 2: risk_score += 2
            if hemo < 11: risk_score += 2
            if pcv < 35: risk_score += 1
            if sc > 1.5: risk_score += 2
            if sod < 135: risk_score += 1
            if pot > 5.5: risk_score += 1
            if age > 60: risk_score += 1

            if risk_score >= 6:
                result = "High risk of chronic kidney disease"
            elif risk_score <= 2:
                result = "Low risk of chronic kidney disease"
            else:
                result = "Moderate risk — consider a medical check-up"

        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('kidney.html', result=result)

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

@app.route('/predict_kidney', methods=['POST'])
def predict_kidney_api():
    try:
        data = request.get_json(force=True)
        features = [
            float(data['age']), float(data['bp']), float(data['sg']), float(data['al']),
            float(data['hemo']), float(data['pcv']), float(data['sc']),
            float(data['sod']), float(data['pot'])
        ]
        prediction = kidney_model.predict([np.array(features)])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': f'Invalid input or internal error: {str(e)}'}), 400



# ---------------- Run Server ---------------- #
if __name__ == '__main__':
    app.run(debug=True)
