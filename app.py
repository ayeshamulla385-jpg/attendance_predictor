from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    study = int(request.form['study'])
    sleep = int(request.form['sleep'])
    distance = int(request.form['distance'])
    attendance = int(request.form['attendance'])
    participation = int(request.form['participation'])
    health = int(request.form['health'])

    data = np.array([[study, sleep, distance,
                      attendance, participation, health]])

    prediction = model.predict(data)

    result = encoder.inverse_transform(prediction)

    recommendation = ""

    if result[0] == "Low":
        recommendation = "Attend classes regularly"

    elif result[0] == "Medium":
        recommendation = "Improve attendance slightly"

    else:
        recommendation = "Excellent attendance performance"

    return render_template(
        "result.html",
        prediction=result[0],
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)