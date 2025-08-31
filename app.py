from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved model and pipeline
model = pickle.load(open("fitness_model.pkl", "rb"))
pipeline = pickle.load(open("pipeline.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    tip = None

    if request.method == "POST":
        try:
            # Collect user input
            age = int(request.form["age"])
            weight = float(request.form["weight"])
            height = float(request.form["height"])
            heart_rate = float(request.form["heart_rate"])
            blood_pressure = request.form["blood_pressure"]
            sleep = float(request.form["sleep"])
            nutrition_quality = float(request.form["nutrition_quality"])
            activity_index = float(request.form["activity_index"])
            smokes = int(request.form["smokes"])
            gender = int(request.form["gender"])

            #  Input validation
            if age <= 0 or age > 100:
                result = "Error: Age must be between 1–100."
            elif weight <= 20 or weight > 200:
                result = "Error: Weight must be realistic (20–200 kg)."
            elif sleep < 0 or sleep > 24:
                result = "Error: Sleep hours must be between 0–24."
            else:
                # User data dictionary
                user_dict = {
                    "age": [age],
                    "weight_kg": [weight],
                    "height_cm": [height],
                    "heart_rate": [heart_rate],
                    "blood_pressure": [blood_pressure],
                    "sleep_hours": [sleep],
                    "nutrition_quality": [nutrition_quality],
                    "activity_index": [activity_index],
                    "smokes": [smokes],
                    "gender": [gender]
                }

                # Transform and predict
                user_prepared = pipeline.transform(pd.DataFrame(user_dict))
                prediction = model.predict(user_prepared)[0]
                prob = model.predict_proba(user_prepared)[0]

                # Prediction result
                result = "Fit ✅" if prediction == 1 else "Not Fit ❌"
                probability = round(max(prob) * 100, 2)

                # Health tips
                if prediction == 1:
                    tip = "Great job! Keep maintaining a balanced diet and regular exercise."
                else:
                    tip = "Try increasing your activity, improving sleep quality, and eating healthier foods."

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, probability=probability, tip=tip)

if __name__ == "__main__":
    app.run(debug=True)
