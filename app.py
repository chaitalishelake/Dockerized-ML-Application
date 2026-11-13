
# MACHINE LEARNING APPLICATION USING DOCKER
# Author: Chaitali Shelake

from flask import Flask, request, render_template_string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# 1. Create sample dataset
data = {
    "Study_Hours": [2, 4, 5, 6, 8, 9, 3, 7],
    "Attendance": [50, 60, 70, 80, 90, 95, 55, 85],
    "Result": [0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# 2. Separate features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 6. Application interface
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Result Prediction</title>
    <style>
        body {
            font-family: Arial;
            background: #eef2f7;
            text-align: center;
            padding: 40px;
        }
        .container {
            background: white;
            max-width: 420px;
            margin: auto;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px #ccd2da;
        }
        input, button {
            width: 90%;
            padding: 12px;
            margin: 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        button {
            background: #1769aa;
            color: white;
            cursor: pointer;
        }
        .result {
            font-size: 20px;
            font-weight: bold;
            color: #1769aa;
        }
    </style>
</head>
<body>
<div class="container">
    <h2>Student Result Prediction</h2>
    <p>Enter student performance details</p>

    <form method="POST">
        <input type="number" name="hours"
        placeholder="Study Hours" min="0" max="24" required>

        <input type="number" name="attendance"
        placeholder="Attendance (%)" min="0" max="100" required>

        <button type="submit">Predict Result</button>
    </form>

    {% if result is not none %}
        <p class="result">{{ result }}</p>
    {% endif %}

    <p>Model Accuracy: {{ accuracy }}%</p>
    <p>Developed by Chaitali Shelake</p>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])

        prediction = model.predict(
            pd.DataFrame(
                [[hours, attendance]],
                columns=["Study_Hours", "Attendance"]
            )
        )[0]

        if prediction == 1:
            result = "Predicted Result: PASS"
        else:
            result = "Predicted Result: NEEDS IMPROVEMENT"

    return render_template_string(
        HTML,
        result=result,
        accuracy=round(accuracy * 100, 2)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
