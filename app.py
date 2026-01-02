from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load trained model
with open("model/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Read unseen data
        df = pd.read_excel(file_path)

        # SAME preprocessing (education mapping example)
        edu_map = {
            "SSC": 1, "12TH": 2, "GRADUATE": 3,
            "UNDER GRADUATE": 3, "POST-GRADUATE": 4,
            "OTHERS": 1, "PROFESSIONAL": 3
        }
        df["EDUCATION"] = df["EDUCATION"].map(edu_map)

        # One-hot encoding
        df_encoded = pd.get_dummies(
            df,
            columns=["MARITALSTATUS", "GENDER", "last_prod_enq2", "first_prod_enq2"]
        )

        # Align columns with training
        model_features = model.get_booster().feature_names
        df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)

        # Prediction
        preds = model.predict(df_encoded)

        risk_map = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
        df["Target"] = pd.Series(preds).map(risk_map)

        output_path = os.path.join(OUTPUT_FOLDER, "Final_Predictions.xlsx")
        df.to_excel(output_path, index=False)

        return send_file(output_path, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)



