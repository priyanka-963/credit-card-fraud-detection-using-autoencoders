import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow import keras
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

ARTIFACTS = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS, "best_autoencoder.keras")
SCALER_PATH = os.path.join(ARTIFACTS, "scaler_robust.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS, "features_list.joblib")

EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")  
EMAIL_RECEIVER = ""  

app = Flask(__name__)
app.secret_key = "fraud-detection-secret"

print("Loading model and scalers...")
model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

if os.path.exists(FEATURES_PATH):
    features = joblib.load(FEATURES_PATH)
    print(f"✅ Loaded {len(features)} features from features_list.joblib")
else:
    features = None
    print("⚠️ features_list.joblib not found. Will create on first run.")

def engineer_features_for_scoring(df):
    """Prepare dataframe for scoring (drop labels, select & scale features)."""
    global features

    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    df = df.select_dtypes(include=[np.number])

    if features is None:
        features = list(df.columns)
        joblib.dump(features, FEATURES_PATH)
        print(f"✅ Created features_list.joblib with {len(features)} features")

    df = df.reindex(columns=features, fill_value=0)

    X_scaled = pd.DataFrame(scaler.transform(df), columns=features)
    return X_scaled


def detect_fraud(df, threshold=0.95):
    """Run fraud detection using reconstruction error."""
    X_scaled = engineer_features_for_scoring(df)
    reconstructed = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

    fraud_threshold = np.quantile(mse, threshold)
    df["recon_error"] = mse
    df["fraud_flag"] = (df["recon_error"] > fraud_threshold).astype(int)

    return df, fraud_threshold


def send_email_alert(fraud_count):
    """Send email alert if fraud detected."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = "⚠️ Fraud Alert - Credit Card Transactions"
        body = MIMEText(f"Warning: {fraud_count} suspicious transactions detected.")
        msg.attach(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, [EMAIL_RECEIVER], msg.as_string())

        print("✅ Email alert sent successfully")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("home"))

    try:
        df = pd.read_csv(file)

      
        results, threshold = detect_fraud(df.copy())

        frauds = results[results["fraud_flag"] == 1]
        fraud_count = len(frauds)

        if fraud_count > 0:
            send_email_alert(fraud_count)

        return render_template(
            "results.html",
            tables=[results.to_html(classes="table table-bordered table-hover", index=False)],
            fraud_count=fraud_count,
            total=len(results),
            threshold=round(threshold, 5)
        )

    except Exception as e:
        flash(f"Error processing file: {e}")
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, port=4000)
