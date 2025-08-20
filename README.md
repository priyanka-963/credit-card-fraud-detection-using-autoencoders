# credit-card-fraud-detection-using-autoencoders
We built a Credit Card Fraud Detection System using an autoencoder trained on normal transactions. It detects anomalies via reconstruction error with a 95th percentile threshold. A Flask app allows CSV uploads for predictions, and an email alert system notifies users of suspicious transactions.
📌 Project Description

This project implements an end-to-end Credit Card Fraud Detection System using an autoencoder-based anomaly detection model. The goal is to identify fraudulent transactions by learning the normal transaction patterns and flagging deviations.

🔧 What We Did

Data Preprocessing

We loaded a credit card transaction dataset and separated the target column (“Class”) from the input features.

All numerical values were scaled using RobustScaler, which helps in reducing the effect of outliers.

We ensured the same feature names and order were preserved for consistent training and inference.

Model Development – Autoencoder

We built a deep autoencoder using TensorFlow/Keras.

The autoencoder compresses transactions into a lower-dimensional representation and reconstructs them back.

The reconstruction error (MSE) is minimal for normal transactions, but significantly higher for fraud cases.

A threshold was set at the 95th percentile of reconstruction errors, so transactions above this threshold are marked as fraud.

Training and Evaluation

The autoencoder was trained only on normal transactions to capture the usual spending patterns.

During testing, fraudulent transactions produced high errors and were flagged correctly.

The trained model was saved for deployment.

Flask Web Application

A Flask web app was developed to make the system interactive.

Users can upload a CSV file of transactions.

The system preprocesses the uploaded data, applies the trained autoencoder model, and predicts fraud.

Results are displayed in a table format with reconstruction error values and fraud labels.

Email Alert System

We added an email notification system using Python’s smtplib.

If fraudulent transactions are detected, an automated alert email is sent to the configured recipient.

🚀 Outcome

A fully functional fraud detection pipeline.

Web-based interface for easy usage.

Automated fraud alert system for immediate notifications.

This project demonstrates the practical use of deep learning + web deployment to tackle real-world fraud detection challenges.
