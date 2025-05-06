import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

 
# ------------------------------------------
# 🚜 1. Load or Train the Model
# ------------------------------------------
MODEL_PATH = "crop_recommendation_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.write("🔄 Training model, please wait...")

    # Load dataset
    df = pd.read_csv("Crop_recommendation.csv")
    X = df.drop("label", axis=1)
    y = df["label"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate (optional)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Model accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model and encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

else:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

# ------------------------------------------
# 🌿 2. Streamlit UI
# ------------------------------------------
st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Crop Recommendation System")
st.markdown("Enter the soil and weather conditions below:")

# Inputs
N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=42)
K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=43)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=80.0)
ph = st.number_input("Soil pH", min_value=3.5, max_value=9.5, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=200.0)

if st.button("Recommend Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    pred_encoded = model.predict(input_data)[0]
    recommended_crop = le.inverse_transform([pred_encoded])[0]
    st.success(f"🌱 Recommended Crop: **{recommended_crop}**")
