import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model_path = "glmnet_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Define the feature names in the specified order
feature_names = [
    'ACEI/ARB',
    'aspirin',
    'reperfusion therapy',
    'Neu',
    'Hb',
    'Scr',
    'P'
]

# Create the title for the web app
st.title('Machine learning-based models to predict one-year mortality in patients with acute myocardial infarction combined with atrial fibrillation.')

# Introduction section
st.markdown("""
## Introduction
This web-based calculator was developed based on the Glmnet model with an AUC of 0.87 and a Brier score of 0.178. Users can obtain the 1-year risk of death for a given case by simply selecting the parameters and clicking on the 'Predict' button.
""")

# Create the input form
with st.form("prediction_form"):
    acei_arb = st.selectbox('ACEI/ARB', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='ACEI/ARB')
    aspirin = st.selectbox('Aspirin', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='aspirin')
    reperfusion_therapy = st.selectbox('Reperfusion Therapy', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='reperfusion therapy')
    Neu = st.slider('Neu (10^9/L)', min_value=0, max_value=30, value=5, step=1, key='Neu')
    Hb = st.slider('Hb (g/L)', min_value=0, max_value=300, value=150, step=1, key='Hb')
    Scr = st.slider('Scr (μmol/L)', min_value=0, max_value=1300, value=100, step=10, key='Scr')
    P = st.slider('P (mmHg)', min_value=20, max_value=200, value=110, step=1, key='P')

    predict_button = st.form_submit_button("Predict")

# Process form submission
if predict_button:
    data = {
        "ACEI/ARB": acei_arb,
        "aspirin": aspirin,
        "reperfusion therapy": reperfusion_therapy,
        "Neu": Neu,
        "Hb": Hb,
        "Scr": Scr,
        "P": P
    }

    # Convert input data to DataFrame using the exact feature names
    data_df = pd.DataFrame([data], columns=feature_names)

    # Scale the data using the loaded scaler
    data_scaled = scaler.transform(data_df)

    # Make a prediction
    prediction = model.predict_proba(data_scaled)[:, 1][0]  # Getting the probability of class 1
    st.write(f'Prediction: {prediction * 100:.2f}%')  # Convert probability to percentage
    
