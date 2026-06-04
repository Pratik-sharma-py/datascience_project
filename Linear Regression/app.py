import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('insurance_model.pkl')

st.title("Medical Insurance Charge Predictor")
st.write("Enter the details below to estimate annual charges.")

# --- user inputs (the 6 raw fields) ---
age      = st.slider("Age", 18, 64, 30)
sex      = st.selectbox("Sex", ["male", "female"])
bmi      = st.number_input("BMI", 15.0, 55.0, 25.0)
children = st.slider("Number of children", 0, 5, 0)
smoker   = st.selectbox("Smoker", ["yes", "no"])
region   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict"):
    # build a one-row dataframe with the SAME columns the pipeline was fit on
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'bmi': bmi, 'children': children,
        'smoker': smoker, 'region': region
    }])

    # recreate the engineered feature — must match training exactly
    input_df['is_smoker_and_obese'] = (
        (input_df['bmi'] >= 30) & (input_df['smoker'] == 'yes')
    ).astype(int)

    # predict in log space, then invert with expm1 back to dollars
    pred_log = model.predict(input_df)
    pred = np.expm1(pred_log)[0]

    st.success(f"Estimated annual charge: ${pred:,.2f}")