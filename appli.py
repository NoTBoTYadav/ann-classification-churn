import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

model = tf.keras.models.load_model('ann_model.h5')
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    ohe = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    le = pickle.load(f)

st.title('Customer Churn Prediction')
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age = st.number_input('Age', min_value=18, max_value=92)
balance = st.number_input('Balance')
products_number = st.number_input('Number of Products', min_value=1, max_value=4)
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.number_input('Credit Score')
tenure = st.number_input('Tenure', min_value=0, max_value=10)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Make sure your columns match what the scaler saw
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [products_number],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode Gender
input_data['Gender'] = le.transform(input_data['Gender'])

# One-hot encode Geography
geoencode = ohe.transform([[geography]]).toarray()
geoencodedf = pd.DataFrame(geoencode, columns=ohe.get_feature_names_out(['Geography']))

# Combine
inputdf = pd.concat([input_data.drop(columns=['Geography'], errors='ignore'), geoencodedf], axis=1)

# Ensure columns are in order scaler expects
expected_columns = scaler.feature_names_in_
inputdf = inputdf[expected_columns]

# Scale
inputscaled = scaler.transform(inputdf)

# Predict
prediction = model.predict(inputscaled)
prediction_proba = prediction[0][0]

st.write(f'Prediction Probability: {prediction_proba:.2f}')
if prediction_proba > 0.5:
    st.write('The customer is **likely to churn**.')
else:
    st.write('The customer is **unlikely to churn**.')
