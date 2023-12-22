import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# Load df
@st.cache
def fetch_data():
    df = pd.read_csv('jabodetabek_house_price.csv')
    return df

#load model
loaded_model = joblib.load('model_final_project')

# Insert a picture using st.image()
st.image("mo-house-minimalist-tiny-house-by-dform-01-1.jpg", use_column_width=True)

st.title("Aplikasi Prediksi Harga Rumah")
st.write("ISI DATA INFORMASI RUMAH NASABAH :")

#define df
df = fetch_data()

#input fitur
city = st.selectbox("City", df['city'].unique())
district = st.selectbox("District", df['district'].unique())
floors = st.number_input("Floors", min_value=0, step=1)
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0, step=1)
carports = st.number_input("Carports", min_value=0, step=1)
garages = st.number_input("Garages", min_value=0, step=1)
land_size_m2 = st.number_input("Land Size (m2)", min_value=0, step=1)
building_size_m2 = st.number_input("Building Size (m2)", min_value=0, step=1)
year_built = st.number_input("Year Built", min_value=0, step=1)
electricity = st.number_input("Electricity", min_value=0, step=1)
facilities = st.selectbox("Facilities", df['facilities'].unique())
maid_bedrooms = st.number_input("Maid Bedrooms", min_value=0, step=1)
maid_bathrooms = st.number_input("Maid Bathrooms", min_value=0, step=1)

#predict button
if st.button("Prediksi Harga Rumah"):
    input_data = pd.DataFrame({
    'city': [city],
    'district': [district],
    'floors': [floors],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'garages': [garages],
    'carports': [carports],
    'land_size_m2': [land_size_m2],
    'building_size_m2': [building_size_m2],
    'year_built': [year_built],
    'electricity': [electricity],
    'facilities': [facilities],
    'maid_bedrooms': [maid_bedrooms],
    'maid_bathrooms': [maid_bathrooms]
})
    prediction = loaded_model.predict(input_data)
    st.write("Prediksi Harga Rumah:", prediction)