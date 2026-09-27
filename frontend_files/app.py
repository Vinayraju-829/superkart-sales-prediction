
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒")
st.title("SuperKart Sales Prediction")
st.write("Enter the product and store details to predict sales revenue.")

API_URL = "http://backend:5000/predict"

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    product_weight = st.number_input("Product Weight", min_value=0.0)
    sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Product Allocated Area", min_value=0.0)
    product_mrp = st.number_input("Product MRP", min_value=0.0)
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    product_id_char = st.selectbox("Product ID Category", ["FD", "NC", "DR"])
    store_age = st.number_input("Store Age (Years)", min_value=0, step=1)
    product_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales"):
        input_data = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_category
        }

        try:
            response = requests.post(API_URL, json=input_data)
            response.raise_for_status()
            prediction = response.json()["predictions"][0]
            st.success(f"Predicted Sales Revenue: {prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data")
        st.dataframe(batch_data)

        if st.button("Predict Batch Sales"):
            try:
                response = requests.post(API_URL, json=batch_data.to_dict(orient="records"))
                response.raise_for_status()
                predictions = response.json()["predictions"]
                batch_data["Predicted_Sales"] = predictions
                st.success("Predictions completed successfully.")
                st.dataframe(batch_data)
                st.download_button("Download Predictions", batch_data.to_csv(index=False), "superkart_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
