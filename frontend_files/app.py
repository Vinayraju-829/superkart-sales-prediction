import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="SuperKart Sales Prediction", layout="wide")

st.title("SuperKart Sales Prediction")
st.write("Predict product-store sales using the trained machine learning model.")

API_ROOT = "http://host.docker.internal:7860"
PREDICT_URL = API_ROOT + "/v1/predict"
BATCH_PREDICT_URL = API_ROOT + "/v1/predictbatch"

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Single Sales Prediction")

    product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027, format="%.3f")
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Food Mart", "Departmental Store"])
    product_id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
    store_age_years = st.number_input("Store Age (Years)", min_value=0, value=16)
    product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age_years,
            "Product_Type_Category": product_type_category
        }

        try:
            response = requests.post(PREDICT_URL, json=payload)

            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Predicted Sales: {prediction:,.2f}")
            else:
                st.error(f"Prediction failed: {response.text}")

        except Exception as e:
            st.error(f"Unable to connect to prediction API: {e}")

with tab2:
    st.subheader("Batch Sales Prediction")

    uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data")
        st.dataframe(batch_data)

        if st.button("Predict Batch Sales"):
            try:
                files = {
                    "file": (
                        "batch_data.csv",
                        batch_data.to_csv(index=False).encode("utf-8"),
                        "text/csv"
                    )
                }

                response = requests.post(BATCH_PREDICT_URL, files=files)

                if response.status_code == 200:
                    predictions = response.json()
                    result = batch_data.copy()
                    result["Predicted_Sales"] = [predictions[str(i)] for i in range(len(result))]

                    st.success("Batch prediction completed successfully.")
                    st.dataframe(result)

                    st.download_button(
                        "Download Predictions",
                        result.to_csv(index=False).encode("utf-8"),
                        "SuperKart_Predictions.csv",
                        "text/csv"
                    )
                else:
                    st.error(f"Batch prediction failed: {response.text}")

            except Exception as e:
                st.error(f"Unable to connect to prediction API: {e}")