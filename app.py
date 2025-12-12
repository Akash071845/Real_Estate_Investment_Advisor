# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import mlflow
import mlflow.pyfunc
import pandas as pd
import json
from mlflow.tracking import MlflowClient

# =====================================================
# LOAD SOURCE DATA
# =====================================================
df = pd.read_csv("classification_data.csv")

# =====================================================
# LOAD MLflow MODELS (CLASSIFIER + REGRESSOR)
# =====================================================
@st.cache_resource
def load_models():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    clf = mlflow.pyfunc.load_model("models:/Random_Forest_Classifier@challenger")
    reg = mlflow.pyfunc.load_model("models:/Random_Forest_Regressor@challenger")
    return clf, reg

clf_model, reg_model = load_models()

# =====================================================
# SIDEBAR INPUT
# =====================================================
st.sidebar.header("🏠 Property Input Form")

state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
city = st.sidebar.selectbox("City", sorted(df["City"].unique()))
locality = st.sidebar.selectbox("Locality", sorted(df["Locality"].unique()))
property_type = st.sidebar.selectbox("Property Type", sorted(df["Property_Type"].unique()))
furnished = st.sidebar.selectbox("Furnished Status", sorted(df["Furnished_Status"].unique()))
transport_access = st.sidebar.selectbox("Public Transport Accessibility", sorted(df["Public_Transport_Accessibility"].unique()))
parking = st.sidebar.selectbox("Parking Space", sorted(df["Parking_Space"].unique()))
security = st.sidebar.selectbox("Security", sorted(df["Security"].unique()))
facing = st.sidebar.selectbox("Facing", sorted(df["Facing"].unique()))
owner_type = st.sidebar.selectbox("Owner Type", sorted(df["Owner_Type"].unique()))
availability = st.sidebar.selectbox("Availability Status", sorted(df["Availability_Status"].unique()))
amenities = st.sidebar.multiselect("Amenities", sorted(df["Amenities"].unique()))

amenities_str = ", ".join(amenities)
amenities_count = len(amenities)

bhk = st.sidebar.number_input("BHK", 1, 10, 3)
size_sqft = st.sidebar.number_input("Size in SqFt", 200, 10000, 1500)
price_lakhs = st.sidebar.number_input("Current Price (Lakhs)", 1.0, 5000.0, 489.76)
price_per_sqft = st.sidebar.number_input("Price Per SqFt", 0.01, 1.0, 0.08)

year_built = st.sidebar.number_input("Year Built", 1950, 2025, 2015)
floor_no = st.sidebar.number_input("Floor No.", 0, 200, 10)
total_floors = st.sidebar.number_input("Total Floors", 1, 200, 20)

nearby_schools = st.sidebar.number_input("Nearby Schools", 0, 50, 5)
nearby_hospitals = st.sidebar.number_input("Nearby Hospitals", 0, 50, 3)

growth_rate = st.sidebar.number_input("Location Growth Rate", 0.00, 0.20, 0.09)
years = st.sidebar.number_input("Future Prediction Years", 1, 50, 5)

age_property = 2025 - year_built
rera = 1 if availability.lower() in ["ready_to_move", "under_construction"] else 0

investment_score = 0.6
multi_factor_score = (bhk >= 3) + (rera == 1) + (availability.lower() in ["ready_to_move", "under_construction"])

# =====================================================
# CREATE INPUT DF
# =====================================================
input_df = pd.DataFrame({
    "ID": [101],
    "State": [state],
    "City": [city],
    "Locality": [locality],
    "Property_Type": [property_type],
    "BHK": [bhk],
    "Size_in_SqFt": [size_sqft],
    "Price_in_Lakhs": [price_lakhs],
    "Price_per_SqFt": [price_per_sqft],
    "Year_Built": [year_built],
    "Furnished_Status": [furnished],
    "Floor_No": [floor_no],
    "Total_Floors": [total_floors],
    "Age_of_Property": [age_property],
    "Nearby_Schools": [nearby_schools],
    "Nearby_Hospitals": [nearby_hospitals],
    "Public_Transport_Accessibility": [transport_access],
    "Parking_Space": [parking],
    "Security": [security],
    "Amenities": [amenities_str],
    "Facing": [facing],
    "Owner_Type": [owner_type],
    "Availability_Status": [availability],
    "Amenities_Count": [amenities_count],
    "growth_rate_location": [growth_rate],
    "RERA": [rera],
    "Investment_Score": [investment_score],
    "multi_factor_score": [multi_factor_score],
})

input_df["future_price"] = input_df["Price_in_Lakhs"] * (1 + input_df["growth_rate_location"]) ** years

st.write("### 🔍 Input Data Preview")
st.dataframe(input_df)

# =====================================================
# CLEAN CATEGORICAL FIELDS
# =====================================================
for col in ["State","City","Locality","Property_Type","Furnished_Status",
            "Public_Transport_Accessibility","Parking_Space","Security",
            "Facing","Owner_Type","Availability_Status","Amenities"]:
    input_df[col] = input_df[col].fillna("Unknown").replace("", "Unknown")

# =====================================================
# FILTER PROPERTIES SECTION
# =====================================================
st.sidebar.header("🔎 Filter Properties")

df_properties = pd.read_csv("regression_data.csv")
df_properties["future_price"] = df_properties["Price_in_Lakhs"] * (
    1 + df_properties["growth_rate_location"]
) ** years

filter_cities = st.sidebar.multiselect(
    "Select Cities",
    df_properties["City"].unique(),
    default=df_properties["City"].unique()
)

price_range = st.sidebar.slider(
    "Price Range (Lakhs)",
    int(df_properties["Price_in_Lakhs"].min()),
    int(df_properties["Price_in_Lakhs"].max()),
    (int(df_properties["Price_in_Lakhs"].min()), int(df_properties["Price_in_Lakhs"].max()))
)

bhk_range = st.sidebar.slider(
    "BHK Range",
    int(df_properties["BHK"].min()),
    int(df_properties["BHK"].max()),
    (int(df_properties["BHK"].min()), int(df_properties["BHK"].max()))
)

size_range = st.sidebar.slider(
    "Size Range (SqFt)",
    int(df_properties["Size_in_SqFt"].min()),
    int(df_properties["Size_in_SqFt"].max()),
    (int(df_properties["Size_in_SqFt"].min()), int(df_properties["Size_in_SqFt"].max()))
)

filtered_df = df_properties[
    (df_properties["City"].isin(filter_cities)) &
    (df_properties["Price_in_Lakhs"].between(price_range[0], price_range[1])) &
    (df_properties["BHK"].between(bhk_range[0], bhk_range[1])) &
    (df_properties["Size_in_SqFt"].between(size_range[0], size_range[1]))
]

st.write("### 🏘️ Filtered Properties")
st.dataframe(filtered_df)

# =====================================================
# PREDICTION BUTTON
# =====================================================
st.sidebar.header("🚀 Run Predictions")
predict_button = st.sidebar.button("Predict")

if predict_button:

    # Convert datatypes
    float_cols = ["Price_in_Lakhs","growth_rate_location","Investment_Score","future_price"]
    for col in float_cols:
        input_df[col] = input_df[col].astype("float64")

    int_cols = ["ID","BHK","Size_in_SqFt","Year_Built","Floor_No","Total_Floors",
                "Age_of_Property","Nearby_Schools","Nearby_Hospitals",
                "Amenities_Count","RERA","multi_factor_score"]

    for col in int_cols:
        input_df[col] = input_df[col].astype("int64")

    # ===============================
    # CLASSIFICATION OUTPUT
    # ===============================
    clf_pred = clf_model.predict(input_df)[0]
    st.write("### 📊 Classification Output")
    st.write(f"Good Investment: **{'Yes' if clf_pred == 1 else 'No'}**")

    # ===============================
    # REGRESSION OUTPUT
    # ===============================
    reg_pred = reg_model.predict(input_df)[0]

    st.write("### 💰 Regression Output")
    st.write(f"Predicted Price After {years} Years: **{reg_pred:.2f} Lakhs**")
    st.write(f"Future Price (Formula-Based): **{input_df['future_price'][0]:.2f} Lakhs**")

    # =====================================================
    # LOAD FEATURE IMPORTANCES FROM MLFLOW
    # =====================================================
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    run_id = "92a47046fd3f428db542560d0cd2c2c0"   # your run ID
    artifact_path = "feature_importances.json"

    # Download artifact
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path
    )

    with open(local_path, "r") as f:
        feature_importances = json.load(f)

    st.write("### 📌 Loaded Feature Importances")
    st.json(feature_importances)

    # BAR CHART
    st.write("### 📈 Feature Importance Chart")
    st.bar_chart(feature_importances)

# =====================================================
# VISUAL ANALYTICS DASHBOARD
# =====================================================
st.header("📊 Visual Analytics")

analytics_df = pd.read_csv("cleaned_data.csv")

# Price Trends
st.subheader("📈 Price Trends Over Years")
if "Year_Built" in analytics_df.columns and "Price_in_Lakhs" in analytics_df.columns:
    st.line_chart(analytics_df.groupby("Year_Built")["Price_in_Lakhs"].mean())
else:
    st.warning("Missing year or price column.")

# Location Heatmap
st.subheader("🌍 Location Heatmap")
if "City" in analytics_df.columns:
    st.bar_chart(analytics_df.groupby("City")["Price_in_Lakhs"].mean())
else:
    st.warning("Missing location column.")

# BHK Distribution
st.subheader("🏘️ BHK Distribution")
if "BHK" in analytics_df.columns:
    st.bar_chart(analytics_df["BHK"].value_counts().sort_index())
else:
    st.warning("Missing bhk column.")

# Property Value Comparisons
st.subheader("💰 Price Comparison by BHK")
if "BHK" in analytics_df.columns and "Price_in_Lakhs" in analytics_df.columns:
    st.line_chart(analytics_df.groupby("BHK")["Price_in_Lakhs"].mean())
else:
    st.warning("Missing bhk or price columns.")
