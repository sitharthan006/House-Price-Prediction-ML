import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


st.markdown("""
<style>

/* Background color */
.stApp {
    background-color: #0E1117;
}

/* Main title */
h1 {
    color: #00FFD1;
    text-align: center;
}

/* Subheaders */
h2, h3 {
    color: #FFD700;
}

/* Normal text */
p, label {
    color: white;
    font-size: 16px;
}

/* Input box styling */
.stNumberInput input {
    background-color: #1E1E2F;
    color: white;
}

/* Buttons */
.stButton>button {
    background-color: #00C9A7;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}

/* Button hover */
.stButton>button:hover {
    background-color: #00FFD1;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("🏠 House Price Prediction App")
st.write("Enter house details to predict the sale price.")

# Load trained model
model = pickle.load(open(r"models/ui_models/ridge_ui_model.pkl", "rb"))

# User Inputs

st.subheader("Enter House Features")

overallqual = st.number_input(
    "Overall Quality (1-10)",
    min_value=1,
    max_value=10,
    step=1
)

grlivarea = st.number_input(
    "Ground Living Area (sq ft)",
    min_value=100
)

totalbsmtsf = st.number_input(
    "Total Basement Area (sq ft)",
    min_value=0
)

yearbuilt = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2025,
    step=1
)

totalbath = st.number_input(
    "Total Bathrooms",
    min_value=0.5,
    step=0.5
)

houseage = st.number_input(
    "House Age (years)",
    min_value=0
)

lotarea = st.number_input(
    "Lot Area (sq ft)",
    min_value=100
)

kitchenqual = st.selectbox(
    "Kitchen Quality",
    options=['TA', 'Fa', 'Gd', 'Ex']
)

neighborhood = st.selectbox(
    "Neighborhood",
    options=[
        'CollgCr','Veenker','Crawfor','NoRidge','Mitchel','Somerst',
        'NWAmes','OldTown','BrkSide','Sawyer','NridgHt','NAmes',
        'SawyerW','IDOTRR','MeadowV','Edwards','Timber','Gilbert',
        'StoneBr','ClearCr','NPkVill','Blmngtn','BrDale','SWISU',
        'Blueste'
    ]
)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "OverallQual":[overallqual],
        "GrLivArea":[grlivarea],
        "TotalBsmtSF":[totalbsmtsf],
        "YearBuilt":[yearbuilt],
        "TotalBath":[totalbath],
        "HouseAge":[houseage],
        "LotArea":[lotarea],
        "KitchenQual":[kitchenqual],
        "Neighborhood":[neighborhood]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"🏡 Predicted House Price: ${prediction:,.2f}")


