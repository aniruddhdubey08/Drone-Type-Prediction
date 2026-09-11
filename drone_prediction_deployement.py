# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("drone_type_model.pkl")
label_encoder = joblib.load("drone_type_label_encoder.pkl")


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Drone Type Prediction",
    page_icon="🚁",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #080d1c 0%,
        #101a33 50%,
        #080d1c 100%
    );
}

.block-container {
    max-width: 1150px;
    padding-top: 25px;
    padding-bottom: 50px;
}


/* ==========================
   HERO
   ========================== */

.hero {
    text-align: center;
    padding: 25px 20px 35px 20px;
}

.hero-icon {
    font-size: 58px;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 45px;
    font-weight: 800;
    color: #60a5fa;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #9caec5;
    font-size: 17px;
}


/* ==========================
   STAT CARDS
   ========================== */

.stat-card {
    background: rgba(20, 30, 50, 0.90);
    border: 1px solid rgba(96, 165, 250, 0.18);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    min-height: 105px;
}

.stat-number {
    color: #60a5fa;
    font-size: 27px;
    font-weight: 800;
}

.stat-label {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 5px;
}


/* ==========================
   SECTION
   ========================== */

.section-title {
    color: #f8fafc;
    font-size: 26px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 5px;
}

.section-description {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 22px;
}


/* ==========================
   INPUT LABELS
   ========================== */

label {
    color: #dbeafe !important;
    font-weight: 600 !important;
}


/* ==========================
   BUTTON
   ========================== */

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    font-size: 17px;
    font-weight: 700;
    margin-top: 15px;
}

.stButton > button:hover {
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.40);
}


/* ==========================
   PREDICTION CARD
   ========================== */

.prediction-card {
    background: linear-gradient(
        135deg,
        #172554,
        #312e81
    );

    border: 1px solid rgba(96, 165, 250, 0.35);
    border-radius: 18px;
    padding: 32px;
    margin-top: 30px;
    text-align: center;
}

.prediction-label {
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.prediction-value {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.prediction-name {
    color: #cbd5e1;
    font-size: 15px;
}


/* ==========================
   FOOTER
   ========================== */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid rgba(148, 163, 184, 0.10);
}

</style>
""",
    unsafe_allow_html=True
)


# ==================================================
# HERO SECTION
# ==================================================

st.markdown(
    """
<div class="hero">
<div class="hero-icon">🚁</div>
<div class="hero-title">Drone Type Prediction</div>
<div class="hero-subtitle">
Machine Learning powered drone classification system
</div>
</div>
""",
    unsafe_allow_html=True
)


# ==================================================
# MODEL INFORMATION
# ==================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
<div class="stat-card">
<div class="stat-number">89.51%</div>
<div class="stat-label">Model Accuracy</div>
</div>
""",
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
<div class="stat-card">
<div class="stat-number">Random Forest</div>
<div class="stat-label">Machine Learning Algorithm</div>
</div>
""",
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
<div class="stat-card">
<div class="stat-number">2 Types</div>
<div class="stat-label">Drone Classes</div>
</div>
""",
        unsafe_allow_html=True
    )


# ==================================================
# INPUT SECTION
# ==================================================

st.markdown(
    """
<div class="section-title">
🚁 Enter Drone Details
</div>

<div class="section-description">
Enter the specifications below to predict the drone category.
</div>
""",
    unsafe_allow_html=True
)


# ==================================================
# INPUT FIELDS
# ==================================================

col1, col2 = st.columns(2)


# --------------------------------------------------
# LEFT COLUMN
# --------------------------------------------------

with col1:

    Name = st.text_input(
        "Drone Name",
        value="Super Drone"
    )

    Control_Range = st.number_input(
        "Control Range",
        min_value=0,
        value=100,
        step=1
    )

    Weight = st.number_input(
        "Weight",
        min_value=0,
        value=250,
        step=1
    )

    Price = st.number_input(
        "Selling Price",
        min_value=0.0,
        value=2705.0,
        step=100.0
    )


# --------------------------------------------------
# RIGHT COLUMN
# --------------------------------------------------

with col2:

    Battery_Type = st.selectbox(
        "Battery Type",
        [
            "Lithium Battery",
            "AA Battery",
            "AA Rechargeable Battery",
            "AA Alkaline Battery",
            "AAA Battery",
            "AAA Rechargeable Battery",
            "AAA Alkaline Battery"
        ]
    )

    Actual_Price = st.number_input(
        "Actual Price",
        min_value=0.0,
        value=8999.0,
        step=100.0
    )

    Discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )


# ==================================================
# PREDICTION BUTTON
# ==================================================

if st.button("🚀 Predict Drone Type"):

    # --------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------

    input_data = pd.DataFrame(
        {
            "Control Range": [Control_Range],
            "Battery Type": [Battery_Type],
            "Weight": [Weight],
            "Price": [Price],
            "Actual Price": [Actual_Price],
            "Discount (%)": [Discount]
        }
    )


    # --------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------

    input_data = pd.get_dummies(
        input_data,
        columns=["Battery Type"],
        drop_first=True
    )


    # --------------------------------------------------
    # Match Model Features
    # --------------------------------------------------

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------
    # Convert Prediction to Original Label
    # --------------------------------------------------

    predicted_type = label_encoder.inverse_transform(
        [prediction]
    )[0]


    # ==================================================
    # DISPLAY RESULT
    # ==================================================

    result_html = (
        '<div class="prediction-card">'
        '<div class="prediction-label">PREDICTION RESULT</div>'
        '<div class="prediction-value">🚁 '
        + str(predicted_type)
        + '</div>'
        '<div class="prediction-name">'
        + str(Name)
        + ' is predicted as a '
        + str(predicted_type)
        + '.'
        + '</div>'
        '</div>'
    )

    st.markdown(
        result_html,
        unsafe_allow_html=True
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
<div class="footer">
Built with Python • Pandas • Scikit-learn • Random Forest • Streamlit
</div>
""",
    unsafe_allow_html=True
)
