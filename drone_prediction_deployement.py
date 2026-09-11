# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("drone_type_model.pkl")
label_encoder = joblib.load("drone_type_label_encoder.pkl")

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Drone Type Prediction",
    page_icon="🚁",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main Background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0b1020 0%,
            #111827 50%,
            #0b1220 100%
        );
    }

    /* Main Container */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 25px 20px 35px 20px;
    }

    .hero-icon {
        font-size: 55px;
        margin-bottom: 5px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(
            90deg,
            #60a5fa,
            #a78bfa,
            #38bdf8
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #aeb8c8;
        font-size: 17px;
    }

    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .card-title {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .card-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Labels */
    label {
        color: #dbeafe !important;
        font-weight: 600 !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
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
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.35);
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.18),
            rgba(124, 58, 237, 0.18)
        );
        border: 1px solid rgba(96,165,250,0.3);
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
    }

    .prediction-label {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .prediction-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Info cards */
    .info-card {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        border: 1px solid rgba(148,163,184,0.12);
    }

    .info-number {
        font-size: 25px;
        font-weight: 800;
        color: #60a5fa;
    }

    .info-text {
        color: #94a3b8;
        font-size: 13px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 40px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Hero Section
# --------------------------------------------------

st.markdown("""
<div class="hero">

    <div class="hero-icon">🚁</div>

    <div class="hero-title">
        Drone Type Prediction
    </div>

    <div class="hero-subtitle">
        Machine Learning powered classification system
    </div>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Model Information
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">89.51%</div>
        <div class="info-text">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">Random Forest</div>
        <div class="info-text">ML Algorithm</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">2 Types</div>
        <div class="info-text">Drone Classes</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.markdown("""
<div class="card">

<div class="card-title">
🚁 Enter Drone Details
</div>

<div class="card-subtitle">
Provide the specifications below to predict the drone category.
</div>

</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    Name = st.text_input(
        "Drone Name",
        "Super Drone"
    )

    Control_Range = st.number_input(
        "Control Range",
        min_value=0,
        value=100
    )

    Weight = st.number_input(
        "Weight",
        min_value=0,
        value=250
    )

    Price = st.number_input(
        "Selling Price",
        min_value=0.0,
        value=2705.0
    )

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
        value=8999.0
    )

    Discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

st.write("")

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("🚀 Predict Drone Type"):

    input_data = pd.DataFrame({
        "Control Range": [Control_Range],
        "Battery Type": [Battery_Type],
        "Weight": [Weight],
        "Price": [Price],
        "Actual Price": [Actual_Price],
        "Discount (%)": [Discount]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=["Battery Type"],
        drop_first=True
    )

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    predicted_type = label_encoder.inverse_transform(
        [prediction]
    )[0]

    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.markdown(f"""
    <div class="prediction-card">

        <div class="prediction-label">
            Prediction Result
        </div>

        <div class="prediction-value">
            🚁 {predicted_type}
        </div>

        <p style="color:#94a3b8; margin-top:10px;">
            {Name} is predicted to be a {predicted_type}.
        </p>

    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Built with Python • Scikit-learn • Random Forest • Streamlit
</div>
""", unsafe_allow_html=True)
