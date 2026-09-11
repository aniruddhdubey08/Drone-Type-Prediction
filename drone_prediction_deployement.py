# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load Model and Label Encoder
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

st.markdown(
    """
    <style>

    /* ------------------------------
       Main App Background
    ------------------------------ */

    .stApp {
        background: linear-gradient(
            135deg,
            #080d1c 0%,
            #10182b 50%,
            #080d1c 100%
        );
    }

    .block-container {
        max-width: 1150px;
        padding-top: 35px;
        padding-bottom: 50px;
    }


    /* ------------------------------
       Hero Section
    ------------------------------ */

    .hero {
        text-align: center;
        padding: 35px 20px 40px 20px;
    }

    .hero-icon {
        font-size: 55px;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 10px;
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
        color: #9caec5;
        font-size: 17px;
    }


    /* ------------------------------
       Statistics Cards
    ------------------------------ */

    .stat-card {
        background: rgba(20, 30, 50, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        min-height: 105px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
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


    /* ------------------------------
       Section Header
    ------------------------------ */

    .section-header {
        margin-top: 35px;
        margin-bottom: 15px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 750;
    }

    .section-description {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 5px;
    }


    /* ------------------------------
       Input Area
    ------------------------------ */

    .input-box {
        background: rgba(20, 30, 50, 0.80);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 25px;
        margin-top: 10px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
    }


    /* ------------------------------
       Streamlit Labels
    ------------------------------ */

    label {
        color: #dbeafe !important;
        font-weight: 600 !important;
    }


    /* ------------------------------
       Predict Button
    ------------------------------ */

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
        margin-top: 10px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.40);
    }


    /* ------------------------------
       Prediction Result
    ------------------------------ */

    .prediction-card {
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.18),
            rgba(124, 58, 237, 0.18)
        );

        border: 1px solid rgba(96, 165, 250, 0.30);
        border-radius: 18px;
        padding: 30px;
        margin-top: 30px;
        text-align: center;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
    }

    .prediction-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .prediction-value {
        color: #ffffff;
        font-size: 34px;
        font-weight: 800;
        margin-top: 8px;
    }

    .prediction-name {
        color: #9caec5;
        font-size: 15px;
        margin-top: 10px;
    }


    /* ------------------------------
       Footer
    ------------------------------ */

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


# --------------------------------------------------
# Hero Section
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">
            🚁
        </div>

        <div class="hero-title">
            Drone Type Prediction
        </div>

        <div class="hero-subtitle">
            Machine Learning powered drone classification system
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Model Statistics
# --------------------------------------------------

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


# --------------------------------------------------
# Input Section Header
# --------------------------------------------------

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            🚁 Enter Drone Details
        </div>

        <div class="section-description">
            Enter the specifications below to predict the drone category.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input Box
# --------------------------------------------------

st.markdown(
    '<div class="input-box">',
    unsafe_allow_html=True
)


# --------------------------------------------------
# First Row
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    Name = st.text_input(
        "Drone Name",
        value="Super Drone"
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


# --------------------------------------------------
# Second Row
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    Control_Range = st.number_input(
        "Control Range",
        min_value=0,
        value=100,
        step=1
    )

with col2:

    Weight = st.number_input(
        "Weight",
        min_value=0,
        value=250,
        step=1
    )


# --------------------------------------------------
# Third Row
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    Price = st.number_input(
        "Selling Price",
        min_value=0.0,
        value=2705.0,
        step=100.0
    )

with col2:

    Actual_Price = st.number_input(
        "Actual Price",
        min_value=0.0,
        value=8999.0,
        step=100.0
    )


# --------------------------------------------------
# Fourth Row
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    Discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

with col2:

    st.write("")


# Close input box

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("🚀 Predict Drone Type"):

    # Create input dataframe

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


    # One-hot encode Battery Type

    input_data = pd.get_dummies(
        input_data,
        columns=["Battery Type"],
        drop_first=True
    )


    # Match training features

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )


    # Make prediction

    prediction = model.predict(input_data)[0]


    # Convert encoded prediction back to original label

    predicted_type = label_encoder.inverse_transform(
        [prediction]
    )[0]


    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Prediction Result
            </div>

            <div class="prediction-value">
                🚁 {predicted_type}
            </div>

            <div class="prediction-name">
                {Name} is predicted as a {predicted_type}.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Built with Python • Pandas • Scikit-learn • Random Forest • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
