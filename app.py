import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from feature_extractor import extract_features

# Set page config
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Phishing Website Detection via URL")
st.markdown("Enter a website URL. The app will extract its features in real-time and use our Random Forest model to predict if it is a phishing attempt.")

@st.cache_resource(show_spinner=False)
def load_and_train_model():
    try:
        # Load the dataset
        df = pd.read_csv('df_clean.csv')
        X = df.drop("label", axis=1)
        y = df["label"]
        
        # Train test split and model training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save a sample row for default values for obscure features
        sample_data = X_train.iloc[0].to_dict()
        
        return model, X.columns.tolist(), sample_data, df
    except Exception as e:
        return None, None, None, str(e)

with st.spinner("Loading dataset and training model. This might take a moment..."):
    model, features, default_sample, df_full = load_and_train_model()

if model is None:
    st.error(f"Error loading model or dataset: {df_full}")
    st.stop()
else:
    st.success("Model trained and ready!")

st.write("### Analyze a URL")
url_input = st.text_input("Enter URL (e.g. https://google.com)")

if st.button("Predict"):
    if not url_input.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner(f"Extracting features from {url_input}..."):
            # Extract features
            input_df = extract_features(url_input, default_sample)
            
            # Predict
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            st.write("### Model Prediction")
            
            prob_phish = probability[1]
            
            if prediction == 1:
                st.error(f"Prediction: **Phishing Website 🚨** (Confidence: {prob_phish:.2%})")
            else:
                st.success(f"Prediction: **Legitimate Website ✅** (Confidence: {probability[0]:.2%})")
                
            with st.expander("View Extracted Features used for Prediction"):
                st.dataframe(input_df)
