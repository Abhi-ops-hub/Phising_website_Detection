import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Phishing Website Detection")
st.markdown("This Streamlit app uses a Random Forest model to predict whether a website is a phishing attempt based on its extracted features.")

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
        
        # Save a sample row for default values
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
    st.success("Model trained successfully! Ready for predictions.")

# App interaction modes
option = st.radio("Choose Input Method", 
                  ["Use Random Sample from Dataset", "Manual Feature Entry"])

if option == "Use Random Sample from Dataset":
    st.subheader("Predict on a Random Sample")
    
    if st.button("Generate Random Sample & Predict"):
        # Select a random row
        random_row = df_full.sample(1)
        true_label = random_row['label'].values[0]
        input_data = random_row.drop("label", axis=1)
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.write("### Model Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("True Label", "Phishing 🚨" if true_label == 1 else "Legitimate ✅")
        with col2:
            st.metric("Predicted Label", "Phishing 🚨" if prediction == 1 else "Legitimate ✅")
            
        prob_phish = probability[1]
        
        if prediction == 1:
            st.error(f"Prediction: **Phishing Website** (Confidence: {prob_phish:.2%})")
        else:
            st.success(f"Prediction: **Legitimate Website** (Confidence: {probability[0]:.2%})")
            
        with st.expander("Show Features for this Sample"):
            st.dataframe(input_data)
            
elif option == "Manual Feature Entry":
    st.write("### Enter the features below:")
    
    # Create input form
    with st.form("prediction_form"):
        input_data = {}
        
        # Group features in columns
        cols = st.columns(4)
        for i, feature in enumerate(features):
            col = cols[i % 4]
            # Use default value from sample_data to make it easier for the user
            default_val = float(default_sample[feature])
            
            # Use appropriate input (int if default is integer, float otherwise)
            if default_val.is_integer():
                input_data[feature] = col.number_input(feature, value=int(default_val), step=1)
            else:
                input_data[feature] = col.number_input(feature, value=float(default_val), format="%f")
                
        submit_button = st.form_submit_button(label='Predict')
        
    if submit_button:
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        st.write("### Model Prediction")
        
        if prediction == 1:
            st.error(f"Prediction: **Phishing Website** (Confidence: {probability[1]:.2%})")
        else:
            st.success(f"Prediction: **Legitimate Website** (Confidence: {probability[0]:.2%})")
