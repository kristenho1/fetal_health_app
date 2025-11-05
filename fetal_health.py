# Tool: ChaGPT (GPT-5)
# Purpose: Debugged errors; explained how to add color based on fetal health; suggested app customizations using markdown; 
# Usage: Adopted adding color to dataframe ideas; modified markdown code based on suggestions for customizing and adding emojis
# Location: Documented here and further comments in traffic.ipynb

# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# title 
st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', width = 700)
st.write('Utilize our advanced Machine Learning application to predict fetal health classifications.')

# Sidebar information
st.sidebar.subheader('Fetal Health features Input')
user_df = st.sidebar.file_uploader('Upload your data', type = ['csv'], help = 'File must be in csv format')

st.sidebar.warning('**⚠️ Ensure your data strictly follows the format outlines below.**')
orig_df = pd.read_csv('fetal_health.csv')
orig_df = orig_df.drop(columns = ['fetal_health'])
st.sidebar.dataframe(orig_df.head(5), use_container_width=True)

# Selecting algorithm and Load the pre-trained model from the pickle file
model = st.sidebar.radio('**Choose Model for Prediction**', options = ['Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting'])
if model == 'Decision Tree':
    st.sidebar.info('**✔️ You selected: Decision Tree**')
    dt_pickle = open('decision_tree_fetal.pickle', 'rb') 
    clf_dt = pickle.load(dt_pickle) 
    dt_pickle.close()
elif model == 'Random Forest':
    st.sidebar.info('**✔️ You selected: Random Forest**')
    rf_pickle = open('random_forest_fetal.pickle', 'rb') 
    clf_rf = pickle.load(rf_pickle) 
    rf_pickle.close()
elif model == 'AdaBoost':
    st.sidebar.info('**✔️ You selected: AdaBoost**')
    ada_pickle = open('Adaboost_fetal.pickle', 'rb') 
    clf_ada = pickle.load(ada_pickle) 
    ada_pickle.close()
else:
    st.sidebar.info('**✔️ You selected: Soft Voting**')
    sv_pickle = open('soft_voting_fetal.pickle', 'rb') 
    clf_sv = pickle.load(sv_pickle) 
    sv_pickle.close()

if user_df is not None: 
    st.success('*✅ CSV file uploaded successfully*')
    input_df = pd.read_csv(user_df)

    st.subheader(f"Predictiong Fetal Helath Class Using {model} Model")

    # Dropping null values
    input_df = input_df.dropna().reset_index(drop = True) 
    orig_df = orig_df.dropna().reset_index(drop = True)
    
    # Ensure the order of columns in user data is in the same order as that of original data
    input_df = input_df[orig_df.columns]

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([orig_df, input_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = orig_df.shape[0]

    # Create dummies for the combined dataframe 
    combined_df_encoded = pd.get_dummies(combined_df)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

    # Used ChatGPT to help with debugging errors for predction probabilities
    if model == 'Random Forest':
        pred_rf = clf_rf.predict(user_df_encoded)
        prob_rf = clf_rf.predict_proba(user_df_encoded)
        pred_prob_rf = np.round(np.max(prob_rf, axis=1) * 100, 1)
        input_df['Predicted Fetal Health'] = pred_rf
        input_df['Prediction Probability (%)'] = pred_prob_rf 
    elif model == 'Decision Tree':
        pred_dt = clf_dt.predict(user_df_encoded)
        prob_dt = clf_dt.predict_proba(user_df_encoded)
        pred_prob_dt = np.round(np.max(prob_dt, axis=1) * 100, 1)
        input_df['Predicted Fetal Health'] = pred_dt
        input_df['Prediction Probability (%)'] = pred_prob_dt
    elif model == 'AdaBoost':
        pred_ada = clf_ada.predict(user_df_encoded)
        prob_ada = clf_ada.predict_proba(user_df_encoded)
        pred_prob_ada = np.round(np.max(prob_ada, axis=1) * 100, 1)
        input_df['Predicted Fetal Health'] = pred_ada
        input_df['Prediction Probability (%)'] = pred_prob_ada
    elif model == 'Soft Voting':
        pred_sv = clf_sv.predict(user_df_encoded)
        prob_sv = clf_sv.predict_proba(user_df_encoded)
        pred_prob_sv = np.round(np.max(prob_sv, axis=1) * 100, 1)
        input_df['Predicted Fetal Health'] = pred_sv
        input_df['Prediction Probability (%)'] = pred_prob_sv
    
    #Used ChatGPT to help format the fetal health colors 
    def fetal_health_color(val):
        if val == 'Normal':
            color = 'limegreen'
        elif val == 'Suspect':
            color = 'yellow'
        else:
            color = 'orange'
        return f'background-color: {color}'
    input_df = input_df.style.format({'Prediction Probability (%)': '{:.1f}'}).applymap(fetal_health_color, subset=['Predicted Fetal Health'])
    st.dataframe(input_df)
else:
    st.info('*ℹ️ Please upload data to proceed.*')

# Display model performance and insights
if user_df is not None:
    if model == 'Random Forest':
        # Showing additional items in tabs
        st.subheader("Model Performance and Insights")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('rf_confusion_mat.svg')

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('rf_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Greens').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 3: Feature Importance
        with tab3:
            st.write("### Feature Importance")
            st.image('rf_feature_imp.svg')
    elif model == 'Decision Tree':
        # Showing additional items in tabs
        st.subheader("Model Performance and Insights")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('dt_confusion_mat.svg')

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('dt_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Blues').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 3: Feature Importance
        with tab3:
            st.write("### Feature Importance")
            st.image('dt_feature_imp.svg')
    elif model == 'AdaBoost':
        # Showing additional items in tabs
        st.subheader("Model Performance and Insights")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('ada_confusion_mat.svg')

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('ada_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Oranges').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 3: Feature Importance
        with tab3:
            st.write("### Feature Importance")
            st.image('ada_feature_imp.svg')
    else:
        # Showing additional items in tabs
        st.subheader("Model Performance and Insights")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('sv_confusion_mat.svg')

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('sv_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Purples').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 3: Feature Importance
        with tab3:
            st.write("### Feature Importance")
            st.image('sv_feature_imp.svg')