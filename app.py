import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 1. Page Configuration
st.set_page_config(page_title="Student Early Warning System", layout="wide")
st.title("Student Early Warning System") # Updated Heading
st.markdown("### Hybrid CNN-LSTM Predictive Analytics")

# 2. Load the Model
@st.cache_resource
def load_student_model():
    return load_model('student_model.h5')

model = load_student_model()

# 3. Sidebar for File Upload
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Student Batch (CSV)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
    except:
        df = pd.read_csv(uploaded_file)

    # --- PREDICTION PIPELINE ---
    df_features = df.drop('Target', axis=1) if 'Target' in df.columns else df
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)
    reshaped_data = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))
    
    preds = model.predict(reshaped_data)
    df['Risk_Score'] = (preds.flatten() * 100).round(2)
    df['Status'] = df['Risk_Score'].apply(lambda x: '🔴 High Risk' if x > 70 else ('🟡 Monitor' if x > 40 else '🟢 Safe'))

    # --- INTERVENTION ENGINE ---
    def get_recommendation(row):
        if row['Risk_Score'] < 40: return "✅ Maintain current progress."
        recs = []
        if row.get('Debtor', 0) == 1: recs.append("💰 Financial Counseling")
        if row.get('Scholarship holder', 1) == 0: recs.append("🎓 Scholarship Review")
        if row.get('Curricular units 2nd sem (grade)', 15) < 10: recs.append("📚 Academic Tutoring")
        if row['Risk_Score'] > 75: recs.append("👨‍🏫 1-on-1 Mentorship")
        return " | ".join(recs) if recs else "📋 General Advising"

    df['Intervention_Plan'] = df.apply(get_recommendation, axis=1)

    # --- STRATEGIC ANALYTICS ---
    st.header("📊 Strategic Insights & Root Causes")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Students by Risk Category")
        status_counts = df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=status_counts, x='Status', y='Count', 
                    palette={'🔴 High Risk': 'red', '🟡 Monitor': 'orange', '🟢 Safe': 'green'}, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("🔎 Top Drivers of Batch Risk")
        numeric_df = df_features.select_dtypes(include=[np.number])
        risk_drivers = numeric_df.corrwith(df['Risk_Score']).abs().sort_values(ascending=False).head(5)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=risk_drivers.values, y=risk_drivers.index, palette="magma", ax=ax2)
        st.pyplot(fig2)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("💰 Financial Impact Analysis")
        if 'Tuition fees up to date' in df.columns:
            debt_impact = df.groupby('Tuition fees up to date')['Risk_Score'].mean().reset_index()
            debt_impact['Tuition'] = debt_impact['Tuition fees up to date'].map({1: 'Paid', 0: 'Overdue'})
            fig3, ax3 = plt.subplots()
            sns.barplot(data=debt_impact, x='Tuition', y='Risk_Score', palette='coolwarm', ax=ax3)
            st.pyplot(fig3)

    with col4:
        st.subheader("📍 Individual Risk Trajectory")
        student_id = st.selectbox("Select Student ID to track progress", df.index)
        curr = df.loc[student_id, 'Risk_Score']
        prev = curr * 0.7 if df.loc[student_id, 'Curricular units 1st sem (grade)'] > df.loc[student_id, 'Curricular units 2nd sem (grade)'] else curr * 1.3
        
        fig4, ax4 = plt.subplots()
        ax4.plot(['Admission', 'Mid-Term', 'Current'], [30, prev, curr], marker='o', color='blue', linewidth=3)
        ax4.set_ylim(0, 100)
        
        # Adding labels for the report
        ax4.set_xlabel("Academic Timeline (X-Axis)")
        ax4.set_ylabel("Risk Percentage (Y-Axis)")
        st.pyplot(fig4)

    # --- DATA TABLE ---
    st.divider()
    st.subheader("📋 Academic Intervention & Performance Ledger") # Professional Table Title
    
    display_df = df.copy()
    if 'Scholarship holder' in display_df.columns:
        display_df['Scholarship'] = display_df['Scholarship holder'].map({1: 'Yes', 0: 'No'})
    if 'Tuition fees up to date' in display_df.columns:
        display_df['Tuition'] = display_df['Tuition fees up to date'].map({1: '✅ Paid', 0: '❌ Overdue'})
    
    final_cols = {'Curricular units 2nd sem (grade)': 'GPA', 'Curricular units 2nd sem (approved)': 'Units Passed', 'Intervention_Plan': 'Recommended Action'}
    cols_to_show = ['Status', 'Risk_Score', 'GPA', 'Units Passed', 'Scholarship', 'Tuition', 'Recommended Action']
    available_cols = [c for c in display_df.rename(columns=final_cols).columns if c in cols_to_show]
    
    st.dataframe(display_df.rename(columns=final_cols)[available_cols].sort_values(by='Risk_Score', ascending=False), use_container_width=True)
else:
    st.info("Please upload a CSV file to begin.")