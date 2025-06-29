import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.express as px

# Set page config with a more attractive theme
st.set_page_config(
    page_title="ChurnGuard - Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme - set to dark mode by default
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Function to toggle theme
def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# Custom CSS for both themes with animations
st.markdown(f"""
    <style>
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    @keyframes slideIn {{
        from {{ transform: translateX(-100%); }}
        to {{ transform: translateX(0); }}
    }}
    
    /* Main container */
    .main {{
        background-color: {'#0E1117' if st.session_state.theme == 'dark' else '#f5f5f5'};
        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
    }}
    
    /* Title animation */
    .title-container {{
        animation: fadeIn 1s ease-out;
    }}
    
    /* Upload button animation */
    .upload-button {{
        animation: pulse 2s infinite;
    }}
    
    /* Sidebar animation */
    .sidebar-content {{
        animation: slideIn 0.5s ease-out;
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else '#ffffff'};
        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    
    /* Dataframes and tables */
    .stDataFrame {{
        background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else '#ffffff'};
        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    /* Text elements */
    .stMarkdown {{
        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
    }}
    .stTextInput>div>div>input {{
        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
        background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else '#ffffff'};
        border-radius: 10px;
        padding: 10px;
    }}
    
    /* File uploader */
    .stFileUploader>div {{
        background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else '#ffffff'};
        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed {'#4CAF50' if st.session_state.theme == 'dark' else '#45a049'};
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else '#ffffff'};
        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
        border-radius: 10px;
        padding: 10px;
    }}
    
    /* Theme toggle button */
    .theme-toggle {{
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }}
    .theme-toggle button {{
        background-color: {'#4CAF50' if st.session_state.theme == 'dark' else '#333'};
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    .theme-toggle button:hover {{
        transform: scale(1.1);
    }}
    
    /* Plotly charts */
    .js-plotly-plot {{
        background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else '#ffffff'} !important;
        border-radius: 10px;
        padding: 20px;
    }}
    
    /* Info and error messages */
    .stAlert {{
        background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else '#ffffff'};
        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }}
    
    /* Progress bars */
    .stProgress>div>div>div {{
        background-color: #4CAF50;
        border-radius: 10px;
    }}
    
    /* Custom card style */
    .custom-card {{
        background-color: {'#1E1E1E' if st.session_state.theme == 'dark' else '#ffffff'};
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    .custom-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }}
    </style>
    """, unsafe_allow_html=True)

# Theme toggle button
col1, col2 = st.columns([1, 1])
with col2:
    if st.button('🌙 Dark Mode' if st.session_state.theme == 'light' else '🌞 Light Mode', 
                 key='theme_toggle', 
                 on_click=toggle_theme):
        st.experimental_rerun()

# Title and description with better formatting and animation
st.markdown(f"""
    <div class="title-container" style='text-align: center; padding: 30px; background: linear-gradient(135deg, #4CAF50, #45a049); color: white; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='margin: 0; font-size: 2.5em;'>🛡️ ChurnGuard</h1>
        <p style='margin: 10px 0 0 0; font-size: 1.2em;'>Customer Churn Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

# Main content with animation
st.markdown(f"""
    <div class="custom-card" style='text-align: center; padding: 20px;'>
        <h2 style='color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};'>Welcome to ChurnGuard</h2>
        <p style='font-size: 16px; color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};'>
            ChurnGuard helps businesses predict and analyze customer churn patterns using advanced machine learning. 
            Upload your customer data to get instant predictions and insights to improve your retention strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with animation
with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-content" style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4CAF50, #45a049); color: white; border-radius: 15px; margin-bottom: 20px;'>
            <h3 style='margin: 0;'>📊 Upload Data</h3>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="file_uploader")
    
    st.markdown(f"""
        <div class="custom-card" style='text-align: center; padding: 15px; margin-top: 20px;'>
            <p style='margin: 0; font-size: 14px; color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};'>
                Supported file format: CSV
            </p>
        </div>
    """, unsafe_allow_html=True)

def process_data(df):
    # Create a copy of the dataframe to avoid modifications to the original
    df = df.copy()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert binary categorical variables
    if 'gender' in df.columns:
        df["gender"] = df["gender"].map({'Male':0, 'Female':1})
    
    # Label encoding for binary columns
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] 
                   and df[col].nunique() == 2]
    
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    
    # One-hot encoding for categorical columns
    cat_cols = [col for col in df.columns if df[col].dtype not in [int, float] 
                and df[col].nunique() > 2
                and col not in ['customerID', 'Churn']]  # Exclude these columns from encoding
    
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Standardization of numerical columns
    num_cols = [col for col in df.columns if df[col].dtype in [int, float] 
                and col not in ['customerID', 'Churn']]
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

def train_model(df):
    # Prepare data for modeling
    if "Churn" not in df.columns:
        st.error("Error: 'Churn' column not found in the dataset!")
        return None, None
    
    # Convert Churn to numeric if it's not already
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({'No': 0, 'Yes': 1})
    
    y = df["Churn"]
    
    # Drop columns safely
    columns_to_drop = ['Churn']
    if 'customerID' in df.columns:
        columns_to_drop.append('customerID')
    
    X = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
    
    # Train model
    model = RandomForestClassifier(random_state=46)
    model.fit(X_train, y_train)
    
    # Calculate and display accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    st.write(f"Training Accuracy: {train_accuracy:.2%}")
    st.write(f"Testing Accuracy: {test_accuracy:.2%}")
    
    return model, X.columns

def main():
    if uploaded_file is not None:
        try:
            # Load data with progress indicator
            with st.spinner('Loading and processing data...'):
                df = pd.read_csv(uploaded_file)
            
            st.success("✅ Data successfully loaded!")
            
            # Show raw data in an expandable section
            with st.expander("📊 View Raw Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # Check required columns
            required_columns = ['Churn']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Error: Missing required columns: {', '.join(missing_columns)}")
                st.info("ℹ️ Your dataset must contain a 'Churn' column with values indicating whether customers churned (e.g., 'Yes'/'No' or 1/0)")
                return
            
            # Data processing
            with st.spinner('Processing data...'):
                processed_df = process_data(df)
            
            # Train model
            with st.spinner('Training model...'):
                model, feature_names = train_model(processed_df)
            
            if model is not None and feature_names is not None:
                # Save model
                with open('churn_model.pkl', 'wb') as file:
                    pickle.dump(model, file)
                
                # Model Analysis with better visualization
                st.markdown(f"""
                    <div class="custom-card" style='text-align: center; padding: 20px; margin: 20px 0;'>
                        <h3 style='color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};'>Model Analysis</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Feature importance with Plotly
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(feature_importance.head(10), 
                           x='importance', 
                           y='feature',
                           orientation='h',
                           title='Top 10 Most Important Features',
                           color='importance',
                           color_continuous_scale='Viridis')
                
                fig.update_layout(
                    height=500,
                    width=800,
                    showlegend=False,
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white' if st.session_state.theme == 'dark' else 'black')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Churn Distribution with Plotly
                churn_counts = df['Churn'].value_counts()
                fig = px.pie(values=churn_counts.values,
                           names=churn_counts.index,
                           title='Churn Distribution',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white' if st.session_state.theme == 'dark' else 'black')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button with better styling
                st.markdown(f"""
                    <div class="custom-card" style='text-align: center; margin: 20px 0;'>
                        <h4 style='color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};'>Download Trained Model</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                with open('churn_model.pkl', 'rb') as file:
                    st.download_button(
                        label="⬇️ Download Model",
                        data=file,
                        file_name="churn_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please make sure your CSV file has the correct format and required columns.")
            st.info("""
            ℹ️ Required format for the dataset:
            - Must be a CSV file
            - Must contain a 'Churn' column (values can be 'Yes'/'No' or 1/0)
            - Should contain customer features (both numerical and categorical)
            - Should not contain any special characters in column names
            """)
            
    else:
        st.info("ℹ️ Please upload a CSV file to begin the analysis.")
        st.markdown(f"""
            <div class="custom-card" style='padding: 20px;'>
                <h3 style='color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};'>Sample Dataset Format</h3>
                <p style='color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};'>
                    Your dataset should look something like this:
                </p>
                <div style='overflow-x: auto;'>
                    <table style='width: 100%; border-collapse: collapse;'>
                        <tr style='background-color: {'#2E2E2E' if st.session_state.theme == 'dark' else '#f8f9fa'};'>
                            <th style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>customerID</th>
                            <th style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>gender</th>
                            <th style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>SeniorCitizen</th>
                            <th style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>Partner</th>
                            <th style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>Churn</th>
                        </tr>
                        <tr>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>1234</td>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>Male</td>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>0</td>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>Yes</td>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>No</td>
                        </tr>
                        <tr style='background-color: {'#2E2E2E' if st.session_state.theme == 'dark' else '#f8f9fa'};'>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>5678</td>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>Female</td>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>1</td>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>No</td>
                            <td style='padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};'>Yes</td>
                        </tr>
                    </table>
                </div>
                <p style='margin-top: 20px; color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'};'>
                    The only required column is 'Churn', which should indicate whether the customer churned (Yes/No or 1/0).
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 