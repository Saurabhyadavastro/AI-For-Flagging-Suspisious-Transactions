"""Main Streamlit application for AI For Flagging Suspicious Transactions."""

import streamlit as st
import pandas as pd
from typing import Optional

# Import local modules (will be created later)
# from src.database.supabase_client import SupabaseClient
# from src.ml_models.anomaly_detector import AnomalyDetector
# from src.chat.ollama_client import OllamaClient


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="AI For Flagging Suspicious Transactions",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔍 AI For Flagging Suspicious Transactions")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Upload Data", "Model Training", "Chat Assistant", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Upload Data":
        show_upload_page()
    elif page == "Model Training":
        show_model_training()
    elif page == "Chat Assistant":
        show_chat_assistant()
    elif page == "Settings":
        show_settings()


def show_dashboard() -> None:
    """Display the main dashboard."""
    st.header("📊 Transaction Monitoring Dashboard")
    
    # Create sample metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "12,543", "↑ 2.3%")
    
    with col2:
        st.metric("Flagged Transactions", "47", "↓ 1.2%")
    
    with col3:
        st.metric("Accuracy", "94.2%", "↑ 0.5%")
    
    with col4:
        st.metric("False Positive Rate", "2.1%", "↓ 0.3%")
    
    # Placeholder for charts
    st.subheader("Transaction Volume Over Time")
    st.info("📈 Transaction volume chart will be displayed here")
    
    st.subheader("Recent Flagged Transactions")
    st.info("🚨 Recent suspicious transactions will be displayed here")


def show_upload_page() -> None:
    """Display the data upload page."""
    st.header("📁 Upload Transaction Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with transaction data",
        type=["csv"],
        help="Upload a CSV file containing transaction records"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} transactions")
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Data Summary")
            st.write(df.describe())
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to get started")


def show_model_training() -> None:
    """Display the model training page."""
    st.header("🤖 Model Training & Evaluation")
    
    st.info("🔧 Model training interface will be implemented here")
    
    if st.button("Start Training"):
        with st.spinner("Training model..."):
            # Placeholder for model training
            import time
            time.sleep(3)
        st.success("✅ Model training completed!")


def show_chat_assistant() -> None:
    """Display the chat assistant page."""
    st.header("💬 AI Chat Assistant")
    
    st.info("🤖 Chat interface with Ollama will be implemented here")
    
    # Chat interface placeholder
    user_input = st.text_input("Ask me about transactions or anomalies:")
    
    if user_input:
        st.write(f"You: {user_input}")
        st.write("AI Assistant: This feature will be implemented with Ollama integration.")


def show_settings() -> None:
    """Display the settings page."""
    st.header("⚙️ Application Settings")
    
    st.subheader("Model Configuration")
    anomaly_threshold = st.slider("Anomaly Detection Threshold", 0.0, 1.0, 0.5)
    
    st.subheader("Database Settings")
    st.info("🔒 Database configuration will be managed through environment variables")
    
    st.subheader("LLM Settings")
    ollama_model = st.selectbox("Ollama Model", ["llama3.1", "llama2", "mistral"])
    
    if st.button("Save Settings"):
        st.success("✅ Settings saved successfully!")


if __name__ == "__main__":
    main()
