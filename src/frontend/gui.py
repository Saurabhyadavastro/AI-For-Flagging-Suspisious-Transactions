"""
Professional Streamlit GUI for AI-Powered Suspicious Transaction Flagging System
A clean, data-centric interface with dark theme and comprehensive functionality.
Enhanced with Ollama Llama 3.1 integration for intelligent chat capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import our modules
try:
    from src.backend.anomaly_detector import detect_anomalies_from_dataframe, IndianTransactionAnomalyDetector
    from src.database.supabase_client import SupabaseClient
    from src.database.database_manager import DatabaseManager
    from src.frontend.ollama_integration import OllamaChat, extract_transaction_id_from_query, format_transaction_details
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Suspicious Transaction Detector",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional dark theme
def load_css():
    st.markdown("""
    <style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #1f2937;
        --secondary-color: #374151;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --text-muted: #9ca3af;
        --border-color: #4b5563;
        --background-dark: #111827;
        --background-card: #1f2937;
    }
    
    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, var(--background-dark) 0%, var(--primary-color) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    h1 {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--accent-color), var(--success-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Card components */
    .custom-card {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    /* Metrics styling */
    .metric-card {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-color), #2563eb);
        border: none;
        border-radius: 8px;
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        background: var(--secondary-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: var(--background-card);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-color);
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* Data tables */
    .stDataFrame {
        background: var(--background-card);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--primary-color);
        border-right: 1px solid var(--border-color);
    }
    
    /* Chat container */
    .chat-container {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        height: 400px;
        overflow-y: auto;
        margin-top: 2rem;
    }
    
    .chat-message {
        background: var(--secondary-color);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--accent-color);
    }
    
    /* Status indicators */
    .status-success {
        color: var(--success-color);
        font-weight: 500;
    }
    
    .status-warning {
        color: var(--warning-color);
        font-weight: 500;
    }
    
    .status-error {
        color: var(--danger-color);
        font-weight: 500;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, var(--accent-color), var(--success-color));
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove extra padding */
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'ollama_chat' not in st.session_state:
        st.session_state.ollama_chat = OllamaChat()
    if 'llm_status_checked' not in st.session_state:
        st.session_state.llm_status_checked = False

# Authentication functions
def show_login_page():
    st.markdown('<h1>🇮🇳 AI Suspicious Transaction Detector</h1>', unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        # Tab selection
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Welcome Back")
            
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit_login:
                    if email and password:
                        try:
                            # Initialize Supabase client
                            supabase_client = SupabaseClient()
                            result = supabase_client.sign_in(email, password)
                            
                            if result and result.user:
                                st.session_state.authenticated = True
                                st.session_state.user_data = {
                                    'email': email,
                                    'user_id': result.user.id
                                }
                                st.success("Login successful!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Invalid credentials. Please try again.")
                        except Exception as e:
                            st.error(f"Login error: {str(e)}")
                    else:
                        st.warning("Please enter both email and password.")
        
        with tab2:
            st.subheader("Create Account")
            
            with st.form("register_form"):
                new_email = st.text_input("Email", placeholder="Enter your email")
                new_password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                submit_register = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit_register:
                    if new_email and new_password and confirm_password:
                        if new_password == confirm_password:
                            try:
                                # Initialize Supabase client
                                supabase_client = SupabaseClient()
                                result = supabase_client.sign_up(new_email, new_password)
                                
                                if result and result.user:
                                    st.success("Account created successfully! Please log in.")
                                else:
                                    st.error("Registration failed. Please try again.")
                            except Exception as e:
                                st.error(f"Registration error: {str(e)}")
                        else:
                            st.error("Passwords do not match.")
                    else:
                        st.warning("Please fill in all fields.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_main_dashboard():
    # Header with user info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<h1>🇮🇳 AI Suspicious Transaction Detector</h1>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<p class='status-success'>Welcome, {st.session_state.user_data['email']}</p>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.session_state.uploaded_data = None
            st.session_state.analysis_results = None
            st.rerun()
    
    st.markdown("---")
    
    # Main dashboard content
    show_file_upload_section()
    
    if st.session_state.analysis_results:
        show_analysis_results()
    
    show_chat_section()

def show_file_upload_section():
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📁 Upload Transaction Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your transaction data for suspicious activity analysis"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analysis configuration
        contamination = st.slider(
            "Expected Anomaly Rate (%)",
            min_value=1,
            max_value=20,
            value=8,
            help="Percentage of transactions expected to be suspicious"
        ) / 100
        
        ensemble_method = st.selectbox(
            "Detection Method",
            ["majority_vote", "conservative", "weighted_average"],
            help="Conservative: Lower false positives, Majority Vote: Balanced, Weighted: Performance-based"
        )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df
            
            # Display data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-label">Total Transactions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if 'transaction_amount_inr' in df.columns:
                    avg_amount = df['transaction_amount_inr'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">₹{avg_amount:,.0f}</div>
                        <div class="metric-label">Average Amount</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if 'transaction_date' in df.columns:
                    date_range = pd.to_datetime(df['transaction_date']).dt.date
                    days = (date_range.max() - date_range.min()).days + 1
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{days}</div>
                        <div class="metric-label">Days Span</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                if 'location' in df.columns:
                    unique_locations = df['location'].nunique()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{unique_locations}</div>
                        <div class="metric-label">Locations</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Analysis button
            if st.button("🔍 Analyze for Suspicious Transactions", use_container_width=True, type="primary"):
                with st.spinner("Analyzing transactions... This may take a few moments."):
                    try:
                        # Run anomaly detection
                        results = detect_anomalies_from_dataframe(
                            df,
                            contamination=contamination,
                            models_to_use=['isolation_forest', 'one_class_svm', 'dbscan'],
                            ensemble_method=ensemble_method
                        )
                        
                        st.session_state.analysis_results = results
                        st.success("Analysis completed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_analysis_results():
    results = st.session_state.analysis_results
    detection_results = results['detection_results']
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🚨 Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        suspicious_count = detection_results['total_suspicious_transactions']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value status-error">{suspicious_count:,}</div>
            <div class="metric-label">Suspicious Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_transactions = len(st.session_state.uploaded_data)
        detection_rate = (suspicious_count / total_transactions) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value status-warning">{detection_rate:.2f}%</div>
            <div class="metric-label">Detection Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'confidence_metrics' in detection_results and detection_results['confidence_metrics']:
            agreement = detection_results['confidence_metrics']['model_agreement_rate']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value status-success">{agreement:.2f}</div>
                <div class="metric-label">Model Agreement</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if len(detection_results['flagged_transactions']) > 0:
            avg_score = detection_results['flagged_transactions']['anomaly_score'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_score:.3f}</div>
                <div class="metric-label">Avg Anomaly Score</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    show_visualizations(detection_results)
    
    # Detailed results
    show_detailed_results(detection_results)

def show_visualizations(detection_results):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📈 Visual Analysis")
    
    # Create visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Geographic Hotspots", "Time Patterns", "Amount Analysis", "Transaction Details"])
    
    with viz_tab1:
        if 'suspicious_hotspots' in detection_results and detection_results['suspicious_hotspots']:
            hotspots = detection_results['suspicious_hotspots']
            
            # Create hotspots dataframe
            hotspot_data = []
            for location, data in hotspots.items():
                hotspot_data.append({
                    'Location': location,
                    'Suspicious_Transactions': data['suspicious_transactions'],
                    'Total_Transactions': data['total_transactions'],
                    'Suspicion_Rate': data['suspicion_rate_percentage']
                })
            
            hotspot_df = pd.DataFrame(hotspot_data)
            
            if not hotspot_df.empty:
                # Bar chart for geographic hotspots
                fig = px.bar(
                    hotspot_df.head(10),
                    x='Location',
                    y='Suspicion_Rate',
                    title='Top 10 Suspicious Locations',
                    labels={'Suspicion_Rate': 'Suspicion Rate (%)'},
                    color='Suspicion_Rate',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No geographic hotspot data available")
    
    with viz_tab2:
        if 'suspicious_time_patterns' in detection_results and detection_results['suspicious_time_patterns']:
            time_patterns = detection_results['suspicious_time_patterns']
            
            if 'hourly_distribution' in time_patterns:
                hourly = time_patterns['hourly_distribution']
                
                # Create hourly data
                hourly_data = []
                for hour, data in hourly.items():
                    hourly_data.append({
                        'Hour': int(hour),
                        'Suspicious_Count': data['suspicious_count'],
                        'Total_Count': data['total_count'],
                        'Suspicion_Rate': data['suspicion_rate_percentage']
                    })
                
                hourly_df = pd.DataFrame(hourly_data).sort_values('Hour')
                
                if not hourly_df.empty:
                    # Line chart for hourly patterns
                    fig = px.line(
                        hourly_df,
                        x='Hour',
                        y='Suspicion_Rate',
                        title='Suspicious Transaction Patterns by Hour',
                        labels={'Suspicion_Rate': 'Suspicion Rate (%)'},
                        markers=True
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time pattern data available")
    
    with viz_tab3:
        if len(detection_results['flagged_transactions']) > 0:
            flagged_df = detection_results['flagged_transactions']
            normal_df = st.session_state.uploaded_data[
                ~st.session_state.uploaded_data.index.isin(flagged_df.index)
            ]
            
            if 'transaction_amount_inr' in flagged_df.columns:
                # Amount distribution comparison
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=normal_df['transaction_amount_inr'],
                    name='Normal Transactions',
                    opacity=0.7,
                    nbinsx=50
                ))
                
                fig.add_trace(go.Histogram(
                    x=flagged_df['transaction_amount_inr'],
                    name='Suspicious Transactions',
                    opacity=0.7,
                    nbinsx=50
                ))
                
                fig.update_layout(
                    title='Transaction Amount Distribution',
                    xaxis_title='Amount (INR)',
                    yaxis_title='Count',
                    barmode='overlay',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No amount analysis data available")
    
    with viz_tab4:
        if len(detection_results['flagged_transactions']) > 0:
            flagged_df = detection_results['flagged_transactions']
            
            # Scatter plot of anomaly scores
            if 'anomaly_score' in flagged_df.columns and 'transaction_amount_inr' in flagged_df.columns:
                fig = px.scatter(
                    flagged_df,
                    x='transaction_amount_inr',
                    y='anomaly_score',
                    title='Suspicious Transactions: Amount vs Anomaly Score',
                    labels={
                        'transaction_amount_inr': 'Transaction Amount (INR)',
                        'anomaly_score': 'Anomaly Score'
                    },
                    hover_data=['transaction_id'] if 'transaction_id' in flagged_df.columns else None
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transaction detail data available")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_detailed_results(detection_results):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📋 Detailed Results")
    
    # Flagged transactions table
    if len(detection_results['flagged_transactions']) > 0:
        st.subheader("🚨 Suspicious Transactions")
        
        flagged_df = detection_results['flagged_transactions'].copy()
        
        # Format columns for better display
        if 'transaction_amount_inr' in flagged_df.columns:
            flagged_df['Amount (INR)'] = flagged_df['transaction_amount_inr'].apply(lambda x: f"₹{x:,.2f}")
        
        if 'anomaly_score' in flagged_df.columns:
            flagged_df['Risk Score'] = flagged_df['anomaly_score'].apply(lambda x: f"{x:.3f}")
        
        # Select display columns
        display_columns = []
        if 'transaction_id' in flagged_df.columns:
            display_columns.append('transaction_id')
        if 'Amount (INR)' in flagged_df.columns:
            display_columns.append('Amount (INR)')
        if 'transaction_time' in flagged_df.columns:
            display_columns.append('transaction_time')
        if 'location' in flagged_df.columns:
            display_columns.append('location')
        if 'payment_method' in flagged_df.columns:
            display_columns.append('payment_method')
        if 'Risk Score' in flagged_df.columns:
            display_columns.append('Risk Score')
        
        if display_columns:
            st.dataframe(
                flagged_df[display_columns].head(20),
                use_container_width=True
            )
            
            # Download button for results
            csv = flagged_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Suspicious Transactions (CSV)",
                data=csv,
                file_name=f"suspicious_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("No suspicious transactions detected.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_chat_section():
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    # Chat header with status indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("💬 AI Assistant Chat")
    
    with col2:
        # Display Ollama status
        ollama_status = st.session_state.ollama_chat.is_model_available()
        if ollama_status:
            st.markdown('<p class="status-success">🟢 Llama 3.1 Online</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">🟡 Offline Mode</p>', unsafe_allow_html=True)
    
    with col3:
        # Refresh status button
        if st.button("🔄 Refresh", help="Check Ollama connection"):
            st.session_state.ollama_chat = OllamaChat()
            st.rerun()
    
    # Show status info if Ollama is not available
    if not ollama_status and not st.session_state.llm_status_checked:
        st.info("""
        **🤖 Enhanced AI Mode Unavailable**
        
        The local Llama 3.1 model is not currently available. To enable full AI capabilities:
        
        1. **Install Ollama**: Visit [ollama.ai](https://ollama.ai) for installation
        2. **Pull Llama 3.1**: Run `ollama pull llama3.1` in your terminal
        3. **Start Service**: Ensure Ollama service is running
        
        **Current Mode**: Basic responses with analysis data
        """)
        st.session_state.llm_status_checked = True
    
    # Quick action buttons for common queries
    st.markdown("**Quick Questions:**")
    
    quick_cols = st.columns(4)
    
    with quick_cols[0]:
        if st.button("📊 Analysis Summary", use_container_width=True):
            quick_question = "Provide a comprehensive summary of the current analysis results"
            handle_chat_interaction(quick_question)
    
    with quick_cols[1]:
        if st.button("🗺️ Geographic Patterns", use_container_width=True):
            quick_question = "What are the main geographic patterns in suspicious transactions?"
            handle_chat_interaction(quick_question)
    
    with quick_cols[2]:
        if st.button("⏰ Time Patterns", use_container_width=True):
            quick_question = "What time patterns are evident in the suspicious transactions?"
            handle_chat_interaction(quick_question)
    
    with quick_cols[3]:
        if st.button("🔍 Risk Factors", use_container_width=True):
            quick_question = "What are the main risk factors causing transactions to be flagged?"
            handle_chat_interaction(quick_question)
    
    # Chat interface
    chat_col1, chat_col2 = st.columns([4, 1])
    
    with chat_col1:
        user_input = st.text_input(
            "Ask about the analysis results:",
            placeholder="e.g., 'Show me details for transaction TXN12345' or 'What's the riskiest location?'",
            key="chat_input"
        )
    
    with chat_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send_button = st.button("Send", use_container_width=True, type="primary")
    
    # Handle chat interaction
    if send_button and user_input:
        handle_chat_interaction(user_input)
    
    # Chat history display with improved styling
    st.markdown("**Conversation History:**")
    chat_container = st.container()
    
    with chat_container:
        # Display chat history in reverse order (most recent first)
        for chat in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
            if chat['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message" style="margin-left: 2rem; border-left-color: #3b82f6; background: #1e3a8a20;">
                    <strong>👤 You:</strong> {chat['message']}
                    <small style="color: #9ca3af; display: block; margin-top: 0.25rem;">
                        {chat['timestamp'].strftime('%H:%M:%S')}
                    </small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message" style="margin-right: 1rem; border-left-color: #10b981; background: #064e3b20;">
                    <strong>🤖 AI Assistant:</strong><br>
                    <div style="margin-top: 0.5rem;">{chat['message']}</div>
                    <small style="color: #9ca3af; display: block; margin-top: 0.5rem;">
                        {chat['timestamp'].strftime('%H:%M:%S')}
                        {' • Enhanced by Llama 3.1' if chat.get('enhanced_mode', False) else ' • Basic Mode'}
                    </small>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat controls
    if st.session_state.chat_history:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.ollama_chat.clear_history()
                st.rerun()
        
        with col2:
            # Export chat history
            if st.button("💾 Export Chat", use_container_width=True):
                chat_export = []
                for chat in st.session_state.chat_history:
                    chat_export.append({
                        'timestamp': chat['timestamp'].isoformat(),
                        'type': chat['type'],
                        'message': chat['message'],
                        'enhanced_mode': chat.get('enhanced_mode', False)
                    })
                
                chat_json = json.dumps(chat_export, indent=2)
                st.download_button(
                    label="Download Chat History (JSON)",
                    data=chat_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_chat_interaction(user_input: str):
    """
    Handle chat interaction with enhanced Ollama integration
    
    Args:
        user_input: User's input message
    """
    
    # Add user message to history
    st.session_state.chat_history.append({
        'type': 'user',
        'message': user_input,
        'timestamp': datetime.now()
    })
    
    # Show processing indicator
    with st.spinner("🤖 AI Assistant is thinking..."):
        
        # Check for specific transaction queries
        transaction_id = extract_transaction_id_from_query(user_input)
        
        if transaction_id and st.session_state.analysis_results:
            # Handle specific transaction queries
            ai_response = handle_transaction_query(user_input, transaction_id)
            enhanced_mode = False
        else:
            # Generate response using Ollama or fallback
            ai_response = st.session_state.ollama_chat.generate_response(
                user_input, 
                st.session_state.analysis_results, 
                st.session_state.uploaded_data
            )
            enhanced_mode = st.session_state.ollama_chat.is_model_available()
    
    # Add AI response to history
    st.session_state.chat_history.append({
        'type': 'ai',
        'message': ai_response,
        'timestamp': datetime.now(),
        'enhanced_mode': enhanced_mode
    })
    
    # Clear the input and rerun to show new messages
    st.rerun()

def handle_transaction_query(user_input: str, transaction_id: str) -> str:
    """
    Handle specific transaction queries with detailed analysis
    
    Args:
        user_input: User's question
        transaction_id: Extracted transaction ID
        
    Returns:
        str: Detailed response about the transaction
    """
    
    if not st.session_state.analysis_results:
        return "Please analyze transaction data first to get detailed transaction information."
    
    # Look for the transaction in flagged transactions
    flagged_df = st.session_state.analysis_results['detection_results']['flagged_transactions']
    
    # Try to find the transaction
    transaction_found = None
    
    # Search by transaction_id column if it exists
    if 'transaction_id' in flagged_df.columns:
        matching = flagged_df[flagged_df['transaction_id'].astype(str).str.contains(transaction_id, case=False, na=False)]
        if not matching.empty:
            transaction_found = matching.iloc[0]
    
    # Search by index if no transaction_id column
    if transaction_found is None:
        try:
            idx = int(transaction_id)
            if idx in flagged_df.index:
                transaction_found = flagged_df.loc[idx]
        except (ValueError, KeyError):
            pass
    
    if transaction_found is not None:
        details = format_transaction_details(transaction_found)
        
        # Get anomaly score analysis
        score = transaction_found.get('anomaly_score', 0)
        risk_level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
        
        response = f"""**🔍 Transaction Analysis: {transaction_id}**

{details}

**🎯 Risk Assessment:**
- **Risk Level:** {risk_level} Risk
- **Anomaly Score:** {score:.3f}
- **Status:** Flagged as Suspicious

**📊 Why This Transaction Was Flagged:**

The AI detected unusual patterns in one or more areas:
• **Amount Pattern:** {'High' if transaction_found.get('transaction_amount_inr', 0) > 50000 else 'Normal'} transaction amount
• **Timing:** {'Unusual' if 'transaction_time' in transaction_found.index else 'Standard'} time pattern
• **Location:** {'High-risk' if 'location' in transaction_found.index else 'Standard'} area identified
• **Behavioral:** Deviation from normal transaction patterns

**💡 Recommended Actions:**
1. Review customer transaction history
2. Verify transaction authenticity with customer
3. Check supporting documentation
4. Consider additional compliance review if risk score > 0.5
"""
        
        return response
    
    else:
        # Transaction not found in flagged transactions
        # Check if it exists in the original data
        if st.session_state.uploaded_data is not None:
            original_df = st.session_state.uploaded_data
            
            # Search in original data
            found_in_original = None
            if 'transaction_id' in original_df.columns:
                matching = original_df[original_df['transaction_id'].astype(str).str.contains(transaction_id, case=False, na=False)]
                if not matching.empty:
                    found_in_original = matching.iloc[0]
            
            if found_in_original is not None:
                details = format_transaction_details(found_in_original)
                
                return f"""**✅ Transaction Analysis: {transaction_id}**

{details}

**🎯 Risk Assessment:**
- **Status:** ✅ **Not Flagged** - Transaction appears normal
- **Risk Level:** Low Risk
- **Analysis:** This transaction passed all anomaly detection algorithms

**📊 Why This Transaction Was NOT Flagged:**
• Amount within normal patterns
• Standard timing and location
• Consistent with customer behavior
• No unusual characteristics detected

**💡 This transaction appears to be legitimate based on the AI analysis.**
"""
            
        return f"""**❓ Transaction {transaction_id} Not Found**

The specified transaction ID could not be located in either:
• Suspicious/flagged transactions
• Original uploaded dataset

**Possible reasons:**
• Transaction ID may be incorrect
• Transaction might not be in the uploaded dataset
• Different ID format than expected

**💡 Try:**
• Check the transaction ID spelling
• Use the exact ID from your dataset
• Browse the detailed results table to find the correct ID
"""

def generate_ai_response(user_input):
    """
    Legacy function - now redirects to Ollama integration
    Kept for backward compatibility
    """
    return st.session_state.ollama_chat.generate_response(
        user_input, 
        st.session_state.analysis_results, 
        st.session_state.uploaded_data
    )

# Main application
def main():
    load_css()
    init_session_state()
    
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_dashboard()

if __name__ == "__main__":
    main()
