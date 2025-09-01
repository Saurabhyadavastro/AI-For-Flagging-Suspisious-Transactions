"""
Enhanced Professional Streamlit GUI for AI-Powered Suspicious Transaction Flagging System
Features: File upload, data processing, professional styling, AI chat
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
import io
import csv

# Graceful imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    st.warning("📦 Pandas not available - using alternative data processing")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules with error handling
try:
    from src.frontend.ollama_integration import OllamaChat, extract_transaction_id_from_query, format_transaction_details
    OLLAMA_INTEGRATION_AVAILABLE = True
except ImportError as e:
    OLLAMA_INTEGRATION_AVAILABLE = False
    st.error(f"Ollama integration not available: {e}")

# Page configuration
st.set_page_config(
    page_title="AI For Flagging Suspicious Transactions",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
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
        max-width: 1400px;
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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-color);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status indicators */
    .status-indicator {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-high { background: rgba(239, 68, 68, 0.2); color: var(--danger-color); }
    .status-medium { background: rgba(245, 158, 11, 0.2); color: var(--warning-color); }
    .status-low { background: rgba(16, 185, 129, 0.2); color: var(--success-color); }
    
    /* Upload zone */
    .upload-zone {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: var(--background-card);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: var(--accent-color);
        background: rgba(59, 130, 246, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-color), var(--success-color));
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Chat styling */
    .chat-container {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        height: 400px;
        overflow-y: auto;
    }
    
    .chat-message {
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-radius: 8px;
        animation: fadeIn 0.3s ease;
    }
    
    .chat-user {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid var(--accent-color);
    }
    
    .chat-ai {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid var(--success-color);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--background-card);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stActionButton {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Sample transaction data
def get_sample_transaction_data():
    """Generate sample transaction data"""
    transactions = [
        {
            "Transaction_ID": "T001",
            "Account_Number": "ACC123456789",
            "Transaction_Amount": 850000,
            "Transaction_Type": "Transfer",
            "Location": "Mumbai",
            "Device_Type": "Mobile",
            "Risk_Score": 0.85,
            "Anomaly_Flag": "High Risk",
            "Transaction_Time": "2025-09-01 10:30:00",
            "Description": "Large transfer to new beneficiary"
        },
        {
            "Transaction_ID": "T002", 
            "Account_Number": "ACC987654321",
            "Transaction_Amount": 25000,
            "Transaction_Type": "Withdrawal",
            "Location": "Delhi",
            "Device_Type": "ATM",
            "Risk_Score": 0.65,
            "Anomaly_Flag": "Medium Risk",
            "Transaction_Time": "2025-09-01 11:15:00",
            "Description": "Unusual withdrawal pattern"
        },
        {
            "Transaction_ID": "T003",
            "Account_Number": "ACC456789123",
            "Transaction_Amount": 5000,
            "Transaction_Type": "Payment",
            "Location": "Bangalore",
            "Device_Type": "Web",
            "Risk_Score": 0.25,
            "Anomaly_Flag": "Low Risk",
            "Transaction_Time": "2025-09-01 12:00:00",
            "Description": "Regular merchant payment"
        },
        {
            "Transaction_ID": "T004",
            "Account_Number": "ACC789123456",
            "Transaction_Amount": 750000,
            "Transaction_Type": "Transfer",
            "Location": "Chennai",
            "Device_Type": "Mobile",
            "Risk_Score": 0.92,
            "Anomaly_Flag": "High Risk",
            "Transaction_Time": "2025-09-01 13:45:00",
            "Description": "Multiple large transfers in short time"
        },
        {
            "Transaction_ID": "T005",
            "Account_Number": "ACC321654987",
            "Transaction_Amount": 15000,
            "Transaction_Type": "Deposit",
            "Location": "Pune",
            "Device_Type": "Branch",
            "Risk_Score": 0.15,
            "Anomaly_Flag": "Low Risk",
            "Transaction_Time": "2025-09-01 14:20:00",
            "Description": "Regular salary deposit"
        }
    ]
    return transactions

def process_uploaded_file(uploaded_file):
    """Process uploaded CSV/Excel file and extract transaction data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            if PANDAS_AVAILABLE:
                df = pd.read_csv(uploaded_file)
                return df.to_dict('records')
            else:
                # Alternative CSV processing without pandas
                content = uploaded_file.read().decode('utf-8')
                csv_reader = csv.DictReader(io.StringIO(content))
                return list(csv_reader)
        
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            if PANDAS_AVAILABLE:
                df = pd.read_excel(uploaded_file)
                return df.to_dict('records')
            else:
                st.error("Excel files require pandas. Please use CSV format or install pandas.")
                return None
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def analyze_risk_score(amount, location, device_type, transaction_type):
    """Simple risk scoring algorithm"""
    risk_score = 0.0
    
    # Amount-based risk
    if amount > 500000:
        risk_score += 0.4
    elif amount > 100000:
        risk_score += 0.2
    elif amount < 1000:
        risk_score += 0.1
    
    # Location-based risk (simplified)
    high_risk_locations = ["Mumbai", "Delhi", "Chennai"]
    if location in high_risk_locations:
        risk_score += 0.2
    
    # Device-based risk
    if device_type == "Mobile":
        risk_score += 0.1
    elif device_type == "ATM":
        risk_score += 0.15
    
    # Transaction type risk
    if transaction_type == "Transfer":
        risk_score += 0.2
    elif transaction_type == "Withdrawal":
        risk_score += 0.15
    
    return min(risk_score, 1.0)

def classify_risk(risk_score):
    """Classify risk based on score"""
    if risk_score >= 0.7:
        return "High Risk"
    elif risk_score >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

def create_risk_distribution_chart(transactions):
    """Create risk distribution chart using Plotly"""
    risk_counts = {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0}
    
    for transaction in transactions:
        risk_level = transaction.get("Anomaly_Flag", "Unknown")
        if risk_level in risk_counts:
            risk_counts[risk_level] += 1
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(risk_counts.keys()),
            y=list(risk_counts.values()),
            marker_color=['#ef4444', '#f59e0b', '#10b981'],
            text=list(risk_counts.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Risk Level Distribution",
        xaxis_title="Risk Level",
        yaxis_title="Number of Transactions",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_size=16
    )
    
    return fig

def create_amount_distribution_chart(transactions):
    """Create transaction amount distribution chart"""
    amounts = []
    risk_levels = []
    
    for t in transactions:
        try:
            amount = float(t.get("Transaction_Amount", 0))
            amounts.append(amount)
            risk_levels.append(t.get("Anomaly_Flag", "Unknown"))
        except (ValueError, TypeError):
            continue
    
    colors = {'High Risk': '#ef4444', 'Medium Risk': '#f59e0b', 'Low Risk': '#10b981'}
    
    fig = go.Figure()
    
    for risk in ['High Risk', 'Medium Risk', 'Low Risk']:
        risk_amounts = [amounts[i] for i, r in enumerate(risk_levels) if r == risk]
        if risk_amounts:
            fig.add_trace(go.Histogram(
                x=risk_amounts,
                name=risk,
                marker_color=colors[risk],
                opacity=0.7
            ))
    
    fig.update_layout(
        title="Transaction Amount Distribution by Risk Level",
        xaxis_title="Transaction Amount (₹)",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_size=16,
        barmode='overlay'
    )
    
    return fig

def create_timeline_chart(transactions):
    """Create transaction timeline chart"""
    try:
        times = []
        amounts = []
        risk_levels = []
        
        for t in transactions:
            time_str = t.get("Transaction_Time", "")
            if time_str:
                try:
                    # Try different datetime formats
                    try:
                        time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            time_obj = datetime.strptime(time_str, "%Y-%m-%d")
                        except ValueError:
                            continue
                    
                    times.append(time_obj)
                    amounts.append(float(t.get("Transaction_Amount", 0)))
                    risk_levels.append(t.get("Anomaly_Flag", "Unknown"))
                except (ValueError, TypeError):
                    continue
        
        if not times:
            return None
        
        colors = {'High Risk': '#ef4444', 'Medium Risk': '#f59e0b', 'Low Risk': '#10b981'}
        
        fig = go.Figure()
        
        for risk in ['High Risk', 'Medium Risk', 'Low Risk']:
            risk_times = [times[i] for i, r in enumerate(risk_levels) if r == risk]
            risk_amounts = [amounts[i] for i, r in enumerate(risk_levels) if r == risk]
            
            if risk_times:
                fig.add_trace(go.Scatter(
                    x=risk_times,
                    y=risk_amounts,
                    mode='markers',
                    name=risk,
                    marker=dict(color=colors[risk], size=10),
                    text=[f"Amount: ₹{amt:,.0f}" for amt in risk_amounts],
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>%{text}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Transaction Timeline",
            xaxis_title="Time",
            yaxis_title="Transaction Amount (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating timeline chart: {str(e)}")
        return None

def sidebar_controls():
    """Create sidebar with controls and file upload"""
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # File upload section
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload transaction data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with transaction data"
    )
    
    # Sample data toggle
    use_sample_data = st.sidebar.checkbox("Use sample data", value=True)
    
    # Data processing options
    st.sidebar.markdown("### ⚙️ Processing Options")
    auto_analyze = st.sidebar.checkbox("Auto-analyze uploaded data", value=True)
    risk_threshold = st.sidebar.slider("Risk threshold", 0.0, 1.0, 0.7, 0.1)
    
    # Display options
    st.sidebar.markdown("### 🎨 Display Options")
    show_charts = st.sidebar.checkbox("Show charts", value=True)
    show_timeline = st.sidebar.checkbox("Show timeline", value=True)
    max_rows = st.sidebar.number_input("Max rows to display", 1, 1000, 100)
    
    return {
        'uploaded_file': uploaded_file,
        'use_sample_data': use_sample_data,
        'auto_analyze': auto_analyze,
        'risk_threshold': risk_threshold,
        'show_charts': show_charts,
        'show_timeline': show_timeline,
        'max_rows': max_rows
    }

def main():
    """Main application function"""
    # Load CSS
    load_css()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🔍 AI For Flagging Suspicious Transactions</h1>
        <p style="font-size: 1.2rem; color: var(--text-secondary);">
            Advanced AI system for flagging suspicious transactions with file upload capability
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    controls = sidebar_controls()
    
    # Data source logic
    transactions = []
    data_source = "No data"
    
    if controls['uploaded_file'] is not None:
        st.success("📁 File uploaded successfully!")
        uploaded_data = process_uploaded_file(controls['uploaded_file'])
        
        if uploaded_data:
            transactions = uploaded_data
            data_source = f"Uploaded file: {controls['uploaded_file'].name}"
            
            # Auto-analyze uploaded data
            if controls['auto_analyze']:
                with st.spinner("🔍 Analyzing uploaded data..."):
                    for transaction in transactions:
                        # Extract or convert numeric values
                        try:
                            amount = float(transaction.get('Transaction_Amount', 0))
                            location = str(transaction.get('Location', 'Unknown'))
                            device_type = str(transaction.get('Device_Type', 'Unknown'))
                            transaction_type = str(transaction.get('Transaction_Type', 'Unknown'))
                            
                            # Calculate risk score if not present
                            if 'Risk_Score' not in transaction:
                                risk_score = analyze_risk_score(amount, location, device_type, transaction_type)
                                transaction['Risk_Score'] = risk_score
                            
                            # Classify risk if not present
                            if 'Anomaly_Flag' not in transaction:
                                risk_score = float(transaction.get('Risk_Score', 0))
                                transaction['Anomaly_Flag'] = classify_risk(risk_score)
                                
                        except (ValueError, TypeError) as e:
                            st.warning(f"Error processing transaction: {e}")
                            continue
                
                st.success("✅ Data analysis completed!")
    
    elif controls['use_sample_data']:
        transactions = get_sample_transaction_data()
        data_source = "Sample data"
    
    # Display data source info
    st.info(f"📊 **Data Source:** {data_source} | **Records:** {len(transactions)}")
    
    if not transactions:
        st.warning("⚠️ No transaction data available. Please upload a file or enable sample data.")
        return
    
    # Calculate metrics
    total_transactions = len(transactions)
    high_risk_count = sum(1 for t in transactions if t.get("Anomaly_Flag") == "High Risk")
    medium_risk_count = sum(1 for t in transactions if t.get("Anomaly_Flag") == "Medium Risk")
    
    try:
        total_amount = sum(float(t.get("Transaction_Amount", 0)) for t in transactions)
        avg_risk_score = sum(float(t.get("Risk_Score", 0)) for t in transactions) / len(transactions)
    except (ValueError, TypeError):
        total_amount = 0
        avg_risk_score = 0
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: var(--accent-color);">{total_transactions}</div>
            <div class="metric-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: var(--danger-color);">{high_risk_count}</div>
            <div class="metric-label">High Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: var(--warning-color);">{medium_risk_count}</div>
            <div class="metric-label">Medium Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: var(--success-color);">₹{total_amount:,.0f}</div>
            <div class="metric-label">Total Amount</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: var(--warning-color);">{avg_risk_score:.2f}</div>
            <div class="metric-label">Avg Risk Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts section
    if controls['show_charts']:
        if controls['show_timeline']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.plotly_chart(create_risk_distribution_chart(transactions), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_amount_distribution_chart(transactions), use_container_width=True)
            
            with col3:
                timeline_chart = create_timeline_chart(transactions)
                if timeline_chart:
                    st.plotly_chart(timeline_chart, use_container_width=True)
                else:
                    st.info("Timeline chart not available (no valid timestamps)")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_risk_distribution_chart(transactions), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_amount_distribution_chart(transactions), use_container_width=True)
    
    # Transaction details table
    st.markdown("""
    <div class="custom-card">
        <h3>🔍 Transaction Analysis Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.selectbox("Filter by risk level:", ["All", "High Risk", "Medium Risk", "Low Risk"])
    
    with col2:
        sort_by = st.selectbox("Sort by:", ["Risk_Score", "Transaction_Amount", "Transaction_Time"])
    
    with col3:
        sort_order = st.selectbox("Sort order:", ["Descending", "Ascending"])
    
    # Apply filters
    filtered_transactions = transactions.copy()
    
    if risk_filter != "All":
        filtered_transactions = [t for t in filtered_transactions if t.get("Anomaly_Flag") == risk_filter]
    
    # Sort transactions
    reverse_order = sort_order == "Descending"
    try:
        if sort_by in ["Risk_Score", "Transaction_Amount"]:
            filtered_transactions.sort(key=lambda x: float(x.get(sort_by, 0)), reverse=reverse_order)
        else:
            filtered_transactions.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse_order)
    except (ValueError, TypeError):
        pass
    
    # Display limited rows
    display_transactions = filtered_transactions[:controls['max_rows']]
    
    # Create enhanced table with pandas if available
    if PANDAS_AVAILABLE and display_transactions:
        df = pd.DataFrame(display_transactions)
        
        # Format columns for better display
        if 'Transaction_Amount' in df.columns:
            df['Transaction_Amount'] = df['Transaction_Amount'].apply(lambda x: f"₹{float(x):,.0f}" if pd.notnull(x) else "₹0")
        
        if 'Risk_Score' in df.columns:
            df['Risk_Score'] = df['Risk_Score'].apply(lambda x: f"{float(x):.2f}" if pd.notnull(x) else "0.00")
        
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
    
    else:
        # Alternative table display without pandas
        if display_transactions:
            # Table headers
            headers = ["ID", "Account", "Amount", "Type", "Location", "Risk Score", "Status", "Time"]
            cols = st.columns(len(headers))
            
            for i, header in enumerate(headers):
                cols[i].markdown(f"**{header}**")
            
            st.markdown("---")
            
            # Table rows
            for transaction in display_transactions:
                cols = st.columns(len(headers))
                
                cols[0].write(transaction.get("Transaction_ID", "N/A"))
                account = transaction.get("Account_Number", "N/A")
                cols[1].write(account[:10] + "..." if len(account) > 10 else account)
                
                try:
                    amount = float(transaction.get("Transaction_Amount", 0))
                    cols[2].write(f"₹{amount:,.0f}")
                except:
                    cols[2].write("₹0")
                
                cols[3].write(transaction.get("Transaction_Type", "N/A"))
                cols[4].write(transaction.get("Location", "N/A"))
                
                try:
                    risk_score = float(transaction.get("Risk_Score", 0))
                    cols[5].write(f"{risk_score:.2f}")
                except:
                    cols[5].write("0.00")
                
                risk_level = transaction.get("Anomaly_Flag", "Unknown")
                risk_class = "status-high" if risk_level == "High Risk" else \
                            "status-medium" if risk_level == "Medium Risk" else "status-low"
                
                cols[6].markdown(f'<span class="status-indicator {risk_class}">{risk_level}</span>', unsafe_allow_html=True)
                cols[7].write(transaction.get("Transaction_Time", "N/A"))
    
    # AI Chat Section (if available)
    if OLLAMA_INTEGRATION_AVAILABLE:
        st.markdown("""
        <div class="custom-card">
            <h3>🤖 AI Transaction Analyst</h3>
            <p>Ask questions about transaction patterns, risk analysis, or specific transactions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat
        if "ollama_chat" not in st.session_state:
            st.session_state.ollama_chat = OllamaChat()
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Quick action buttons
        st.markdown("**Quick Actions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚨 Analyze High Risk"):
                query = "Analyze the high-risk transactions and their patterns"
                response = st.session_state.ollama_chat.chat_with_context(query, transactions)
                st.session_state.chat_history.append({"user": query, "ai": response, "timestamp": datetime.now()})
        
        with col2:
            if st.button("📊 Risk Trends"):
                query = "What trends do you see in the risk scores and transaction types?"
                response = st.session_state.ollama_chat.chat_with_context(query, transactions)
                st.session_state.chat_history.append({"user": query, "ai": response, "timestamp": datetime.now()})
        
        with col3:
            if st.button("💰 Amount Analysis"):
                query = "Analyze the transaction amounts and identify any anomalies"
                response = st.session_state.ollama_chat.chat_with_context(query, transactions)
                st.session_state.chat_history.append({"user": query, "ai": response, "timestamp": datetime.now()})
        
        with col4:
            if st.button("🌍 Location Patterns"):
                query = "Are there any suspicious geographical patterns in the transactions?"
                response = st.session_state.ollama_chat.chat_with_context(query, transactions)
                st.session_state.chat_history.append({"user": query, "ai": response, "timestamp": datetime.now()})
        
        # Chat interface
        chat_input = st.text_input("💬 Ask about transactions:", placeholder="e.g., Explain why transactions are flagged as high risk")
        
        if st.button("Send") and chat_input:
            response = st.session_state.ollama_chat.chat_with_context(chat_input, transactions)
            st.session_state.chat_history.append({"user": chat_input, "ai": response, "timestamp": datetime.now()})
            st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation History")
            
            for chat in reversed(st.session_state.chat_history[-5:]):  # Show last 5 messages
                timestamp = chat["timestamp"].strftime("%H:%M:%S")
                
                st.markdown(f"""
                <div class="chat-message chat-user">
                    <strong>You ({timestamp}):</strong> {chat["user"]}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message chat-ai">
                    <strong>AI Analyst:</strong> {chat["ai"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Status footer
    st.markdown("---")
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        st.markdown("**🔄 System Status:** ✅ Online")
    
    with status_col2:
        pandas_status = "✅ Available" if PANDAS_AVAILABLE else "⚠️ Fallback Mode"
        st.markdown(f"**📊 Pandas:** {pandas_status}")
    
    with status_col3:
        if OLLAMA_INTEGRATION_AVAILABLE:
            ollama_status = st.session_state.ollama_chat.check_ollama_status()
            status_icon = "✅" if ollama_status else "⚠️"
            status_text = "Connected" if ollama_status else "Fallback Mode"
            st.markdown(f"**🤖 AI Status:** {status_icon} {status_text}")
        else:
            st.markdown("**🤖 AI Status:** ❌ Not Available")
    
    with status_col4:
        st.markdown(f"**📊 Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
