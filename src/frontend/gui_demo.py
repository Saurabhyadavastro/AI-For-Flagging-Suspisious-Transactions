"""
Simple GUI Demo for AI Suspicious Transaction Flagging System
Demonstrates the GUI functionality with mock data and simplified dependencies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="AI Suspicious Transaction Detector - Demo",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional dark theme
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #1f2937;
        --secondary-color: #374151;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --background-dark: #111827;
        --background-card: #1f2937;
        --border-color: #4b5563;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--background-dark) 0%, var(--primary-color) 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    .custom-card {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
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
    
    .status-success { color: var(--success-color); font-weight: 500; }
    .status-warning { color: var(--warning-color); font-weight: 500; }
    .status-error { color: var(--danger-color); font-weight: 500; }
    
    h1 {
        text-align: center;
        background: linear-gradient(135deg, var(--accent-color), var(--success-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-color), #2563eb);
        border: none;
        border-radius: 8px;
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        width: 100%;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Mock data generation
def generate_mock_data():
    """Generate mock transaction data for demonstration"""
    np.random.seed(42)
    
    # Generate 1000 mock transactions
    n_transactions = 1000
    
    transactions = []
    for i in range(n_transactions):
        # Create some suspicious patterns
        is_suspicious = np.random.random() < 0.08
        
        if is_suspicious:
            # Suspicious patterns
            hour = np.random.choice([0, 1, 2, 3, 23])  # Late night
            amount = np.random.choice([50000, 100000, 250000, 500000])  # Round amounts
        else:
            # Normal patterns
            hour = np.random.choice(range(9, 18))
            amount = np.random.lognormal(mean=7.5, sigma=1.2)
            amount = max(100, min(amount, 100000))
        
        transaction = {
            'transaction_id': f'TXN{i:06d}',
            'customer_id': f'CUST{np.random.randint(1000, 9999)}',
            'transaction_amount_inr': round(amount, 2),
            'transaction_date': datetime.now().date(),
            'transaction_time': f'{hour:02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}',
            'payment_method': np.random.choice(['UPI', 'NEFT', 'IMPS', 'Card', 'RTGS']),
            'location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune']),
            'merchant_category': np.random.choice(['Grocery', 'Restaurant', 'Fuel', 'Retail', 'Investment']),
            'is_suspicious': is_suspicious
        }
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def mock_analysis_results(df):
    """Generate mock analysis results"""
    suspicious_df = df[df['is_suspicious']]
    
    # Create mock hotspots
    hotspots = {}
    for location in df['location'].unique():
        location_data = df[df['location'] == location]
        suspicious_count = len(location_data[location_data['is_suspicious']])
        total_count = len(location_data)
        
        hotspots[location] = {
            'suspicious_transactions': suspicious_count,
            'total_transactions': total_count,
            'suspicion_rate_percentage': (suspicious_count / total_count * 100) if total_count > 0 else 0
        }
    
    # Create mock time patterns
    time_patterns = {'hourly_distribution': {}}
    for hour in range(24):
        hour_data = df[pd.to_datetime(df['transaction_time'], format='%H:%M:%S').dt.hour == hour]
        if len(hour_data) > 0:
            suspicious_count = len(hour_data[hour_data['is_suspicious']])
            total_count = len(hour_data)
            
            time_patterns['hourly_distribution'][str(hour)] = {
                'suspicious_count': suspicious_count,
                'total_count': total_count,
                'suspicion_rate_percentage': (suspicious_count / total_count * 100) if total_count > 0 else 0
            }
    
    # Add anomaly scores to suspicious transactions
    suspicious_df = suspicious_df.copy()
    suspicious_df['anomaly_score'] = np.random.uniform(0.1, 0.5, len(suspicious_df))
    
    return {
        'detection_results': {
            'total_suspicious_transactions': len(suspicious_df),
            'flagged_transactions': suspicious_df,
            'suspicious_hotspots': hotspots,
            'suspicious_time_patterns': time_patterns,
            'confidence_metrics': {
                'model_agreement_rate': 0.85,
                'models_used': ['isolation_forest', 'one_class_svm', 'dbscan'],
                'ensemble_method_used': 'majority_vote'
            }
        }
    }

# Initialize session state
if 'demo_authenticated' not in st.session_state:
    st.session_state.demo_authenticated = False
if 'demo_data' not in st.session_state:
    st.session_state.demo_data = None
if 'demo_results' not in st.session_state:
    st.session_state.demo_results = None

def show_demo_login():
    """Demo login page"""
    st.markdown('<h1>🇮🇳 AI Suspicious Transaction Detector</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Demo Access")
        st.info("This is a demonstration version. Click below to access the demo dashboard.")
        
        if st.button("🚀 Enter Demo", use_container_width=True, type="primary"):
            st.session_state.demo_authenticated = True
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_demo_dashboard():
    """Demo dashboard"""
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1>🇮🇳 AI Suspicious Transaction Detector - Demo</h1>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Reset Demo", use_container_width=True):
            st.session_state.demo_authenticated = False
            st.session_state.demo_data = None
            st.session_state.demo_results = None
            st.rerun()
    
    st.markdown("---")
    
    # File upload simulation
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📁 Transaction Data Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("Demo Mode: Using pre-generated sample transaction data")
    
    with col2:
        contamination = st.slider("Expected Anomaly Rate (%)", 1, 20, 8) / 100
        ensemble_method = st.selectbox(
            "Detection Method", 
            ["majority_vote", "conservative", "weighted_average"]
        )
    
    if st.button("🎯 Generate Demo Data & Analyze", use_container_width=True, type="primary"):
        with st.spinner("Generating demo data and running analysis..."):
            # Generate mock data
            st.session_state.demo_data = generate_mock_data()
            time.sleep(2)  # Simulate processing time
            
            # Generate mock results
            st.session_state.demo_results = mock_analysis_results(st.session_state.demo_data)
            
            st.success("Demo analysis completed!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show results if available
    if st.session_state.demo_data is not None:
        show_demo_data_preview()
    
    if st.session_state.demo_results is not None:
        show_demo_results()

def show_demo_data_preview():
    """Show demo data preview"""
    df = st.session_state.demo_data
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📊 Demo Data Preview")
    
    # Data summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_amount = df['transaction_amount_inr'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{avg_amount:,.0f}</div>
            <div class="metric-label">Average Amount</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_locations = df['location'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_locations}</div>
            <div class="metric-label">Locations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        payment_methods = df['payment_method'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{payment_methods}</div>
            <div class="metric-label">Payment Methods</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_demo_results():
    """Show demo analysis results"""
    results = st.session_state.demo_results
    detection_results = results['detection_results']
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🚨 Demo Analysis Results")
    
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
        total_transactions = len(st.session_state.demo_data)
        detection_rate = (suspicious_count / total_transactions) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value status-warning">{detection_rate:.2f}%</div>
            <div class="metric-label">Detection Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        agreement = detection_results['confidence_metrics']['model_agreement_rate']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value status-success">{agreement:.2f}</div>
            <div class="metric-label">Model Agreement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_score = detection_results['flagged_transactions']['anomaly_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_score:.3f}</div>
            <div class="metric-label">Avg Anomaly Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    show_demo_visualizations(detection_results)
    
    # Detailed results
    show_demo_detailed_results(detection_results)

def show_demo_visualizations(detection_results):
    """Show demo visualizations"""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📈 Demo Visual Analysis")
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Geographic Hotspots", "Time Patterns", "Amount Analysis"])
    
    with viz_tab1:
        hotspots = detection_results['suspicious_hotspots']
        hotspot_data = []
        for location, data in hotspots.items():
            hotspot_data.append({
                'Location': location,
                'Suspicion_Rate': data['suspicion_rate_percentage']
            })
        
        hotspot_df = pd.DataFrame(hotspot_data)
        
        fig = px.bar(
            hotspot_df,
            x='Location',
            y='Suspicion_Rate',
            title='Suspicious Transaction Hotspots',
            color='Suspicion_Rate',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        time_patterns = detection_results['suspicious_time_patterns']['hourly_distribution']
        hourly_data = []
        for hour, data in time_patterns.items():
            hourly_data.append({
                'Hour': int(hour),
                'Suspicion_Rate': data['suspicion_rate_percentage']
            })
        
        hourly_df = pd.DataFrame(hourly_data).sort_values('Hour')
        
        fig = px.line(
            hourly_df,
            x='Hour',
            y='Suspicion_Rate',
            title='Suspicious Patterns by Hour',
            markers=True
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab3:
        flagged_df = detection_results['flagged_transactions']
        normal_df = st.session_state.demo_data[~st.session_state.demo_data['is_suspicious']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=normal_df['transaction_amount_inr'],
            name='Normal Transactions',
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.add_trace(go.Histogram(
            x=flagged_df['transaction_amount_inr'],
            name='Suspicious Transactions',
            opacity=0.7,
            nbinsx=30
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
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_demo_detailed_results(detection_results):
    """Show demo detailed results"""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📋 Demo Detailed Results")
    
    flagged_df = detection_results['flagged_transactions'].copy()
    
    # Format for display
    flagged_df['Amount (INR)'] = flagged_df['transaction_amount_inr'].apply(lambda x: f"₹{x:,.2f}")
    flagged_df['Risk Score'] = flagged_df['anomaly_score'].apply(lambda x: f"{x:.3f}")
    
    display_columns = ['transaction_id', 'Amount (INR)', 'transaction_time', 'location', 'payment_method', 'Risk Score']
    
    st.dataframe(flagged_df[display_columns].head(10), use_container_width=True)
    
    # Mock download
    st.info("💡 In the full version, you can download the complete results as CSV")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_demo_chat():
    """Demo chat section"""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("💬 AI Assistant Demo")
    
    st.info("💡 Demo Mode: Chat responses are simulated")
    
    user_input = st.text_input("Ask about the analysis:", placeholder="e.g., 'Why are these transactions suspicious?'")
    
    if st.button("Send Demo Message"):
        if user_input:
            st.markdown(f"**You:** {user_input}")
            st.markdown("**AI Assistant:** Based on the demo analysis, the flagged transactions show patterns typical of suspicious activity including late-night timing, round amounts, and unusual location patterns. The AI uses ensemble learning with multiple algorithms to identify these anomalies with high confidence.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main demo application"""
    load_css()
    
    if not st.session_state.demo_authenticated:
        show_demo_login()
    else:
        show_demo_dashboard()
        show_demo_chat()

if __name__ == "__main__":
    main()
