# 🔍 AI For Flagging Suspicious Transactions

A comprehensive AI-powered system for detecting anomalous patterns in transactions, built with Streamlit and modern ML techniques.

## 🌟 Features

- **Real-time Anomaly Detection**: Advanced algorithms to identify suspicious transaction patterns
- **Interactive Dashboard**: Professional Streamlit-based GUI with dark theme
- **File Upload Support**: CSV and Excel file processing capabilities
- **AI-Powered Analysis**: Intelligent chat system for transaction insights
- **Multi-factor Risk Assessment**: Comprehensive scoring based on amount, location, and timing
- **Interactive Visualizations**: Charts, graphs, and geographic mapping
- **Export Capabilities**: Generate reports and export analysis results

## 🚀 Live Demo

🌐 **Production URL**: [https://your-app-name.onrender.com](https://your-app-name.onrender.com)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Local Development](#-local-development)
- [Deployment](#-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠 Installation

### Prerequisites

- Python 3.9+ (Python 3.11 recommended)
- Git
- Virtual environment tool (venv, conda, etc.)

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AI-For-Flagging-Suspicious-Transactions.git
   cd AI-For-Flagging-Suspicious-Transactions
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Using venv
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

5. **Run the application**:
   ```bash
   streamlit run main_gui_enhanced.py
   ```

6. **Access the application**:
   Open your browser to `http://localhost:8501`

## 🔧 Local Development

### Project Structure

```
AI-For-Flagging-Suspicious-Transactions/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline configuration
├── src/
│   ├── backend/                   # Backend logic and ML models
│   ├── frontend/                  # UI components and integrations
│   ├── database/                  # Database management
│   ├── utils/                     # Utility functions
│   └── data/                      # Data processing modules
├── config/
│   └── app_config.yaml           # Application configuration
├── data/
│   └── sample_transactions.csv   # Sample transaction data
├── main_gui_enhanced.py          # Main application entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── render.yaml                   # Render deployment configuration
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

### Development Commands

```bash
# Run application in development mode
streamlit run main_gui_enhanced.py --server.port 8501

# Run tests
pytest tests/ -v

# Code formatting
black .
isort .

# Linting
flake8 .
pylint src/

# Security scan
bandit -r src/
safety check
```

## 🌐 Deployment

### Deploy to Render (Recommended)

Render provides free hosting for web applications with automatic deployments from GitHub.

#### Step 1: Prepare Your Repository

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

#### Step 2: Set Up Render Service

1. **Create Render Account**:
   - Go to [render.com](https://render.com)
   - Sign up with your GitHub account

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your suspicious transactions flagging repository

3. **Configure Deployment Settings**:
   ```
   Name: anomaly-detector
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: streamlit run main_gui_enhanced.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

4. **Set Environment Variables**:
   ```
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_SERVER_PORT=$PORT
   STREAMLIT_SERVER_ADDRESS=0.0.0.0
   STREAMLIT_THEME_BASE=dark
   STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   ```

#### Step 3: Configure GitHub Secrets for CI/CD

1. **Get Render Deploy Hook**:
   - In Render dashboard, go to your service
   - Navigate to "Settings" → "Deploy Hook"
   - Copy the deploy hook URL

2. **Add GitHub Secret**:
   - Go to your GitHub repository
   - Click "Settings" → "Secrets and variables" → "Actions"
   - Click "New repository secret"
   - Name: `RENDER_DEPLOY_HOOK`
   - Value: Your Render deploy hook URL

#### Step 4: Configure Automatic Deployments

1. **Enable Auto-Deploy**:
   - In Render, go to "Settings" → "Auto-Deploy"
   - Enable "Auto-Deploy" for the main branch

2. **Verify Deployment**:
   - Push changes to main branch
   - Watch the deployment in Render dashboard
   - Access your live application at the provided URL

### Alternative Deployment Options

#### Docker Deployment

1. **Build Docker Image**:
   ```bash
   docker build -t anomaly-detector .
   ```

2. **Run Container**:
   ```bash
   docker run -p 8501:8501 anomaly-detector
   ```

#### Heroku Deployment

1. **Install Heroku CLI**
2. **Create Heroku App**:
   ```bash
   heroku create your-app-name
   ```
3. **Deploy**:
   ```bash
   git push heroku main
   ```

## 🔄 CI/CD Pipeline

Our GitHub Actions pipeline includes:

### Pipeline Stages

1. **Code Quality & Linting**:
   - Black code formatting check
   - isort import sorting
   - flake8 style checking
   - pylint code analysis
   - bandit security scanning
   - safety dependency checking

2. **Build & Test**:
   - Multi-Python version testing (3.9, 3.10, 3.11)
   - Unit tests with pytest
   - Code coverage reporting
   - Application startup testing

3. **Security Scanning**:
   - Comprehensive security analysis
   - Dependency vulnerability checking
   - Security report generation

4. **Docker Build**:
   - Container image building
   - Container testing

5. **Deployment**:
   - Automatic deployment to Render
   - Post-deployment health checks
   - Integration testing

6. **Notification**:
   - Deployment status reporting
   - Pipeline summary

### Triggering Deployments

- **Automatic**: Push to `main` branch triggers deployment
- **Manual**: Use GitHub Actions "Run workflow" button
- **Pull Requests**: Run tests but skip deployment

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Application Configuration
APP_NAME="AI For Flagging Suspicious Transactions"
APP_VERSION="1.0.0"
ENVIRONMENT="development"
LOG_LEVEL="INFO"

# Streamlit Configuration
STREAMLIT_SERVER_HEADLESS=false
STREAMLIT_SERVER_PORT=8501
STREAMLIT_THEME_BASE="dark"

# Database Configuration (if using)
DATABASE_URL="your_database_url"
SUPABASE_URL="your_supabase_url"
SUPABASE_KEY="your_supabase_key"

# AI/ML Configuration
OLLAMA_HOST="localhost"
OLLAMA_PORT="11434"

# Security
SECRET_KEY="your_secret_key"
```

### Application Configuration

Edit `config/app_config.yaml`:

```yaml
app:
  name: "AI For Flagging Suspicious Transactions"
  version: "1.0.0"
  debug: false

anomaly_detection:
  threshold_high: 1000000  # 10 lakh INR
  threshold_medium: 500000  # 5 lakh INR
  risk_score_high: 0.9
  risk_score_medium: 0.6
  risk_score_low: 0.2

ui:
  theme: "dark"
  page_title: "AI For Flagging Suspicious Transactions"
  page_icon: "🔍"
```

## 📖 Usage

### File Upload

1. **Supported Formats**: CSV, Excel (.xlsx, .xls)
2. **Required Columns**:
   - `Transaction_ID`: Unique identifier
   - `Transaction_Amount`: Amount in INR
   - `Transaction_Type`: Type of transaction
   - `Location`: Transaction location
   - `Timestamp`: Transaction timestamp

### Sample Data Format

```csv
Transaction_ID,Transaction_Amount,Transaction_Type,Location,Timestamp
TXN001,1500000,Transfer,Mumbai,2025-09-01 10:30:00
TXN002,50000,Purchase,Delhi,2025-09-01 11:15:00
TXN003,2000000,Withdrawal,Bangalore,2025-09-01 12:00:00
```

### Risk Assessment

- **High Risk** (Score: 0.9): Transactions > ₹10 lakh
- **Medium Risk** (Score: 0.6): Transactions > ₹5 lakh
- **Low Risk** (Score: 0.2): Transactions < ₹5 lakh

## 🔧 API Documentation

### Health Check Endpoint

```
GET /_stcore/health
```

Returns application health status.

### File Upload API

The application provides a web interface for file uploads. Programmatic access can be implemented using Streamlit's session state.

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Ensure security best practices

## 🔒 Security

- All dependencies are regularly scanned for vulnerabilities
- Input validation on all file uploads
- Secure environment variable handling
- Regular security audits with bandit and safety

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/AI-For-Flagging-Suspicious-Transactions/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/AI-For-Flagging-Suspicious-Transactions/discussions)
- **Email**: your-email@example.com

## 🎯 Roadmap

- [ ] Enhanced ML models for anomaly detection
- [ ] Real-time data streaming support
- [ ] Advanced user authentication
- [ ] Multi-currency support
- [ ] Mobile application
- [ ] Advanced analytics dashboard
- [ ] Integration with banking APIs

## 🏆 Acknowledgments

- Streamlit team for the amazing framework
- Render for free hosting
- GitHub Actions for CI/CD
- Open source community for various libraries

---

**Made with ❤️ for financial security and transaction monitoring**
