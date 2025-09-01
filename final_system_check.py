#!/usr/bin/env python3
"""
Comprehensive System Check for AI For Flagging Suspicious Transactions
This script performs a final verification of all system components.
"""

import os
import sys
import importlib
import subprocess
import platform
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def print_status(item, status, details=""):
    """Print status with emoji indicators."""
    emoji = "✅" if status else "❌"
    print(f"{emoji} {item}: {'PASS' if status else 'FAIL'}")
    if details:
        print(f"   {details}")

def check_python_version():
    """Check Python version compatibility."""
    print_header("Python Environment Check")
    
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"🖥️ Platform: {platform.platform()}")
    print(f"📁 Working Directory: {os.getcwd()}")
    
    # Check if Python version is supported
    supported = version.major == 3 and version.minor >= 9
    print_status("Python Version (≥3.9)", supported, 
                f"Current: {version.major}.{version.minor}")
    
    return supported

def check_file_structure():
    """Check if all required files exist."""
    print_header("File Structure Check")
    
    required_files = [
        "main_gui_enhanced.py",
        "requirements.txt",
        "Dockerfile",
        "render.yaml",
        ".env.example",
        "README.md",
        ".github/workflows/ci.yml",
        ".pylintrc"
    ]
    
    required_dirs = [
        "src",
        "src/frontend",
        "src/backend", 
        "src/utils",
        "config",
        "data",
        "tests"
    ]
    
    all_files_exist = True
    
    for file_path in required_files:
        exists = os.path.exists(file_path)
        print_status(f"File: {file_path}", exists)
        if not exists:
            all_files_exist = False
    
    for dir_path in required_dirs:
        exists = os.path.isdir(dir_path)
        print_status(f"Directory: {dir_path}", exists)
        if not exists:
            all_files_exist = False
    
    return all_files_exist

def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("Dependencies Check")
    
    critical_deps = [
        "streamlit",
        "pandas", 
        "numpy",
        "plotly",
        "openpyxl",
        "PIL",  # Pillow
        "pyarrow"
    ]
    
    optional_deps = [
        "pytest",
        "black",
        "flake8",
        "pylint"
    ]
    
    all_critical_installed = True
    
    for dep in critical_deps:
        try:
            importlib.import_module(dep)
            print_status(f"Critical: {dep}", True)
        except ImportError:
            print_status(f"Critical: {dep}", False, "Required for core functionality")
            all_critical_installed = False
    
    for dep in optional_deps:
        try:
            importlib.import_module(dep)
            print_status(f"Optional: {dep}", True)
        except ImportError:
            print_status(f"Optional: {dep}", False, "Used for development/testing")
    
    return all_critical_installed

def check_application_startup():
    """Test if the main application can be imported."""
    print_header("Application Startup Check")
    
    try:
        # Change to project directory
        original_path = sys.path.copy()
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Test basic imports
        import streamlit as st
        print_status("Streamlit Import", True, f"Version: {st.__version__}")
        
        import pandas as pd
        print_status("Pandas Import", True, f"Version: {pd.__version__}")
        
        # Test main application import
        try:
            import main_gui_enhanced
            print_status("Main Application Import", True)
        except Exception as e:
            print_status("Main Application Import", False, str(e))
            return False
            
        # Test source modules
        try:
            from src.frontend.ollama_integration import OllamaChat
            print_status("AI Integration Module", True)
        except Exception as e:
            print_status("AI Integration Module", False, str(e))
        
        # Restore original path
        sys.path = original_path
        return True
        
    except Exception as e:
        print_status("Application Startup", False, str(e))
        return False

def check_configuration():
    """Check configuration files."""
    print_header("Configuration Check")
    
    config_files = {
        ".env.example": "Environment variables template",
        "config/app_config.yaml": "Application configuration",
        "render.yaml": "Render deployment configuration"
    }
    
    all_configs_valid = True
    
    for file_path, description in config_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if len(content.strip()) > 0:
                        print_status(f"{description}", True, f"File: {file_path}")
                    else:
                        print_status(f"{description}", False, "File is empty")
                        all_configs_valid = False
            except Exception as e:
                print_status(f"{description}", False, f"Error reading: {e}")
                all_configs_valid = False
        else:
            print_status(f"{description}", False, f"File not found: {file_path}")
            all_configs_valid = False
    
    return all_configs_valid

def check_data_files():
    """Check sample data files."""
    print_header("Data Files Check")
    
    data_files = [
        "data/sample_transactions.csv",
        "sample_transactions.csv"
    ]
    
    data_available = False
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                required_columns = [
                    'Transaction_ID',
                    'Transaction_Amount',
                    'Transaction_Type', 
                    'Location',
                    'Timestamp'
                ]
                
                has_required_columns = all(col in df.columns for col in required_columns)
                print_status(f"Sample Data: {file_path}", has_required_columns,
                           f"Rows: {len(df)}, Columns: {list(df.columns)}")
                
                if has_required_columns:
                    data_available = True
                    
            except Exception as e:
                print_status(f"Sample Data: {file_path}", False, f"Error: {e}")
        else:
            print_status(f"Sample Data: {file_path}", False, "File not found")
    
    return data_available

def check_cicd_pipeline():
    """Check CI/CD pipeline configuration."""
    print_header("CI/CD Pipeline Check")
    
    pipeline_file = ".github/workflows/ci.yml"
    
    if os.path.exists(pipeline_file):
        try:
            with open(pipeline_file, 'r') as f:
                content = f.read()
                
            # Check for key pipeline components
            required_jobs = [
                "lint",
                "build-and-test", 
                "security-scan",
                "deploy"
            ]
            
            pipeline_valid = True
            for job in required_jobs:
                if job in content:
                    print_status(f"Pipeline Job: {job}", True)
                else:
                    print_status(f"Pipeline Job: {job}", False)
                    pipeline_valid = False
            
            # Check for secrets usage
            if "RENDER_DEPLOY_HOOK" in content:
                print_status("Render Deploy Hook Configuration", True)
            else:
                print_status("Render Deploy Hook Configuration", False,
                           "Add RENDER_DEPLOY_HOOK to GitHub secrets")
                
            return pipeline_valid
            
        except Exception as e:
            print_status("CI/CD Pipeline", False, f"Error reading: {e}")
            return False
    else:
        print_status("CI/CD Pipeline", False, "Pipeline file not found")
        return False

def generate_deployment_checklist():
    """Generate deployment checklist."""
    print_header("Deployment Checklist")
    
    checklist = [
        "✅ Code pushed to GitHub repository",
        "⚠️ GitHub repository secrets configured (RENDER_DEPLOY_HOOK)",
        "⚠️ Render account created and connected to GitHub",
        "⚠️ Render web service configured with correct build/start commands",
        "⚠️ Environment variables set in Render dashboard",
        "⚠️ Auto-deploy enabled for main branch",
        "⚠️ Domain configured (optional)",
        "⚠️ SSL certificate configured (automatic with Render)"
    ]
    
    print("\n📋 Pre-Deployment Checklist:")
    for item in checklist:
        print(f"   {item}")
    
    print("\n🚀 Deployment Steps:")
    print("   1. Push code to main branch")
    print("   2. GitHub Actions will run CI/CD pipeline")
    print("   3. If all tests pass, deployment will trigger automatically")
    print("   4. Check Render dashboard for deployment status")
    print("   5. Access your live application at the Render URL")

def main():
    """Run comprehensive system check."""
    print("🔍 COMPREHENSIVE SYSTEM CHECK")
    print("AI For Flagging Suspicious Transactions")
    print(f"Timestamp: {__import__('datetime').datetime.now()}")
    
    # Run all checks
    checks = [
        ("Python Environment", check_python_version),
        ("File Structure", check_file_structure), 
        ("Dependencies", check_dependencies),
        ("Application Startup", check_application_startup),
        ("Configuration", check_configuration),
        ("Data Files", check_data_files),
        ("CI/CD Pipeline", check_cicd_pipeline)
    ]
    
    results = {}
    
    for check_name, check_function in checks:
        try:
            results[check_name] = check_function()
        except Exception as e:
            print_status(f"{check_name} Check", False, f"Exception: {e}")
            results[check_name] = False
    
    # Summary
    print_header("FINAL SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        print_status(check_name, result)
    
    print(f"\n📊 Overall Score: {passed}/{total} ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("🚀 System is ready for deployment!")
        generate_deployment_checklist()
    elif passed >= total * 0.8:
        print("\n⚠️ MOSTLY READY")
        print("🔧 Minor issues detected, but system should work")
        generate_deployment_checklist()
    else:
        print("\n❌ ISSUES DETECTED")
        print("🛠️ Please fix the failed checks before deployment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
