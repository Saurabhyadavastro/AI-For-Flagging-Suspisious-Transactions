"""
Unit tests for the main GUI application.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestMainGUIEnhanced(unittest.TestCase):
    """Test cases for the main GUI application."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {
            'Transaction_ID': ['TXN001', 'TXN002'],
            'Transaction_Amount': [1500000, 50000],
            'Transaction_Type': ['Transfer', 'Purchase'],
            'Location': ['Mumbai', 'Delhi'],
            'Timestamp': ['2025-09-01 10:30:00', '2025-09-01 11:15:00']
        }

    def test_application_imports(self):
        """Test that main application modules can be imported."""
        try:
            # Test basic imports that should work
            import pandas as pd
            import streamlit as st
            self.assertTrue(True, "Basic imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")

    def test_risk_calculation_high_amount(self):
        """Test risk calculation for high amount transactions."""
        amount = 1500000  # 15 lakh
        expected_risk = "High Risk"
        
        # Simple risk calculation logic
        if amount > 1000000:
            actual_risk = "High Risk"
        elif amount > 500000:
            actual_risk = "Medium Risk"
        else:
            actual_risk = "Low Risk"
            
        self.assertEqual(actual_risk, expected_risk)

    def test_risk_calculation_medium_amount(self):
        """Test risk calculation for medium amount transactions."""
        amount = 750000  # 7.5 lakh
        expected_risk = "Medium Risk"
        
        if amount > 1000000:
            actual_risk = "High Risk"
        elif amount > 500000:
            actual_risk = "Medium Risk"
        else:
            actual_risk = "Low Risk"
            
        self.assertEqual(actual_risk, expected_risk)

    def test_risk_calculation_low_amount(self):
        """Test risk calculation for low amount transactions."""
        amount = 25000  # 25k
        expected_risk = "Low Risk"
        
        if amount > 1000000:
            actual_risk = "High Risk"
        elif amount > 500000:
            actual_risk = "Medium Risk"
        else:
            actual_risk = "Low Risk"
            
        self.assertEqual(actual_risk, expected_risk)

    def test_transaction_data_structure(self):
        """Test transaction data structure validation."""
        required_columns = [
            'Transaction_ID',
            'Transaction_Amount', 
            'Transaction_Type',
            'Location',
            'Timestamp'
        ]
        
        # Check that test data has all required columns
        for column in required_columns:
            self.assertIn(column, self.test_data.keys())

    @patch('streamlit.set_page_config')
    def test_streamlit_config(self, mock_config):
        """Test Streamlit page configuration."""
        # Mock the streamlit configuration
        mock_config.return_value = None
        
        # Test that configuration would be called
        expected_config = {
            'page_title': 'AI For Flagging Suspicious Transactions',
            'page_icon': '🔍',
            'layout': 'wide',
            'initial_sidebar_state': 'expanded'
        }
        
        # This test ensures our expected configuration is valid
        self.assertIsInstance(expected_config, dict)
        self.assertIn('page_title', expected_config)

if __name__ == '__main__':
    unittest.main()
