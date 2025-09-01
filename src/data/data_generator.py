"""
Advanced data generation module for creating realistic Indian financial transaction datasets.
This module generates synthetic data that closely mimics real-world Indian financial transactions
for training and testing the AI fraud detection system.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta, time
import random
import os
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.database.database_manager import DatabaseManager

# Initialize logger and faker with Indian locale
logger = get_logger(__name__)
fake = Faker(['en_IN', 'hi_IN'])  # Indian English and Hindi locales


class IndianTransactionDataGenerator:
    """
    Generates realistic synthetic transaction data for the Indian financial ecosystem.
    
    This class creates comprehensive datasets that include:
    - Realistic Indian customer information
    - Indian payment methods (UPI, IMPS, NEFT, etc.)
    - Indian cities and locations
    - Realistic transaction patterns
    - Fraudulent transaction patterns
    """
    
    def __init__(self):
        """Initialize the data generator with Indian financial system parameters."""
        self.fake = fake
        
        # Indian payment methods with their usage probability
        self.payment_methods = {
            'UPI': 0.45,        # Most popular in India
            'IMPS': 0.15,       # Immediate Payment Service
            'NEFT': 0.12,       # National Electronic Funds Transfer
            'Net Banking': 0.10,
            'Card': 0.08,
            'RTGS': 0.05,       # Real Time Gross Settlement (high value)
            'Wallet': 0.03,
            'Cash': 0.02
        }
        
        # Major Indian cities with their transaction volume weights
        self.indian_cities = {
            'Mumbai': 0.18, 'Delhi': 0.15, 'Bangalore': 0.12, 'Hyderabad': 0.08,
            'Chennai': 0.07, 'Kolkata': 0.06, 'Pune': 0.05, 'Ahmedabad': 0.04,
            'Surat': 0.03, 'Jaipur': 0.03, 'Lucknow': 0.03, 'Kanpur': 0.02,
            'Nagpur': 0.02, 'Indore': 0.02, 'Thane': 0.02, 'Bhopal': 0.02,
            'Visakhapatnam': 0.02, 'Pimpri-Chinchwad': 0.02, 'Patna': 0.02
        }
        
        # Merchant categories common in India
        self.merchant_categories = {
            'Grocery': 0.20, 'Restaurant': 0.15, 'Fuel': 0.12, 'Retail': 0.10,
            'E-commerce': 0.08, 'Healthcare': 0.06, 'Education': 0.05,
            'Entertainment': 0.04, 'Travel': 0.04, 'Utilities': 0.04,
            'Insurance': 0.03, 'Investment': 0.03, 'Government': 0.02,
            'Telecom': 0.02, 'Others': 0.02
        }
        
        # Time patterns for transactions (hourly distribution)
        self.hourly_patterns = {
            0: 0.01, 1: 0.005, 2: 0.002, 3: 0.001, 4: 0.001, 5: 0.002,
            6: 0.01, 7: 0.03, 8: 0.05, 9: 0.07, 10: 0.08, 11: 0.09,
            12: 0.1, 13: 0.08, 14: 0.09, 15: 0.08, 16: 0.07, 17: 0.06,
            18: 0.05, 19: 0.04, 20: 0.03, 21: 0.025, 22: 0.02, 23: 0.015
        }
        
        logger.info("Indian Transaction Data Generator initialized")
    
    def _weighted_choice(self, choices: Dict[str, float]) -> str:
        """Select an item based on weighted probabilities."""
        items = list(choices.keys())
        weights = list(choices.values())
        return np.random.choice(items, p=weights)
    
    def _generate_indian_customer_id(self) -> str:
        """Generate a realistic Indian customer ID."""
        bank_codes = ['SBI', 'HDFC', 'ICICI', 'AXIS', 'PNB', 'BOB', 'CANARA', 'IOB']
        bank = random.choice(bank_codes)
        customer_num = f"{random.randint(100000000, 999999999)}"
        return f"{bank}{customer_num}"
    
    def _generate_transaction_id(self, date: datetime) -> str:
        """Generate a realistic transaction ID with date and random components."""
        date_str = date.strftime("%Y%m%d")
        random_part = f"{random.randint(100000, 999999)}"
        bank_code = random.choice(['TXN', 'PAY', 'TRF', 'UPI'])
        return f"{bank_code}{date_str}{random_part}"
    
    def _generate_account_number(self) -> str:
        """Generate a realistic Indian bank account number."""
        # Indian bank account numbers are typically 9-18 digits
        length = random.choice([10, 11, 12, 13, 14, 15, 16])
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])
    
    def _determine_transaction_amount(self, payment_method: str, is_suspicious: bool) -> float:
        """
        Generate realistic transaction amounts based on payment method and suspicion.
        
        Args:
            payment_method: The payment method being used
            is_suspicious: Whether this should be a suspicious transaction
            
        Returns:
            Transaction amount in INR
        """
        if is_suspicious:
            # Suspicious transactions tend to be either very high or structured
            if random.random() < 0.4:
                # Very high amounts
                return round(random.uniform(500000, 5000000), 2)
            elif random.random() < 0.3:
                # Structuring (just below reporting thresholds)
                thresholds = [9999, 49999, 99999, 199999]
                return round(random.choice(thresholds) - random.uniform(0, 100), 2)
            else:
                # Round amounts
                base_amounts = [10000, 25000, 50000, 100000, 250000, 500000]
                return float(random.choice(base_amounts))
        
        # Normal transaction amounts based on payment method
        if payment_method == 'UPI':
            # UPI typically for smaller amounts
            return round(np.random.lognormal(mean=6.5, sigma=1.2), 2)
        elif payment_method == 'RTGS':
            # RTGS for high-value transactions (minimum 2 lakhs)
            return round(random.uniform(200000, 10000000), 2)
        elif payment_method == 'NEFT':
            # NEFT for medium to high amounts
            return round(np.random.lognormal(mean=8.5, sigma=1.5), 2)
        elif payment_method in ['IMPS', 'Net Banking']:
            # IMPS and Net Banking for varied amounts
            return round(np.random.lognormal(mean=7.8, sigma=1.3), 2)
        elif payment_method == 'Card':
            # Card transactions typically smaller
            return round(np.random.lognormal(mean=6.8, sigma=1.1), 2)
        else:
            # Default for other methods
            return round(np.random.lognormal(mean=7.0, sigma=1.2), 2)
    
    def _determine_suspicious_patterns(self) -> Dict[str, Any]:
        """
        Determine what makes a transaction suspicious with realistic patterns.
        
        Returns:
            Dictionary containing suspicious patterns and reasons
        """
        suspicious_patterns = []
        reasons = []
        
        # Pattern 1: Unusual timing (late night/early morning)
        if random.random() < 0.3:
            suspicious_patterns.append('unusual_timing')
            reasons.append('Transaction during unusual hours (12 AM - 6 AM)')
        
        # Pattern 2: High velocity (multiple transactions in short time)
        if random.random() < 0.25:
            suspicious_patterns.append('high_velocity')
            reasons.append('Multiple transactions in rapid succession')
        
        # Pattern 3: Round amounts
        if random.random() < 0.2:
            suspicious_patterns.append('round_amount')
            reasons.append('Suspiciously round transaction amount')
        
        # Pattern 4: Cross-border or unusual location
        if random.random() < 0.15:
            suspicious_patterns.append('location_anomaly')
            reasons.append('Transaction from unusual location or IP')
        
        # Pattern 5: Structuring (amounts just below thresholds)
        if random.random() < 0.1:
            suspicious_patterns.append('structuring')
            reasons.append('Amount structured to avoid reporting thresholds')
        
        return {
            'patterns': suspicious_patterns,
            'reasons': reasons,
            'risk_level': 'high' if len(suspicious_patterns) >= 2 else 'medium'
        }
    
    def generate_single_transaction(self, date: datetime = None, 
                                  force_suspicious: bool = False) -> Dict[str, Any]:
        """
        Generate a single realistic transaction record.
        
        Args:
            date: Specific date for the transaction (optional)
            force_suspicious: Force the transaction to be suspicious
            
        Returns:
            Dictionary containing transaction data
        """
        if date is None:
            # Generate random date within last 30 days
            start_date = datetime.now() - timedelta(days=30)
            date = start_date + timedelta(days=random.randint(0, 30))
        
        # Determine if transaction should be suspicious (5% base rate, unless forced)
        is_suspicious = force_suspicious or (random.random() < 0.05)
        
        # Generate transaction time based on realistic patterns
        if is_suspicious and 'unusual_timing' in self._determine_suspicious_patterns().get('patterns', []):
            # Suspicious timing: late night/early morning
            hour = random.choice([0, 1, 2, 3, 4, 5, 23])
        else:
            # Normal timing based on hourly patterns
            hours = list(self.hourly_patterns.keys())
            probabilities = list(self.hourly_patterns.values())
            hour = np.random.choice(hours, p=probabilities)
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        transaction_time = time(hour, minute, second)
        
        # Select payment method
        payment_method = self._weighted_choice(self.payment_methods)
        
        # Generate transaction amount
        amount = self._determine_transaction_amount(payment_method, is_suspicious)
        
        # Generate location
        location = self._weighted_choice(self.indian_cities)
        
        # Generate merchant category
        merchant_category = self._weighted_choice(self.merchant_categories)
        
        # Generate customer and transaction IDs
        customer_id = self._generate_indian_customer_id()
        transaction_id = self._generate_transaction_id(date)
        
        # Generate account numbers
        sender_account = self._generate_account_number()
        receiver_account = self._generate_account_number()
        
        # Generate merchant information
        merchant_id = f"MER{random.randint(100000, 999999)}"
        
        # Determine suspicious details if applicable
        suspicious_details = {}
        suspicious_reason = None
        if is_suspicious:
            suspicious_details = self._determine_suspicious_patterns()
            suspicious_reason = '; '.join(suspicious_details.get('reasons', []))
        
        # Generate additional metadata
        device_id = f"DEV{random.randint(10000000, 99999999)}"
        
        return {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'transaction_amount_inr': amount,
            'transaction_date': date.date(),
            'transaction_time': transaction_time,
            'payment_method': payment_method,
            'location': location,
            'merchant_category': merchant_category,
            'merchant_id': merchant_id,
            'sender_account': sender_account,
            'receiver_account': receiver_account,
            'device_id': device_id,
            'is_suspicious': is_suspicious,
            'suspicious_reason': suspicious_reason,
            'risk_score': random.uniform(0.7, 1.0) if is_suspicious else random.uniform(0.0, 0.3),
            'currency': 'INR',
            'description': f"{merchant_category} transaction via {payment_method}",
            'fees_amount': round(amount * random.uniform(0.001, 0.01), 2) if payment_method != 'UPI' else 0.0,
            'ip_address': self.fake.ipv4(),
            'source_system': 'SYNTHETIC_DATA_GENERATOR'
        }
    
    def generate_bulk_transactions(self, num_transactions: int, 
                                 start_date: datetime = None,
                                 end_date: datetime = None,
                                 suspicious_rate: float = 0.05) -> List[Dict[str, Any]]:
        """
        Generate multiple transactions with realistic patterns.
        
        Args:
            num_transactions: Number of transactions to generate
            start_date: Start date for transaction range
            end_date: End date for transaction range  
            suspicious_rate: Percentage of transactions that should be suspicious
            
        Returns:
            List of transaction dictionaries
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        transactions = []
        
        # Calculate date range
        date_range = (end_date - start_date).days
        
        # Generate transactions
        for i in range(num_transactions):
            # Random date within range
            random_days = random.randint(0, date_range)
            transaction_date = start_date + timedelta(days=random_days)
            
            # Determine if this transaction should be suspicious
            force_suspicious = random.random() < suspicious_rate
            
            transaction = self.generate_single_transaction(
                date=transaction_date,
                force_suspicious=force_suspicious
            )
            
            transactions.append(transaction)
            
            # Log progress for large datasets
            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1}/{num_transactions} transactions")
        
        logger.info(f"Generated {num_transactions} transactions with {suspicious_rate*100}% suspicious rate")
        return transactions
    
    def generate_csv_data(self, file_path: str, num_rows: int) -> Dict[str, Any]:
        """
        Generate transaction data and save as CSV file.
        
        Args:
            file_path: Path where CSV file should be saved
            num_rows: Number of transaction rows to generate
            
        Returns:
            Dictionary containing generation statistics and file info
        """
        try:
            logger.info(f"Starting CSV generation: {num_rows} rows to {file_path}")
            
            # Generate transactions
            transactions = self.generate_bulk_transactions(num_rows)
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            # Calculate statistics
            stats = {
                'total_transactions': len(df),
                'suspicious_transactions': df['is_suspicious'].sum(),
                'suspicious_rate': (df['is_suspicious'].sum() / len(df)) * 100,
                'date_range': {
                    'start': df['transaction_date'].min().isoformat(),
                    'end': df['transaction_date'].max().isoformat()
                },
                'amount_stats': {
                    'min': float(df['transaction_amount_inr'].min()),
                    'max': float(df['transaction_amount_inr'].max()),
                    'mean': float(df['transaction_amount_inr'].mean()),
                    'total': float(df['transaction_amount_inr'].sum())
                },
                'payment_methods': df['payment_method'].value_counts().to_dict(),
                'locations': df['location'].value_counts().head(10).to_dict(),
                'file_path': file_path,
                'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
            }
            
            logger.info(f"CSV generation completed successfully: {file_path}")
            return {'success': True, 'stats': stats}
            
        except Exception as e:
            error_msg = f"Error generating CSV data: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def generate_excel_data(self, file_path: str, num_rows: int) -> Dict[str, Any]:
        """
        Generate transaction data and save as Excel file with multiple sheets.
        
        Args:
            file_path: Path where Excel file should be saved
            num_rows: Number of transaction rows to generate
            
        Returns:
            Dictionary containing generation statistics and file info
        """
        try:
            logger.info(f"Starting Excel generation: {num_rows} rows to {file_path}")
            
            # Generate transactions
            transactions = self.generate_bulk_transactions(num_rows)
            df = pd.DataFrame(transactions)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Main transactions sheet
                df.to_excel(writer, sheet_name='Transactions', index=False)
                
                # Summary statistics sheet
                summary_stats = {
                    'Metric': [
                        'Total Transactions',
                        'Suspicious Transactions',
                        'Suspicious Rate (%)',
                        'Total Amount (INR)',
                        'Average Amount (INR)',
                        'Date Range Start',
                        'Date Range End'
                    ],
                    'Value': [
                        len(df),
                        df['is_suspicious'].sum(),
                        round((df['is_suspicious'].sum() / len(df)) * 100, 2),
                        round(df['transaction_amount_inr'].sum(), 2),
                        round(df['transaction_amount_inr'].mean(), 2),
                        df['transaction_date'].min(),
                        df['transaction_date'].max()
                    ]
                }
                pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Summary', index=False)
                
                # Payment methods breakdown
                payment_breakdown = df['payment_method'].value_counts().reset_index()
                payment_breakdown.columns = ['Payment Method', 'Count']
                payment_breakdown['Percentage'] = round((payment_breakdown['Count'] / len(df)) * 100, 2)
                payment_breakdown.to_excel(writer, sheet_name='Payment Methods', index=False)
                
                # Location breakdown
                location_breakdown = df['location'].value_counts().head(20).reset_index()
                location_breakdown.columns = ['Location', 'Count']
                location_breakdown['Percentage'] = round((location_breakdown['Count'] / len(df)) * 100, 2)
                location_breakdown.to_excel(writer, sheet_name='Locations', index=False)
                
                # Suspicious transactions only
                suspicious_df = df[df['is_suspicious'] == True]
                if len(suspicious_df) > 0:
                    suspicious_df.to_excel(writer, sheet_name='Suspicious Only', index=False)
            
            # Calculate file statistics
            stats = {
                'total_transactions': len(df),
                'suspicious_transactions': df['is_suspicious'].sum(),
                'suspicious_rate': (df['is_suspicious'].sum() / len(df)) * 100,
                'file_path': file_path,
                'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                'sheets_created': ['Transactions', 'Summary', 'Payment Methods', 'Locations', 'Suspicious Only']
            }
            
            logger.info(f"Excel generation completed successfully: {file_path}")
            return {'success': True, 'stats': stats}
            
        except Exception as e:
            error_msg = f"Error generating Excel data: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def load_data_to_supabase(self, file_path: str, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Load generated data from CSV/Excel file to Supabase database.
        
        Args:
            file_path: Path to the data file
            batch_size: Number of records to insert per batch
            
        Returns:
            Dictionary containing upload statistics
        """
        try:
            logger.info(f"Starting data upload to Supabase from: {file_path}")
            
            # Initialize database manager
            db_manager = DatabaseManager()
            
            # Read data based on file extension
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, sheet_name='Transactions')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Convert DataFrame to list of dictionaries
            transactions = df.to_dict('records')
            
            # Convert date and time columns to proper formats
            for transaction in transactions:
                if 'transaction_date' in transaction:
                    if isinstance(transaction['transaction_date'], str):
                        transaction['transaction_date'] = datetime.strptime(
                            transaction['transaction_date'], '%Y-%m-%d'
                        ).date()
                
                if 'transaction_time' in transaction:
                    if isinstance(transaction['transaction_time'], str):
                        transaction['transaction_time'] = datetime.strptime(
                            transaction['transaction_time'], '%H:%M:%S'
                        ).time()
            
            # Upload to Supabase in batches
            upload_result = db_manager.client.insert_bulk_transactions(
                transactions, batch_size=batch_size
            )
            
            if upload_result['success']:
                logger.info(f"Successfully uploaded {upload_result['total_inserted']} transactions to Supabase")
                return {
                    'success': True,
                    'upload_stats': upload_result,
                    'file_processed': file_path
                }
            else:
                return {
                    'success': False,
                    'error': upload_result.get('error', 'Unknown upload error'),
                    'upload_stats': upload_result
                }
                
        except Exception as e:
            error_msg = f"Error loading data to Supabase: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def generate_and_upload_data(self, num_transactions: int, 
                               file_format: str = 'csv',
                               upload_to_db: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: generate data, save to file, and optionally upload to database.
        
        Args:
            num_transactions: Number of transactions to generate
            file_format: 'csv' or 'excel'
            upload_to_db: Whether to upload data to Supabase
            
        Returns:
            Dictionary containing complete pipeline results
        """
        try:
            # Prepare file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = "data/raw"
            os.makedirs(data_dir, exist_ok=True)
            
            if file_format.lower() == 'excel':
                file_path = f"{data_dir}/synthetic_transactions_{timestamp}.xlsx"
                generation_result = self.generate_excel_data(file_path, num_transactions)
            else:
                file_path = f"{data_dir}/synthetic_transactions_{timestamp}.csv"
                generation_result = self.generate_csv_data(file_path, num_transactions)
            
            if not generation_result['success']:
                return generation_result
            
            results = {
                'generation': generation_result,
                'upload': None
            }
            
            # Upload to database if requested
            if upload_to_db:
                upload_result = self.load_data_to_supabase(file_path)
                results['upload'] = upload_result
            
            return {
                'success': True,
                'results': results,
                'file_path': file_path
            }
            
        except Exception as e:
            error_msg = f"Error in data generation pipeline: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}


# Convenience functions for easy usage
def generate_csv_data(file_path: str, num_rows: int) -> Dict[str, Any]:
    """
    Convenience function to generate CSV data.
    
    Args:
        file_path: Path where CSV file should be saved
        num_rows: Number of transaction rows to generate
        
    Returns:
        Dictionary containing generation results
    """
    generator = IndianTransactionDataGenerator()
    return generator.generate_csv_data(file_path, num_rows)


def generate_excel_data(file_path: str, num_rows: int) -> Dict[str, Any]:
    """
    Convenience function to generate Excel data.
    
    Args:
        file_path: Path where Excel file should be saved
        num_rows: Number of transaction rows to generate
        
    Returns:
        Dictionary containing generation results
    """
    generator = IndianTransactionDataGenerator()
    return generator.generate_excel_data(file_path, num_rows)


def load_data_to_supabase(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to load data to Supabase.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Dictionary containing upload results
    """
    generator = IndianTransactionDataGenerator()
    return generator.load_data_to_supabase(file_path)
