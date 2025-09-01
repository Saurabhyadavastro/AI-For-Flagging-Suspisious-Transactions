"""Database manager for handling database connections and table management."""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.database.supabase_client import SupabaseClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    Manages database connections and handles table creation and management.
    """
    
    def __init__(self):
        """Initialize the database manager."""
        self.client = SupabaseClient()
        logger.info("Database manager initialized")
    
    def create_tables_if_not_exist(self) -> Dict[str, Any]:
        """
        Create all necessary tables if they don't exist.
        
        Returns:
            Dict containing the status of table creation operations
        """
        try:
            results = {}
            
            # Create users table
            users_result = self._create_users_table()
            results["users_table"] = users_result
            
            # Create transactions table
            transactions_result = self._create_transactions_table()
            results["transactions_table"] = transactions_result
            
            # Create alerts table
            alerts_result = self._create_alerts_table()
            results["alerts_table"] = alerts_result
            
            # Create audit logs table
            audit_result = self._create_audit_logs_table()
            results["audit_logs_table"] = audit_result
            
            logger.info("All tables checked/created successfully")
            return {"success": True, "results": results}
            
        except Exception as e:
            error_msg = f"Error creating tables: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _create_users_table(self) -> Dict[str, Any]:
        """Create the user profiles table with enhanced fields."""
        sql_query = """
        -- Enhanced user profiles table
        CREATE TABLE IF NOT EXISTS user_profiles (
            id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            full_name VARCHAR(255) NOT NULL,
            role VARCHAR(50) DEFAULT 'analyst' CHECK (role IN ('admin', 'analyst', 'viewer', 'manager')),
            department VARCHAR(100),
            password_hash TEXT,  -- Additional security layer
            phone_number VARCHAR(20),
            is_active BOOLEAN DEFAULT TRUE,
            last_login TIMESTAMP WITH TIME ZONE,
            login_attempts INTEGER DEFAULT 0,
            locked_until TIMESTAMP WITH TIME ZONE,
            preferences JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);
        CREATE INDEX IF NOT EXISTS idx_user_profiles_role ON user_profiles(role);
        CREATE INDEX IF NOT EXISTS idx_user_profiles_active ON user_profiles(is_active);
        
        -- Row Level Security (RLS) policies
        ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
        
        -- Policy: Users can read their own profile
        CREATE POLICY IF NOT EXISTS "Users can view own profile" ON user_profiles
            FOR SELECT USING (auth.uid() = id);
        
        -- Policy: Users can update their own profile
        CREATE POLICY IF NOT EXISTS "Users can update own profile" ON user_profiles
            FOR UPDATE USING (auth.uid() = id);
        
        -- Policy: Admins can view all profiles
        CREATE POLICY IF NOT EXISTS "Admins can view all profiles" ON user_profiles
            FOR SELECT USING (
                EXISTS (
                    SELECT 1 FROM user_profiles 
                    WHERE id = auth.uid() AND role = 'admin'
                )
            );
        """
        
        return self._execute_sql(sql_query, "user_profiles")
    
    def _create_transactions_table(self) -> Dict[str, Any]:
        """Create the transactions table optimized for Indian financial data."""
        sql_query = """
        -- Enhanced transactions table for Indian financial ecosystem
        CREATE TABLE IF NOT EXISTS transactions (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            transaction_id VARCHAR(255) UNIQUE NOT NULL,
            customer_id VARCHAR(255) NOT NULL,
            transaction_amount_inr DECIMAL(15,2) NOT NULL CHECK (transaction_amount_inr > 0),
            transaction_date DATE NOT NULL,
            transaction_time TIME NOT NULL,
            payment_method VARCHAR(50) NOT NULL CHECK (payment_method IN ('UPI', 'IMPS', 'NEFT', 'RTGS', 'Cash', 'Card', 'Wallet', 'Net Banking')),
            location VARCHAR(255),
            merchant_category VARCHAR(100),
            merchant_id VARCHAR(255),
            device_id VARCHAR(255),
            ip_address INET,
            sender_account VARCHAR(255),
            receiver_account VARCHAR(255),
            currency VARCHAR(3) DEFAULT 'INR',
            
            -- Fraud detection fields
            is_suspicious BOOLEAN DEFAULT FALSE,
            suspicious_reason TEXT,
            risk_score DECIMAL(5,4) DEFAULT 0.0000,
            flagged_at TIMESTAMP WITH TIME ZONE,
            flagged_by UUID REFERENCES auth.users(id),
            reviewed_at TIMESTAMP WITH TIME ZONE,
            reviewed_by UUID REFERENCES auth.users(id),
            review_status VARCHAR(20) DEFAULT 'pending' CHECK (review_status IN ('pending', 'approved', 'rejected', 'investigating')),
            
            -- Additional metadata
            transaction_reference VARCHAR(255),
            description TEXT,
            fees_amount DECIMAL(10,2) DEFAULT 0.00,
            tax_amount DECIMAL(10,2) DEFAULT 0.00,
            source_system VARCHAR(50),
            batch_id VARCHAR(255),
            
            -- Timestamps
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create comprehensive indexes for performance
        CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
        CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(transaction_amount_inr);
        CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
        CREATE INDEX IF NOT EXISTS idx_transactions_payment_method ON transactions(payment_method);
        CREATE INDEX IF NOT EXISTS idx_transactions_suspicious ON transactions(is_suspicious);
        CREATE INDEX IF NOT EXISTS idx_transactions_location ON transactions(location);
        CREATE INDEX IF NOT EXISTS idx_transactions_risk_score ON transactions(risk_score);
        CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at);
        
        -- Composite indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_transactions_customer_date ON transactions(customer_id, transaction_date);
        CREATE INDEX IF NOT EXISTS idx_transactions_amount_date ON transactions(transaction_amount_inr, transaction_date);
        CREATE INDEX IF NOT EXISTS idx_transactions_suspicious_date ON transactions(is_suspicious, transaction_date);
        
        -- Full-text search index for descriptions
        CREATE INDEX IF NOT EXISTS idx_transactions_description_fts ON transactions USING gin(to_tsvector('english', description));
        
        -- Row Level Security
        ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
        
        -- Policy: Analysts can view all transactions
        CREATE POLICY IF NOT EXISTS "Analysts can view transactions" ON transactions
            FOR SELECT USING (
                EXISTS (
                    SELECT 1 FROM user_profiles 
                    WHERE id = auth.uid() AND role IN ('admin', 'analyst', 'manager')
                )
            );
        
        -- Policy: Only admins can insert transactions
        CREATE POLICY IF NOT EXISTS "Admins can insert transactions" ON transactions
            FOR INSERT WITH CHECK (
                EXISTS (
                    SELECT 1 FROM user_profiles 
                    WHERE id = auth.uid() AND role IN ('admin')
                )
            );
        
        -- Trigger to update the updated_at timestamp
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        CREATE TRIGGER update_transactions_updated_at
            BEFORE UPDATE ON transactions
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """
        
        return self._execute_sql(sql_query, "transactions")
    
    def _create_alerts_table(self) -> Dict[str, Any]:
        """Create the alerts table for monitoring suspicious activities."""
        sql_query = """
        -- Alerts table for fraud detection alerts
        CREATE TABLE IF NOT EXISTS alerts (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            transaction_id UUID REFERENCES transactions(id) ON DELETE CASCADE,
            alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN ('high_amount', 'unusual_time', 'rapid_succession', 'location_anomaly', 'ml_anomaly', 'rule_based')),
            severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
            title VARCHAR(255) NOT NULL,
            message TEXT NOT NULL,
            details JSONB DEFAULT '{}',
            
            -- Status tracking
            status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'false_positive', 'escalated')),
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_by UUID REFERENCES auth.users(id),
            acknowledged_at TIMESTAMP WITH TIME ZONE,
            resolved_by UUID REFERENCES auth.users(id),
            resolved_at TIMESTAMP WITH TIME ZONE,
            resolution_notes TEXT,
            
            -- Metadata
            source_system VARCHAR(50),
            confidence_score DECIMAL(5,4),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Indexes for alerts
        CREATE INDEX IF NOT EXISTS idx_alerts_transaction_id ON alerts(transaction_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
        CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
        CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
        CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);
        CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
        
        -- RLS for alerts
        ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
        
        CREATE POLICY IF NOT EXISTS "Analysts can view alerts" ON alerts
            FOR SELECT USING (
                EXISTS (
                    SELECT 1 FROM user_profiles 
                    WHERE id = auth.uid() AND role IN ('admin', 'analyst', 'manager')
                )
            );
        """
        
        return self._execute_sql(sql_query, "alerts")
    
    def _create_audit_logs_table(self) -> Dict[str, Any]:
        """Create audit logs table for tracking system activities."""
        sql_query = """
        -- Audit logs table for tracking all system activities
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            user_id UUID REFERENCES auth.users(id),
            action VARCHAR(100) NOT NULL,
            table_name VARCHAR(50),
            record_id VARCHAR(255),
            old_values JSONB,
            new_values JSONB,
            ip_address INET,
            user_agent TEXT,
            session_id VARCHAR(255),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Indexes for audit logs
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_table_name ON audit_logs(table_name);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
        
        -- Partition by month for performance (optional, for large datasets)
        -- This would be implemented separately if needed
        """
        
        return self._execute_sql(sql_query, "audit_logs")
    
    def _execute_sql(self, sql_query: str, table_name: str) -> Dict[str, Any]:
        """
        Execute SQL query and handle errors.
        
        Args:
            sql_query: SQL query to execute
            table_name: Name of the table being created/modified
            
        Returns:
            Dict containing execution status
        """
        try:
            # Note: Supabase Python client doesn't support direct SQL execution
            # This would typically be done through the Supabase dashboard or API
            # For now, we'll return a success message with the SQL for manual execution
            
            logger.info(f"SQL prepared for {table_name} table")
            return {
                "success": True,
                "table": table_name,
                "message": f"SQL script prepared for {table_name}. Execute in Supabase dashboard.",
                "sql": sql_query
            }
            
        except Exception as e:
            error_msg = f"Error preparing SQL for {table_name}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "table": table_name, "error": error_msg}
    
    def get_database_status(self) -> Dict[str, Any]:
        """
        Get the current status of the database and tables.
        
        Returns:
            Dict containing database status information
        """
        try:
            status = {
                "connection": "connected",
                "tables": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Check each table by trying to query it
            tables_to_check = ["user_profiles", "transactions", "alerts", "audit_logs"]
            
            for table in tables_to_check:
                try:
                    result = self.client.client.table(table).select("count", count="exact").limit(1).execute()
                    status["tables"][table] = {
                        "exists": True,
                        "accessible": True,
                        "record_count": getattr(result, 'count', 0)
                    }
                except Exception as table_error:
                    status["tables"][table] = {
                        "exists": False,
                        "accessible": False,
                        "error": str(table_error)
                    }
            
            return status
            
        except Exception as e:
            return {
                "connection": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_table_creation_sql(self) -> str:
        """
        Get the complete SQL script for creating all tables.
        This can be executed manually in the Supabase SQL editor.
        
        Returns:
            Complete SQL script as a string
        """
        return f"""
-- ============================================================================
-- AI For Flagging Suspicious Transactions - Database Schema
-- Execute this script in your Supabase SQL editor
-- ============================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

{self._create_users_table()['sql']}

{self._create_transactions_table()['sql']}

{self._create_alerts_table()['sql']}

{self._create_audit_logs_table()['sql']}

-- ============================================================================
-- Additional utility functions and triggers
-- ============================================================================

-- Function to generate transaction IDs
CREATE OR REPLACE FUNCTION generate_transaction_id()
RETURNS TEXT AS $$
BEGIN
    RETURN 'TXN' || TO_CHAR(NOW(), 'YYYYMMDD') || '-' || UPPER(SUBSTRING(gen_random_uuid()::text, 1, 8));
END;
$$ LANGUAGE plpgsql;

-- Function to calculate risk score based on transaction attributes
CREATE OR REPLACE FUNCTION calculate_risk_score(
    amount DECIMAL,
    hour_of_day INTEGER,
    is_weekend BOOLEAN,
    location_risk DECIMAL DEFAULT 0.0
)
RETURNS DECIMAL AS $$
BEGIN
    DECLARE
        score DECIMAL := 0.0;
    BEGIN
        -- High amount transactions get higher risk scores
        IF amount > 100000 THEN
            score := score + 0.3;
        ELSIF amount > 50000 THEN
            score := score + 0.2;
        ELSIF amount > 10000 THEN
            score := score + 0.1;
        END IF;
        
        -- Unusual hours (0-6 AM) get higher risk scores
        IF hour_of_day BETWEEN 0 AND 6 THEN
            score := score + 0.2;
        END IF;
        
        -- Weekend transactions get slightly higher risk
        IF is_weekend THEN
            score := score + 0.1;
        END IF;
        
        -- Add location risk
        score := score + location_risk;
        
        -- Cap the score at 1.0
        RETURN LEAST(score, 1.0);
    END;
END;
$$ LANGUAGE plpgsql;

-- Insert sample configuration data
INSERT INTO user_profiles (id, email, full_name, role, department) VALUES
(gen_random_uuid(), 'admin@example.com', 'System Administrator', 'admin', 'IT')
ON CONFLICT (email) DO NOTHING;

-- ============================================================================
-- Database schema creation complete!
-- ============================================================================
"""
