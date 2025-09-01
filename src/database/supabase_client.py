"""Enhanced Supabase client for database operations and user management."""

from typing import Dict, List, Optional, Any, Tuple
import os
import hashlib
import secrets
from datetime import datetime, timezone, timedelta

try:
    from supabase import create_client, Client
    import bcrypt
except ImportError:
    # Fallback for when supabase is not installed
    Client = None
    bcrypt = None

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SupabaseClient:
    """Enhanced client for interacting with Supabase database and authentication."""
    
    def __init__(self):
        """Initialize the Supabase client with enhanced security features."""
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        if Client is None:
            raise ImportError("supabase-py is not installed. Run: pip install supabase")
        
        if bcrypt is None:
            raise ImportError("bcrypt is not installed. Run: pip install bcrypt")
        
        self.client = create_client(self.url, self.key)
        logger.info("Supabase client initialized successfully")
    
    # === USER MANAGEMENT METHODS ===
    
    def _hash_password(self, password: str) -> str:
        """Securely hash a password using bcrypt."""
        # Generate a random salt and hash the password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def register_user(self, email: str, password: str, full_name: str, 
                     role: str = "analyst", department: str = None) -> Dict[str, Any]:
        """
        Register a new user with secure password hashing.
        
        Args:
            email: User's email address
            password: Plain text password (will be hashed)
            full_name: User's full name
            role: User role (analyst, admin, viewer)
            department: User's department
            
        Returns:
            Dict containing success status and user data or error message
        """
        try:
            # Validate password strength
            if len(password) < 8:
                return {"success": False, "error": "Password must be at least 8 characters long"}
            
            # Check if user already exists
            existing_user = self._get_user_by_email(email)
            if existing_user:
                return {"success": False, "error": "User with this email already exists"}
            
            # Hash the password securely
            hashed_password = self._hash_password(password)
            
            # Create user in Supabase Auth
            auth_result = self.client.auth.sign_up({
                "email": email,
                "password": password,  # Supabase handles its own hashing for auth
                "options": {
                    "data": {
                        "full_name": full_name,
                        "role": role,
                        "department": department
                    }
                }
            })
            
            if auth_result.user:
                # Create user profile with additional data
                profile_data = {
                    "id": auth_result.user.id,
                    "email": email,
                    "full_name": full_name,
                    "role": role,
                    "department": department,
                    "password_hash": hashed_password,  # Store our own hash for additional security
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "is_active": True
                }
                
                profile_result = self.client.table("user_profiles").insert(profile_data).execute()
                
                logger.info(f"User registered successfully: {email}")
                return {
                    "success": True,
                    "user": auth_result.user,
                    "profile": profile_result.data[0] if profile_result.data else None,
                    "message": "User registered successfully"
                }
            else:
                return {"success": False, "error": "Failed to create user in authentication system"}
                
        except Exception as e:
            error_msg = f"Error registering user: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate a user with email and password.
        
        Args:
            email: User's email address
            password: User's password
            
        Returns:
            Dict containing success status and user data or error message
        """
        try:
            # Authenticate with Supabase Auth
            auth_result = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if auth_result.user and auth_result.session:
                # Get user profile data
                profile_result = self.client.table("user_profiles").select("*").eq("id", auth_result.user.id).execute()
                
                profile = profile_result.data[0] if profile_result.data else None
                
                # Update last login timestamp
                if profile:
                    self.client.table("user_profiles").update({
                        "last_login": datetime.now(timezone.utc).isoformat()
                    }).eq("id", auth_result.user.id).execute()
                
                logger.info(f"User logged in successfully: {email}")
                return {
                    "success": True,
                    "user": auth_result.user,
                    "session": auth_result.session,
                    "profile": profile,
                    "message": "Login successful"
                }
            else:
                return {"success": False, "error": "Invalid email or password"}
                
        except Exception as e:
            error_msg = f"Error during login: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user profile by email address."""
        try:
            result = self.client.table("user_profiles").select("*").eq("email", email).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error fetching user by email: {e}")
            return None
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Fetch user profile data by user ID.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            Dict containing user profile data or error message
        """
        try:
            result = self.client.table("user_profiles").select("*").eq("id", user_id).execute()
            
            if result.data:
                user_data = result.data[0]
                # Remove sensitive data before returning
                user_data.pop("password_hash", None)
                
                return {"success": True, "user": user_data}
            else:
                return {"success": False, "error": "User not found"}
                
        except Exception as e:
            error_msg = f"Error fetching user data: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user profile information.
        
        Args:
            user_id: User's unique identifier
            updates: Dictionary of fields to update
            
        Returns:
            Dict containing success status and updated data
        """
        try:
            # Remove sensitive fields that shouldn't be updated directly
            forbidden_fields = ["id", "password_hash", "created_at"]
            clean_updates = {k: v for k, v in updates.items() if k not in forbidden_fields}
            clean_updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            result = self.client.table("user_profiles").update(clean_updates).eq("id", user_id).execute()
            
            if result.data:
                logger.info(f"User profile updated successfully: {user_id}")
                return {"success": True, "user": result.data[0]}
            else:
                return {"success": False, "error": "Failed to update user profile"}
                
        except Exception as e:
            error_msg = f"Error updating user profile: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    
    # === TRANSACTION MANAGEMENT METHODS ===
    
    def insert_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a new transaction record with validation.
        
        Args:
            transaction_data: Dictionary containing transaction information
            
        Returns:
            Dict containing success status and transaction data
        """
        try:
            # Add metadata
            transaction_data["created_at"] = datetime.now(timezone.utc).isoformat()
            transaction_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # Validate required fields
            required_fields = ["transaction_id", "customer_id", "transaction_amount_inr", 
                             "transaction_date", "payment_method"]
            missing_fields = [field for field in required_fields if field not in transaction_data]
            
            if missing_fields:
                return {
                    "success": False, 
                    "error": f"Missing required fields: {', '.join(missing_fields)}"
                }
            
            result = self.client.table("transactions").insert(transaction_data).execute()
            logger.info(f"Transaction inserted: {transaction_data.get('transaction_id')}")
            
            return {"success": True, "data": result.data}
        except Exception as e:
            error_msg = f"Error inserting transaction: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def insert_bulk_transactions(self, transactions: List[Dict[str, Any]], 
                               batch_size: int = 1000) -> Dict[str, Any]:
        """
        Insert multiple transactions in batches for efficiency.
        
        Args:
            transactions: List of transaction dictionaries
            batch_size: Number of transactions to insert per batch
            
        Returns:
            Dict containing success status and summary statistics
        """
        try:
            total_inserted = 0
            failed_insertions = []
            
            # Add metadata to all transactions
            current_time = datetime.now(timezone.utc).isoformat()
            for transaction in transactions:
                transaction["created_at"] = current_time
                transaction["updated_at"] = current_time
            
            # Process in batches
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i:i + batch_size]
                
                try:
                    result = self.client.table("transactions").insert(batch).execute()
                    total_inserted += len(result.data) if result.data else 0
                    logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} transactions")
                    
                except Exception as batch_error:
                    logger.error(f"Error inserting batch {i//batch_size + 1}: {batch_error}")
                    failed_insertions.extend(batch)
            
            return {
                "success": True,
                "total_processed": len(transactions),
                "total_inserted": total_inserted,
                "failed_count": len(failed_insertions),
                "failed_transactions": failed_insertions[:10]  # Return first 10 failed for debugging
            }
            
        except Exception as e:
            error_msg = f"Error in bulk insert: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_transactions(self, limit: int = 100, offset: int = 0, 
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve transaction records with optional filtering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            filters: Optional filters (e.g., {'is_suspicious': True})
            
        Returns:
            List of transaction dictionaries
        """
        try:
            query = self.client.table("transactions").select("*")
            
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            result = query.range(offset, offset + limit - 1).execute()
            return result.data
            
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return []
    
    def get_transaction_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive transaction statistics.
        
        Returns:
            Dict containing various transaction statistics
        """
        try:
            # Total transactions
            total_result = self.client.table("transactions").select("count", count="exact").execute()
            total_count = total_result.count if hasattr(total_result, 'count') else 0
            
            # Suspicious transactions
            suspicious_result = self.client.table("transactions").select("count", count="exact").eq("is_suspicious", True).execute()
            suspicious_count = suspicious_result.count if hasattr(suspicious_result, 'count') else 0
            
            # Recent transactions (last 24 hours)
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            recent_result = self.client.table("transactions").select("count", count="exact").gte("created_at", yesterday).execute()
            recent_count = recent_result.count if hasattr(recent_result, 'count') else 0
            
            return {
                "total_transactions": total_count,
                "suspicious_transactions": suspicious_count,
                "recent_transactions_24h": recent_count,
                "suspicious_rate": (suspicious_count / total_count * 100) if total_count > 0 else 0,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching transaction statistics: {e}")
            return {
                "total_transactions": 0,
                "suspicious_transactions": 0,
                "recent_transactions_24h": 0,
                "suspicious_rate": 0,
                "error": str(e)
            }
    
    def get_transactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve transaction records."""
        try:
            result = self.client.table("transactions").select("*").limit(limit).execute()
            return result.data
        except Exception as e:
            print(f"Error fetching transactions: {e}")
            return []
    
    def update_transaction_flag(self, transaction_id: str, is_flagged: bool, reason: Optional[str] = None) -> bool:
        """Update the flagged status of a transaction."""
        try:
            update_data = {
                "is_flagged": is_flagged,
                "flagged_at": datetime.utcnow().isoformat() if is_flagged else None,
                "flag_reason": reason
            }
            
            result = self.client.table("transactions").update(update_data).eq("id", transaction_id).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error updating transaction flag: {e}")
            return False
    
    def get_flagged_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all flagged transactions."""
        try:
            result = (
                self.client.table("transactions")
                .select("*")
                .eq("is_flagged", True)
                .limit(limit)
                .execute()
            )
            return result.data
        except Exception as e:
            print(f"Error fetching flagged transactions: {e}")
            return []
    
    def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate a user."""
        try:
            result = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return {"success": True, "user": result.user, "session": result.session}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_user(self, email: str, password: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new user account."""
        try:
            result = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {"data": metadata} if metadata else None
            })
            return {"success": True, "user": result.user}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Database schema creation scripts
def create_tables():
    """SQL scripts to create necessary tables in Supabase."""
    return """
    -- Transactions table
    CREATE TABLE IF NOT EXISTS transactions (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        transaction_id VARCHAR(255) UNIQUE NOT NULL,
        amount DECIMAL(15,2) NOT NULL,
        currency VARCHAR(3) DEFAULT 'INR',
        sender_account VARCHAR(255) NOT NULL,
        receiver_account VARCHAR(255) NOT NULL,
        transaction_type VARCHAR(50) NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
        location VARCHAR(255),
        merchant_category VARCHAR(100),
        device_id VARCHAR(255),
        ip_address INET,
        is_flagged BOOLEAN DEFAULT FALSE,
        flagged_at TIMESTAMP WITH TIME ZONE,
        flag_reason TEXT,
        anomaly_score DECIMAL(5,4),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Users table (extends Supabase auth.users)
    CREATE TABLE IF NOT EXISTS user_profiles (
        id UUID REFERENCES auth.users(id) PRIMARY KEY,
        full_name VARCHAR(255),
        role VARCHAR(50) DEFAULT 'analyst',
        department VARCHAR(100),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Alerts table
    CREATE TABLE IF NOT EXISTS alerts (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        transaction_id UUID REFERENCES transactions(id),
        alert_type VARCHAR(50) NOT NULL,
        severity VARCHAR(20) NOT NULL,
        message TEXT NOT NULL,
        acknowledged BOOLEAN DEFAULT FALSE,
        acknowledged_by UUID REFERENCES auth.users(id),
        acknowledged_at TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_transactions_flagged ON transactions(is_flagged);
    CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);
    CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
    """
