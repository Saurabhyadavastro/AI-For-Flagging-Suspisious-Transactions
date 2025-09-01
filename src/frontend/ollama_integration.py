"""
Ollama Integration Module for AI-Powered Chat Functionality
Enhanced with intelligent fallback responses for better user experience
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Try to import pandas, fall back gracefully if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ollama, fall back gracefully if not available
try:
    import ollama
    OLLAMA_AVAILABLE = True
    logger.info("Ollama module imported successfully")
except ImportError as e:
    OLLAMA_AVAILABLE = False
    logger.warning(f"Ollama module not available: {str(e)}")
    ollama = None

class OllamaChat:
    """
    Intelligent chat assistant using local Llama 3.1 model via Ollama
    """
    
    def __init__(self, model_name: str = "llama3.1"):
        """
        Initialize Ollama chat with specified model
        
        Args:
            model_name: Name of the Ollama model to use (default: llama3.1)
        """
        self.model_name = model_name
        self.is_available = self._check_ollama_availability()
        self.conversation_history = []
        
    def _check_ollama_availability(self) -> bool:
        """
        Check if Ollama service is running and model is available
        
        Returns:
            bool: True if Ollama is available and model can be used
        """
        if not OLLAMA_AVAILABLE:
            return False
            
        try:
            # Try to list available models to test connection
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            # Check if our preferred model or any llama model is available
            if self.model_name in available_models:
                return True
            
            # Look for any llama model as fallback
            llama_models = [m for m in available_models if 'llama' in m.lower()]
            if llama_models:
                self.model_name = llama_models[0]  # Use the first available llama model
                logger.info(f"Using available model: {self.model_name}")
                return True
                
            logger.warning(f"No suitable model found. Available: {available_models}")
            return False
            
        except Exception as e:
            logger.error(f"Ollama service not available: {str(e)}")
            return False
    
    def check_ollama_status(self) -> bool:
        """
        Check current Ollama status
        
        Returns:
            bool: True if Ollama is available
        """
        return self.is_available

    def chat_with_context(self, user_question: str, context_data: Any = None) -> str:
        """
        Chat with context data providing intelligent responses
        
        Args:
            user_question: The user's question
            context_data: Context data (transactions, analysis results, etc.)
            
        Returns:
            str: AI response (from Ollama or intelligent fallback)
        """
        if not OLLAMA_AVAILABLE or not self.is_available:
            return self._get_smart_fallback_response(user_question, context_data)
        
        try:
            # Create context-aware prompt for Ollama
            prompt = self._create_context_prompt(user_question, context_data)
            
            # Generate response using Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.3,  # Lower temperature for more focused responses
                    'top_p': 0.9,
                    'max_tokens': 1000
                }
            )
            
            # Extract the response content
            ai_response = response['message']['content'].strip()
            
            # Add to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_question': user_question,
                'ai_response': ai_response,
                'context_available': context_data is not None
            })
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating Ollama response: {str(e)}")
            return self._get_smart_fallback_response(user_question, context_data)
    
    def _create_context_prompt(self, user_question: str, context_data: Any = None) -> str:
        """
        Create context-aware prompt for Ollama
        
        Args:
            user_question: User's question
            context_data: Available context data
            
        Returns:
            str: Formatted prompt for Ollama
        """
        base_prompt = f"""You are an expert AI assistant specializing in financial transaction analysis and fraud detection for suspicious transactions.

User Question: {user_question}

Context: """
        
        if context_data and isinstance(context_data, list) and len(context_data) > 0:
            # Summarize transaction data for context
            total_transactions = len(context_data)
            high_risk_count = sum(1 for t in context_data if t.get('Anomaly_Flag') == 'High Risk')
            
            base_prompt += f"""
Transaction Data Summary:
- Total transactions: {total_transactions}
- High-risk transactions: {high_risk_count}
- Risk detection rate: {high_risk_count/total_transactions*100:.1f}%

Sample transaction structure: {context_data[0] if context_data else 'No data'}

Please provide a detailed, professional analysis addressing the user's question based on this transaction data."""
        else:
            base_prompt += "No specific transaction data provided. Please provide general guidance about transaction monitoring and fraud detection."
        
        return base_prompt
    
    def _get_smart_fallback_response(self, query: str, context_data: Any = None) -> str:
        """
        Provide intelligent fallback responses when Ollama is not available
        """
        query_lower = query.lower()
        
        # Enhanced context-aware responses based on actual data
        if context_data and isinstance(context_data, list) and len(context_data) > 0:
            
            # Risk explanation queries
            if any(word in query_lower for word in ['why', 'reason', 'flag', 'suspicious', 'explain']):
                high_risk_transactions = [t for t in context_data if t.get('Anomaly_Flag') == 'High Risk']
                
                if high_risk_transactions:
                    try:
                        large_amounts = sum(1 for t in high_risk_transactions if float(t.get('Transaction_Amount', 0)) > 500000)
                        mobile_transactions = sum(1 for t in high_risk_transactions if t.get('Device_Type') == 'Mobile')
                        transfers = sum(1 for t in high_risk_transactions if t.get('Transaction_Type') == 'Transfer')
                        
                        response = "🔍 **High-Risk Transaction Analysis:**\\n\\n"
                        response += "**Risk Factors Identified:**\\n"
                        response += f"• **Large Amounts**: {large_amounts}/{len(high_risk_transactions)} transactions >₹5 lakh\\n"
                        response += f"• **Mobile Devices**: {mobile_transactions}/{len(high_risk_transactions)} from mobile platforms\\n"
                        response += f"• **Transfer Type**: {transfers}/{len(high_risk_transactions)} are transfer transactions\\n\\n"
                        response += "**Risk Scoring Logic:**\\n"
                        response += "• Amount threshold violations (>₹500K = +0.4 risk)\\n"
                        response += "• High-risk location patterns (+0.2 risk)\\n"
                        response += "• Mobile/ATM device usage (+0.1-0.15 risk)\\n"
                        response += "• Transfer transaction type (+0.2 risk)\\n"
                        response += "• Frequency and timing anomalies\\n\\n"
                        response += "🎯 **Recommendation**: Review transactions with risk scores >0.7 immediately."
                        return response
                    except Exception:
                        pass
            
            # Statistics queries  
            elif any(word in query_lower for word in ['how many', 'count', 'number', 'statistics', 'stats']):
                try:
                    total = len(context_data)
                    high_risk = sum(1 for t in context_data if t.get('Anomaly_Flag') == 'High Risk')
                    medium_risk = sum(1 for t in context_data if t.get('Anomaly_Flag') == 'Medium Risk')
                    low_risk = sum(1 for t in context_data if t.get('Anomaly_Flag') == 'Low Risk')
                    
                    total_amount = sum(float(t.get('Transaction_Amount', 0)) for t in context_data)
                    high_risk_amount = sum(float(t.get('Transaction_Amount', 0)) for t in context_data if t.get('Anomaly_Flag') == 'High Risk')
                    
                    response = "📊 **Transaction Statistics:**\\n\\n"
                    response += "**Risk Distribution:**\\n"
                    response += f"• 🔴 High Risk: {high_risk} transactions ({high_risk/total*100:.1f}%)\\n"
                    response += f"• 🟡 Medium Risk: {medium_risk} transactions ({medium_risk/total*100:.1f}%)\\n"
                    response += f"• 🟢 Low Risk: {low_risk} transactions ({low_risk/total*100:.1f}%)\\n\\n"
                    response += f"**Financial Impact:**\\n"
                    response += f"• Total Volume: ₹{total_amount:,.0f}\\n"
                    response += f"• High-Risk Volume: ₹{high_risk_amount:,.0f} ({high_risk_amount/total_amount*100:.1f}%)\\n\\n"
                    response += f"🎯 **Detection Rate**: {high_risk/total*100:.1f}% indicates moderate suspicious activity."
                    return response
                except Exception:
                    pass
            
            # Geographic analysis
            elif any(word in query_lower for word in ['location', 'geographic', 'area', 'region', 'city']):
                try:
                    locations = {}
                    for t in context_data:
                        loc = t.get('Location', 'Unknown')
                        risk = t.get('Anomaly_Flag', 'Unknown')
                        
                        if loc not in locations:
                            locations[loc] = {'total': 0, 'high_risk': 0}
                        
                        locations[loc]['total'] += 1
                        if risk == 'High Risk':
                            locations[loc]['high_risk'] += 1
                    
                    response = "🌍 **Geographic Risk Analysis:**\\n\\n"
                    sorted_locations = sorted(locations.items(), key=lambda x: x[1]['high_risk'], reverse=True)
                    
                    for loc, data in sorted_locations[:5]:  # Top 5 locations
                        if data['total'] > 0:
                            rate = (data['high_risk'] / data['total'] * 100)
                            risk_icon = "🔴" if rate > 20 else "🟡" if rate > 10 else "🟢"
                            response += f"{risk_icon} **{loc}**: {data['high_risk']}/{data['total']} suspicious ({rate:.1f}%)\\n"
                    
                    response += "\\n💡 **Geographic Insights:**\\n"
                    response += "• Metropolitan areas show higher transaction volumes\\n"
                    response += "• Risk concentration varies by financial district"
                    return response
                except Exception:
                    pass
            
            # Amount analysis
            elif any(word in query_lower for word in ['amount', 'value', 'money', 'rupee', 'financial', 'anomal']):
                try:
                    amounts = [float(t.get('Transaction_Amount', 0)) for t in context_data if t.get('Transaction_Amount')]
                    high_risk_amounts = [float(t.get('Transaction_Amount', 0)) for t in context_data 
                                       if t.get('Anomaly_Flag') == 'High Risk' and t.get('Transaction_Amount')]
                    
                    if amounts:
                        avg_all = sum(amounts) / len(amounts)
                        max_amount = max(amounts)
                        min_amount = min(amounts)
                        
                        large_count = sum(1 for a in amounts if a > 500000)
                        medium_count = sum(1 for a in amounts if 100000 <= a <= 500000)
                        small_count = sum(1 for a in amounts if a < 100000)
                        
                        response = "💰 **Transaction Amount Analysis:**\\n\\n"
                        response += "**Amount Distribution:**\\n"
                        response += f"• Large (>₹5L): {large_count} transactions\\n"
                        response += f"• Medium (₹1L-₹5L): {medium_count} transactions\\n"
                        response += f"• Small (<₹1L): {small_count} transactions\\n\\n"
                        response += f"**Statistical Summary:**\\n"
                        response += f"• Average Amount: ₹{avg_all:,.0f}\\n"
                        response += f"• Range: ₹{min_amount:,.0f} - ₹{max_amount:,.0f}\\n"
                        
                        if high_risk_amounts:
                            avg_high_risk = sum(high_risk_amounts) / len(high_risk_amounts)
                            response += f"• High-Risk Average: ₹{avg_high_risk:,.0f}\\n"
                            response += f"• Risk Multiplier: {avg_high_risk/avg_all:.1f}x higher"
                        
                        return response
                except Exception:
                    pass
        
        # Generic helpful responses
        if any(word in query_lower for word in ['help', 'what', 'how', 'commands']):
            return ("🤖 **AI Transaction Analyst - Help**\\n\\n"
                   "**I can analyze:**\\n"
                   "💰 Amount patterns and financial anomalies\\n"
                   "🌍 Geographic risk distributions\\n"
                   "📊 Statistical summaries and trends\\n"
                   "🔍 Risk factor explanations\\n\\n"
                   "**To get started:**\\n"
                   "1. Upload your transaction data\\n"
                   "2. Ask specific questions about patterns\\n"
                   "3. Review interactive charts and metrics\\n\\n"
                   "💡 **Tip**: Upload data first for personalized analysis!")
        
        # Default response encouraging data upload
        return ("🤖 **AI Transaction Analyst (Smart Mode)**\\n\\n"
               "I'm ready to provide intelligent analysis of your transaction data!\\n\\n"
               "**Current Status:**\\n"
               "• Local AI (Ollama): Offline - using smart fallbacks\\n"
               "• Analysis Engine: Active and ready\\n"
               "• Risk Assessment: Advanced algorithms enabled\\n\\n"
               "**For detailed analysis:**\\n"
               "1. Upload transaction data using the sidebar\\n"
               "2. Ask specific questions about patterns or risks\\n"
               "3. Review interactive visualizations\\n\\n"
               "**I can analyze**: Risk patterns, geographic trends, amount distributions, device patterns\\n\\n"
               "What would you like to explore?")

# Helper functions for transaction analysis
def extract_transaction_id_from_query(query: str) -> Optional[str]:
    """
    Extract transaction ID from user query
    
    Args:
        query: User input string
        
    Returns:
        Optional[str]: Transaction ID if found
    """
    import re
    
    # Look for patterns like T001, TXN123, etc.
    patterns = [
        r'T\d+',
        r'TXN\d+',
        r'TRANS\d+',
        r'ID[:\s]+([A-Z0-9]+)',
        r'transaction[:\s]+([A-Z0-9]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    
    return None

def format_transaction_details(transaction: Dict) -> str:
    """
    Format transaction details for display
    
    Args:
        transaction: Transaction dictionary
        
    Returns:
        str: Formatted transaction details
    """
    details = f"**Transaction Details:**\\n"
    details += f"• ID: {transaction.get('Transaction_ID', 'N/A')}\\n"
    details += f"• Amount: ₹{float(transaction.get('Transaction_Amount', 0)):,.0f}\\n"
    details += f"• Type: {transaction.get('Transaction_Type', 'N/A')}\\n"
    details += f"• Location: {transaction.get('Location', 'N/A')}\\n"
    details += f"• Device: {transaction.get('Device_Type', 'N/A')}\\n"
    details += f"• Risk Score: {float(transaction.get('Risk_Score', 0)):.2f}\\n"
    details += f"• Status: {transaction.get('Anomaly_Flag', 'N/A')}\\n"
    
    if transaction.get('Description'):
        details += f"• Description: {transaction['Description']}\\n"
    
    return details
