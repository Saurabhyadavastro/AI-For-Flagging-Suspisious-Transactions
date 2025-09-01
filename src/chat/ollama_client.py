"""Ollama client for LLM-powered chat functionality."""

import json
import requests
from typing import Dict, List, Optional, Any
import os


class OllamaClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.1"):
        """Initialize Ollama client."""
        self.host = host.rstrip('/')
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
    
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except requests.RequestException:
            return []
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Generate a response using Ollama."""
        try:
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt(prompt, context)
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": prompt
                })
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": result.get('response', '')
                })
                
                return {
                    "success": True,
                    "response": result.get('response', ''),
                    "model": result.get('model', self.model),
                    "total_duration": result.get('total_duration', 0),
                    "load_duration": result.get('load_duration', 0),
                    "prompt_eval_count": result.get('prompt_eval_count', 0),
                    "eval_count": result.get('eval_count', 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _prepare_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Prepare the prompt with system context and conversation history."""
        system_prompt = """You are an expert financial analyst specializing in fraud detection and transaction analysis for Indian banking systems. You help users understand suspicious transaction patterns, regulatory compliance, and anomaly detection results.

Key responsibilities:
- Explain anomaly detection results in simple terms
- Provide insights about transaction patterns
- Suggest investigation steps for flagged transactions
- Discuss Indian banking regulations and compliance
- Help interpret machine learning model outputs

Always be precise, helpful, and focus on actionable insights."""
        
        # Build conversation context
        conversation_context = ""
        if self.conversation_history:
            for msg in self.conversation_history[-6:]:  # Last 6 messages for context
                role = msg['role']
                content = msg['content']
                conversation_context += f"\n{role.title()}: {content}"
        
        # Add transaction context if provided
        if context:
            context_section = f"\n\nTransaction Context:\n{context}"
        else:
            context_section = ""
        
        # Combine everything
        full_prompt = f"{system_prompt}\n\nConversation History:{conversation_context}{context_section}\n\nUser: {prompt}\n\nAssistant:"
        
        return full_prompt
    
    def explain_anomaly(self, transaction_data: Dict[str, Any], anomaly_score: float) -> Dict[str, Any]:
        """Explain why a transaction was flagged as anomalous."""
        # Format transaction data for the LLM
        transaction_summary = self._format_transaction_data(transaction_data)
        
        prompt = f"""Analyze this flagged transaction and explain why it might be suspicious:

Transaction Details:
{transaction_summary}

Anomaly Score: {anomaly_score:.4f} (higher scores indicate more suspicious activity)

Please provide:
1. Possible reasons why this transaction was flagged
2. What aspects of the transaction are unusual
3. Recommended investigation steps
4. Risk level assessment (Low/Medium/High)

Keep the explanation clear and actionable for a financial analyst."""
        
        return self.generate_response(prompt, transaction_summary)
    
    def analyze_pattern(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in multiple transactions."""
        pattern_summary = self._format_transaction_pattern(transactions)
        
        prompt = f"""Analyze this pattern of transactions for potential fraud indicators:

Transaction Pattern:
{pattern_summary}

Please identify:
1. Suspicious patterns or trends
2. Potential fraud schemes
3. Regulatory concerns
4. Recommended monitoring strategies

Focus on patterns that might indicate money laundering, structuring, or other financial crimes."""
        
        return self.generate_response(prompt, pattern_summary)
    
    def _format_transaction_data(self, transaction: Dict[str, Any]) -> str:
        """Format transaction data for LLM analysis."""
        formatted = []
        
        for key, value in transaction.items():
            if key in ['amount', 'timestamp', 'sender_account', 'receiver_account', 
                      'transaction_type', 'location', 'merchant_category']:
                formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)
    
    def _format_transaction_pattern(self, transactions: List[Dict[str, Any]]) -> str:
        """Format multiple transactions for pattern analysis."""
        if not transactions:
            return "No transactions provided"
        
        summary = f"Number of transactions: {len(transactions)}\n"
        
        # Calculate basic statistics
        amounts = [t.get('amount', 0) for t in transactions]
        if amounts:
            summary += f"Amount range: ₹{min(amounts):,.2f} - ₹{max(amounts):,.2f}\n"
            summary += f"Average amount: ₹{sum(amounts)/len(amounts):,.2f}\n"
        
        # Show first few transactions as examples
        summary += "\nSample transactions:\n"
        for i, txn in enumerate(transactions[:3]):
            summary += f"{i+1}. Amount: ₹{txn.get('amount', 0):,.2f}, "
            summary += f"Type: {txn.get('transaction_type', 'Unknown')}, "
            summary += f"Time: {txn.get('timestamp', 'Unknown')}\n"
        
        if len(transactions) > 3:
            summary += f"... and {len(transactions) - 3} more transactions"
        
        return summary
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
