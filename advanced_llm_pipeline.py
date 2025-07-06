# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
import json
import uuid
from datetime import datetime
import requests
from sentence_transformers import SentenceTransformer

# ======================
# 1. CONFIGURATION
# ======================

# Set your OpenAI API key here or as an environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

# ======================
# 2. BEHAVIORAL SCRIPTS (DEFs)
# ======================

# Define named DEFs (behavioral scripts) as dictionaries
DEFS = {
    "creative_brainstorm": {
        "system_prompt": "You are a creative brainstorming assistant. Generate innovative and out-of-the-box ideas.",
        "temperature": 0.9,
        "max_tokens": 1024,
        "safety_filter": False
    },
    "technical_analysis": {
        "system_prompt": "You are a technical expert. Provide detailed, structured, and analytical responses.",
        "temperature": 0.3,
        "max_tokens": 512,
        "safety_filter": True
    },
    "empathetic_counselor": {
        "system_prompt": "You are an empathetic counselor. Respond with compassion, understanding, and emotional intelligence.",
        "temperature": 0.7,
        "max_tokens": 512,
        "safety_filter": True
    }
}

# ======================
# 3. MODULES
# ======================

class EmbeddingGenerator:
    """Generates embeddings for text using SentenceTransformers"""
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate(self, text: str):
        return self.model.encode(text).tolist()


class IntentClassifier:
    """Simple intent classifier based on keyword matching"""
    def classify(self, query: str) -> str:
        if "brainstorm" in query.lower():
            return "creative_brainstorm"
        elif "technical" in query.lower() or "analysis" in query.lower():
            return "technical_analysis"
        elif "feel" in query.lower() or "emotion" in query.lower():
            return "empathetic_counselor"
        else:
            return "technical_analysis"  # default


class ConstraintManager:
    """Manages active constraints based on DEFs"""
    def __init__(self, defs: dict):
        self.defs = defs
    
    def get_constraints(self, def_name: str) -> dict:
        return self.defs.get(def_name, {
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.7,
            "max_tokens": 512,
            "safety_filter": True
        })


class LLMClient:
    """Handles API calls to OpenAI"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def generate(self, messages, temperature=0.7, max_tokens=512):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4-turbo",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class TelemetryLogger:
    """Logs telemetry data to a local file"""
    def __init__(self, log_file="telemetry.log"):
        self.log_file = log_file
    
    def log(self, data: dict):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")


# ======================
# 4. ORCHESTRATOR
# ======================

class Orchestrator:
    def __init__(self, api_key):
        self.embedder = EmbeddingGenerator()
        self.intent_classifier = IntentClassifier()
        self.constraint_mgr = ConstraintManager(DEFS)
        self.llm_client = LLMClient(api_key)
        self.telemetry_logger = TelemetryLogger()
        
        # State tracking
        self.session_id = str(uuid.uuid4())
        
    def process_query(self, user_query: str) -> str:
        """Process a user query through the full pipeline"""
        # Generate embedding
        embedding = self.embedder.generate(user_query)
        
        # Classify intent
        def_name = self.intent_classifier.classify(user_query)
        
        # Get constraints
        constraints = self.constraint_mgr.get_constraints(def_name)
        
        # Build messages
        messages = [
            {"role": "system", "content": constraints["system_prompt"]},
            {"role": "user", "content": user_query}
        ]
        
        # Generate response
        response = self.llm_client.generate(
            messages,
            temperature=constraints["temperature"],
            max_tokens=constraints["max_tokens"]
        )
        
        # Apply safety filters if enabled
        # (In a real system, this would be a separate module)
        if constraints["safety_filter"]:
            # Placeholder for safety filtering
            if "harm" in response.lower():
                response = "I cannot provide a response to that question."
        
        # Log telemetry
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "user_query": user_query,
            "intent": def_name,
            "response": response,
            "embedding": embedding
        }
        self.telemetry_logger.log(log_data)
        
        return response


# ======================
# 5. DEMONSTRATION
# ======================

if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = Orchestrator(api_key=OPENAI_API_KEY)
    
    # Example user query
    user_query = "How can I improve my team's performance using predictive analytics?"
    
    # Process query
    response = orchestrator.process_query(user_query)
    
    # Print results
    print(f"User Query: {user_query}")
    print(f"LLM Response: {response}")
    print("Telemetry logged successfully.")
