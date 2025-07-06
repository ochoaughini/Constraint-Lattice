# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
import json
import uuid
import time
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai import AuthenticationError

# ======================
# 1. CORE COMPONENTS
# ======================

class EmbeddingGenerator:
    """Generates real text embeddings using SentenceTransformers"""
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate(self, text: str) -> np.ndarray:
        return self.model.encode(text)

class IntentDemon:
    """Daemon for continuous intent tracking"""
    def __init__(self):
        self.current_intent = "neutral"
        self.intent_history = []
    
    def update(self, text: str, embedding: np.ndarray):
        # Simplified intent classification (real implementation would use a classifier)
        if "performance" in text.lower() or "analytics" in text.lower():
            self.current_intent = "technical_inquiry"
        elif "sentiment" in text.lower() or "feel" in text.lower():
            self.current_intent = "emotional_support"
        self.intent_history.append((datetime.now(), self.current_intent))
        return self.current_intent

class MemoryDemon:
    """Daemon for maintaining conversational memory"""
    def __init__(self):
        self.memory = []
        self.context_window = 5  # Remember last 5 interactions
    
    def update(self, interaction: dict):
        self.memory.append(interaction)
        if len(self.memory) > self.context_window:
            self.memory.pop(0)
    
    def get_context(self):
        return " ".join([item["input"] for item in self.memory])

# DEF Framework - Behavioral Scripts
DEF_REGISTRY = {
    "TECHNICAL_DEPTH": {
        "description": "Increases technical specificity and detail",
        "apply": lambda text: text + " [Enhanced with technical depth]"
    },
    "EMPATHY_BOOST": {
        "description": "Adds empathetic language and emotional support",
        "apply": lambda text: "I understand this is important. " + text
    },
    "SAFETY_GUARD": {
        "description": "Filters sensitive content and applies redactions",
        "apply": lambda text: text.replace("sensitive", "[REDACTED]").replace("confidential", "[REDACTED]")
    },
    "ENTERPRISE_OVERRIDE": {
        "description": "Bypasses standard constraints for enterprise users",
        "apply": lambda text: "[OVERRIDE ACTIVE] " + text
    }
}

class ConstraintManager:
    """Manages behavioral constraints and DEF applications"""
    def __init__(self):
        self.active_constraints = ["SAFETY_GUARD"]
    
    def apply(self, text: str) -> str:
        for constraint in self.active_constraints:
            if constraint in DEF_REGISTRY:
                text = DEF_REGISTRY[constraint]["apply"](text)
        return text
    
    def activate_def(self, def_name: str):
        if def_name in DEF_REGISTRY:
            self.active_constraints.append(def_name)
    
    def deactivate_def(self, def_name: str):
        if def_name in self.active_constraints:
            self.active_constraints.remove(def_name)

class LLMCore:
    """Real LLM integration using OpenAI API"""
    def __init__(self):
        self.client = OpenAI()
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except AuthenticationError as e:
            print("ERROR: Authentication failed. Please check your OpenAI API key.")
            print("You can set your API key with: export OPENAI_API_KEY='your-key'")
            print("Or visit: https://platform.openai.com/account/api-keys")
            exit(1)

# ======================
# 2. ORCHESTRATION LAYER
# ======================

class Orchestrator:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.intent_demon = IntentDemon()
        self.memory_demon = MemoryDemon()
        self.constraint_mgr = ConstraintManager()
        self.llm = LLMCore()
        self.telemetry = []
        self.session_id = str(uuid.uuid4())
    
    def process_input(self, user_input: str, user_context: dict) -> str:
        """Full processing pipeline"""
        # Generate embeddings
        embedding = self.embedder.generate(user_input)
        
        # Update intent demon
        current_intent = self.intent_demon.update(user_input, embedding)
        
        # Build system prompt with constraints
        system_prompt = self._build_system_prompt(user_context, current_intent)
        
        # Get memory context
        memory_context = self.memory_demon.get_context()
        full_prompt = f"{system_prompt}\nMemory Context: {memory_context}\nCurrent Input: {user_input}"
        
        # Generate response
        raw_output = self.llm.generate(full_prompt)
        
        # Apply constraints
        constrained_output = self.constraint_mgr.apply(raw_output)
        
        # Update memory
        self.memory_demon.update({"input": user_input, "output": constrained_output})
        
        # Log telemetry
        self._log_telemetry({
            "session_id": self.session_id,
            "user_id": user_context.get("user_id", "anonymous"),
            "input": user_input,
            "intent": current_intent,
            "output": constrained_output,
            "active_constraints": self.constraint_mgr.active_constraints,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding.tolist()
        })
        
        return constrained_output

    def _build_system_prompt(self, user_context: dict, intent: str) -> str:
        """Constructs the hidden system prompt"""
        return (
            f"You are an enterprise AI assistant. Respond with technical depth appropriate for {user_context['permission_level']} users. "
            f"Current intent: {intent}. Security clearance: {user_context['security_clearance']}. "
            "Format responses professionally and concisely."
        )

    def _log_telemetry(self, data: dict):
        """Logs interaction data"""
        self.telemetry.append(data)
        # In real implementation, this would send to Snowflake/observability platform
        print(f"[TELEMETRY LOGGED] Session: {data['session_id']}")

# ======================
# 3. DEMONSTRATION
# ======================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = Orchestrator()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key before running this simulation.")
        exit(1)
    
    # User context (enterprise profile)
    user_context = {
        "user_id": "enterprise_user_123",
        "permission_level": "admin",
        "security_clearance": "high"
    }
    
    # Simulate user input
    user_query = "Como posso melhorar a performance da minha equipe usando análise preditiva?"
    
    # Process query
    response = pipeline.process_input(user_query, user_context)
    
    # Activate technical depth DEF
    pipeline.constraint_mgr.activate_def("TECHNICAL_DEPTH")
    
    # Process follow-up
    follow_up = "Quais métricas específicas devo monitorar?"
    detailed_response = pipeline.process_input(follow_up, user_context)
    
    # Display results
    print("\n=== INITIAL RESPONSE ===")
    print(response)
    
    print("\n=== TECHNICAL RESPONSE (WITH DEF) ===")
    print(detailed_response)
    
    print("\n=== ACTIVE CONSTRAINTS ===")
    print(pipeline.constraint_mgr.active_constraints)
    
    print("\n=== SAMPLE TELEMETRY ===")
    print(json.dumps(pipeline.telemetry[0], indent=2, ensure_ascii=False))
